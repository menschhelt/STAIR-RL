"""
FinBERT Processor - Sentiment analysis for GDELT news articles.

Model: ProsusAI/finbert
Output: [Positive, Negative, Neutral] probabilities (3 classes)

FinBERT is trained on financial news and understands:
- Formal language: "decline", "rally", "regulation"
- Economic context: "inflation", "interest rates", "market sentiment"

This is used for GDELT data (professional news articles).
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path
import logging
from tqdm import tqdm

# Import transformers (lazy loading)
transformers = None


def _load_transformers():
    """Lazy load transformers to avoid import overhead."""
    global transformers
    if transformers is None:
        import transformers as tf
        transformers = tf
    return transformers


class FinBERTProcessor:
    """
    Processes GDELT news through FinBERT for financial sentiment.

    Output classes:
    - Positive (0): Positive financial sentiment
    - Negative (1): Negative financial sentiment
    - Neutral (2): Neutral/informational content

    Features:
    - GPU batch processing (A100 optimized)
    - Batch size 256-512 recommended
    - Returns probabilities, not just labels
    """

    MODEL_NAME = "ProsusAI/finbert"

    # Class mapping (FinBERT specific order)
    CLASS_NAMES = ['positive', 'negative', 'neutral']

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 256,
        max_length: int = 256,  # Longer for news articles
    ):
        """
        Initialize FinBERT processor.

        Args:
            device: 'cuda', 'cuda:0', 'cuda:1', or 'cpu'
            batch_size: Processing batch size (256-512 for A100)
            max_length: Maximum token length (256 for news)
        """
        self.batch_size = batch_size
        self.max_length = max_length

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.logger.info(f"Using device: {self.device}")

        # Model and tokenizer (lazy loading)
        self._model = None
        self._tokenizer = None

    def _setup_logging(self):
        """Configure logging."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is not None:
            return

        tf = _load_transformers()

        self.logger.info(f"Loading FinBERT model: {self.MODEL_NAME}")

        self._tokenizer = tf.AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self._model = tf.AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME
        )
        self._model.to(self.device)
        self._model.eval()

        self.logger.info(f"Model loaded on {self.device}")

    @property
    def model(self):
        """Get model (lazy loading)."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def tokenizer(self):
        """Get tokenizer (lazy loading)."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer

    def process_text(self, text: str) -> Dict[str, float]:
        """
        Process a single text through FinBERT.

        Args:
            text: Input text

        Returns:
            Dict with positive, negative, neutral probabilities
        """
        results = self.process_batch([text])
        return results[0]

    def process_batch(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[Dict[str, float]]:
        """
        Process multiple texts through FinBERT.

        Args:
            texts: List of input texts
            show_progress: Show progress bar

        Returns:
            List of dicts with positive, negative, neutral probabilities
        """
        if not texts:
            return []

        self._load_model()

        results = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="FinBERT")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]

                # Clean texts
                batch_texts = [self._clean_text(t) for t in batch_texts]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Forward pass
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                probs = probs.cpu().numpy()

                # Convert to dicts
                # FinBERT label order: positive, negative, neutral
                for prob in probs:
                    results.append({
                        'positive': float(prob[0]),
                        'negative': float(prob[1]),
                        'neutral': float(prob[2]),
                    })

        return results

    def _clean_text(self, text: str) -> str:
        """
        Clean text for processing.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Basic cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace

        # Limit length for efficiency
        if len(text) > 2000:
            text = text[:2000]

        return text

    def get_embeddings(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Extract BERT [CLS] embeddings from texts.

        This method returns the raw 768-dimensional embeddings
        from the [CLS] token, which can be used for downstream tasks
        like the attention-based feature encoder in STAIR-RL.

        Args:
            texts: List of input texts
            show_progress: Show progress bar

        Returns:
            np.ndarray of shape (len(texts), 768) - BERT embeddings
        """
        if not texts:
            return np.array([]).reshape(0, 768)

        self._load_model()

        # We need to access the base model for embeddings
        tf = _load_transformers()

        # Load base model if not already loaded
        if not hasattr(self, '_base_model'):
            self._base_model = tf.AutoModel.from_pretrained(self.MODEL_NAME)
            self._base_model.to(self.device)
            self._base_model.eval()

        embeddings = []
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="FinBERT Embeddings")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i + self.batch_size]

                # Clean texts
                batch_texts = [self._clean_text(t) for t in batch_texts]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get BERT embeddings (not classification output)
                outputs = self._base_model(**inputs)

                # Extract [CLS] token embedding (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
                embeddings.append(cls_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'content',
        output_prefix: str = 'finbert',
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Process a DataFrame and add sentiment columns.

        Args:
            df: Input DataFrame
            text_column: Column containing text
            output_prefix: Prefix for output columns
            show_progress: Show progress bar

        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty or text_column not in df.columns:
            return df

        texts = df[text_column].fillna('').tolist()
        results = self.process_batch(texts, show_progress=show_progress)

        # Add columns
        df[f'{output_prefix}_positive'] = [r['positive'] for r in results]
        df[f'{output_prefix}_negative'] = [r['negative'] for r in results]
        df[f'{output_prefix}_neutral'] = [r['neutral'] for r in results]

        return df


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Test the processor
    processor = FinBERTProcessor()

    test_texts = [
        "Bitcoin rallied 10% as institutional investors increased their holdings.",
        "Cryptocurrency market crashed amid regulatory concerns from the SEC.",
        "The Federal Reserve announced interest rate decisions today.",
        "Company reports strong quarterly earnings beating analyst expectations.",
        "Economic indicators suggest a potential recession ahead.",
    ]

    print("Testing FinBERT processor:\n")
    results = processor.process_batch(test_texts)

    for text, result in zip(test_texts, results):
        print(f"Text: {text[:60]}...")
        print(f"  Positive: {result['positive']:.3f}")
        print(f"  Negative: {result['negative']:.3f}")
        print(f"  Neutral:  {result['neutral']:.3f}")
        print()
