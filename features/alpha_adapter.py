"""
Alpha Adapter - Bridge between existing alpha code and local Parquet data.

Integrates with:
- BaseAlpha from customized-freqtrade/user_data/strategies/alphas/base/base.py
- CryptoFactorCalculator from base/factor_calculator.py
- Alpha 101 (103 alphas) and Alpha 191 (194 alphas, skip 158)

This adapter allows us to reuse all existing alpha calculations
with our local Parquet-based data pipeline.
"""

import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
import pandas as pd
import numpy as np
import logging

from config.settings import ALPHA_BASE_PATH, AlphaConfig


class AlphaAdapter:
    """
    Bridges existing alpha code with local Parquet data.

    Features:
    - Dynamic loading of Alpha 101/191 implementations
    - Integration with existing BaseAlpha operators
    - Batch calculation for efficiency
    - Missing comparison method patches (ge, le, gt)
    """

    def __init__(
        self,
        alphas_dir: Optional[Path] = None,
        config: Optional[AlphaConfig] = None,
    ):
        """
        Initialize Alpha Adapter.

        Args:
            alphas_dir: Path to alphas directory
            config: Alpha configuration
        """
        self.alphas_dir = alphas_dir or ALPHA_BASE_PATH
        self.config = config or AlphaConfig()

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

        # Add alphas directory to path for imports
        sys.path.insert(0, str(self.alphas_dir.parent.parent))

        # Storage for loaded alpha instances
        self.alpha_instances: Dict[str, Any] = {}
        self.alpha_classes: Dict[str, Type] = {}

        # Load base classes
        self._base_alpha_class = None
        self._factor_calculator_class = None
        self._load_base_classes()

        # Patch missing methods
        self._patch_base_alpha()

        # Load all alpha implementations
        self._load_alphas()

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

    def _load_base_classes(self):
        """Load BaseAlpha and CryptoFactorCalculator classes."""
        try:
            # Use standard import system to ensure class identity matches
            # Alpha files use: from alphas.base.base import BaseAlpha
            # So we must import the same way for issubclass() to work
            from alphas.base.base import BaseAlpha
            self._base_alpha_class = BaseAlpha
            self.logger.info("Loaded BaseAlpha via standard import")

            # Factor calculator is optional
            try:
                from alphas.base.factor_calculator import CryptoFactorCalculator
                self._factor_calculator_class = CryptoFactorCalculator
                self.logger.info("Loaded CryptoFactorCalculator via standard import")
            except ImportError:
                self._factor_calculator_class = None
                self.logger.warning("CryptoFactorCalculator not found - factor calculation disabled")

        except Exception as e:
            self.logger.error(f"Error loading base classes: {e}")
            raise

    def _patch_base_alpha(self):
        """
        Patch missing comparison methods in BaseAlpha.

        The existing BaseAlpha is missing ge(), le(), gt() methods
        that are used by some Alpha 191 implementations.
        """
        if self._base_alpha_class is None:
            return

        # Add missing methods
        def ge(self, x: pd.Series, y) -> pd.Series:
            """Greater than or equal comparison."""
            return x >= y

        def le(self, x: pd.Series, y) -> pd.Series:
            """Less than or equal comparison."""
            return x <= y

        def gt(self, x: pd.Series, y) -> pd.Series:
            """Greater than comparison."""
            return x > y

        def ne(self, x: pd.Series, y) -> pd.Series:
            """Not equal comparison."""
            return x != y

        # Patch the class
        if not hasattr(self._base_alpha_class, 'ge'):
            self._base_alpha_class.ge = ge
            self.logger.debug("Patched BaseAlpha.ge()")

        if not hasattr(self._base_alpha_class, 'le'):
            self._base_alpha_class.le = le
            self.logger.debug("Patched BaseAlpha.le()")

        if not hasattr(self._base_alpha_class, 'gt'):
            self._base_alpha_class.gt = gt
            self.logger.debug("Patched BaseAlpha.gt()")

        if not hasattr(self._base_alpha_class, 'ne'):
            self._base_alpha_class.ne = ne
            self.logger.debug("Patched BaseAlpha.ne()")

    def _load_alphas(self):
        """Dynamically load all alpha implementations."""
        # Load Alpha 101
        alpha_101_dir = self.alphas_dir / 'alpha_101'
        if alpha_101_dir.exists():
            self._load_alpha_directory(alpha_101_dir, 'alpha_101')

        # Load Alpha 191 - DISABLED (only using Alpha 101)
        # alpha_191_dir = self.alphas_dir / 'alpha_191'
        # if alpha_191_dir.exists():
        #     self._load_alpha_directory(alpha_191_dir, 'alpha_191')

        # Load custom alphas
        for py_file in self.alphas_dir.glob('*.py'):
            if py_file.stem not in ['__init__', 'alpha_strategy']:
                self._load_alpha_file(py_file, 'custom')

        self.logger.info(
            f"Loaded {len(self.alpha_classes)} alpha implementations"
        )

    def _load_alpha_directory(self, directory: Path, prefix: str):
        """
        Load all alpha files from a directory.

        Args:
            directory: Directory containing alpha files
            prefix: Prefix for alpha names
        """
        for py_file in sorted(directory.glob('*.py')):
            if py_file.stem.startswith('__'):
                continue

            # Skip excluded alphas
            alpha_name = f"{prefix}_{py_file.stem.split('_')[-1]}"
            if alpha_name in self.config.skip_alphas:
                self.logger.debug(f"Skipping excluded alpha: {alpha_name}")
                continue

            self._load_alpha_file(py_file, prefix)

    def _load_alpha_file(self, file_path: Path, prefix: str):
        """
        Load alpha class from a Python file.

        Args:
            file_path: Path to Python file
            prefix: Prefix for alpha name
        """
        try:
            module_name = f"{prefix}_{file_path.stem}"

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find alpha class in module
            for name, obj in vars(module).items():
                if (
                    isinstance(obj, type) and
                    self._base_alpha_class is not None and
                    issubclass(obj, self._base_alpha_class) and
                    obj is not self._base_alpha_class
                ):
                    alpha_name = name.lower()
                    self.alpha_classes[alpha_name] = obj
                    self.logger.debug(f"Loaded alpha class: {alpha_name}")
                    break

        except Exception as e:
            self.logger.warning(f"Error loading {file_path}: {e}")

    # ========== Alpha Calculation ==========

    def get_alpha_instance(self, alpha_name: str) -> Any:
        """
        Get or create an alpha instance.

        Args:
            alpha_name: Name of the alpha

        Returns:
            Alpha instance
        """
        if alpha_name not in self.alpha_instances:
            if alpha_name not in self.alpha_classes:
                raise ValueError(f"Unknown alpha: {alpha_name}")

            self.alpha_instances[alpha_name] = self.alpha_classes[alpha_name]()

        return self.alpha_instances[alpha_name]

    def calculate_alpha(
        self,
        alpha_name: str,
        dataframe: pd.DataFrame,
        pair: str,
    ) -> pd.Series:
        """
        Calculate a single alpha using existing implementation.

        Args:
            alpha_name: Name of the alpha to calculate
            dataframe: OHLCV DataFrame with columns: open, high, low, close, volume
            pair: Trading pair name (e.g., 'BTCUSDT')

        Returns:
            Series with alpha values
        """
        alpha = self.get_alpha_instance(alpha_name)

        try:
            result = alpha.calculate(dataframe, pair)

            # Ensure result is a Series with same index as input
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=dataframe.index)

            # Fill NaN values
            result = result.fillna(0)

            return result

        except Exception as e:
            self.logger.warning(f"Error calculating {alpha_name}: {e}")
            return pd.Series(0.0, index=dataframe.index)

    def calculate_multiple_alphas(
        self,
        alpha_names: List[str],
        dataframe: pd.DataFrame,
        pair: str,
    ) -> pd.DataFrame:
        """
        Calculate multiple alphas efficiently.

        Args:
            alpha_names: List of alpha names
            dataframe: OHLCV DataFrame
            pair: Trading pair name

        Returns:
            DataFrame with alpha values as columns
        """
        results = {}

        for name in alpha_names:
            try:
                results[name] = self.calculate_alpha(name, dataframe, pair)
            except Exception as e:
                self.logger.warning(f"Skipping {name}: {e}")
                results[name] = pd.Series(0.0, index=dataframe.index)

        return pd.DataFrame(results, index=dataframe.index)

    def calculate_all_alphas(
        self,
        dataframe: pd.DataFrame,
        pair: str,
    ) -> pd.DataFrame:
        """
        Calculate all loaded alphas for a symbol.

        Args:
            dataframe: OHLCV DataFrame
            pair: Trading pair name

        Returns:
            DataFrame with all alpha values
        """
        return self.calculate_multiple_alphas(
            list(self.alpha_classes.keys()),
            dataframe,
            pair
        )

    # ========== Factor Calculation ==========

    def get_factor_calculator(self) -> Any:
        """Get CryptoFactorCalculator instance."""
        if self._factor_calculator_class is None:
            raise RuntimeError("CryptoFactorCalculator not loaded")
        return self._factor_calculator_class()

    def calculate_factors(
        self,
        dataframe: pd.DataFrame,
        pair: str,
        btc_dataframe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate 9-factor loadings for a symbol.

        Factors:
        1. Market (Beta to BTC)
        2. Momentum
        3. Short-term Reversal
        4. Volatility
        5. Liquidity
        6. Size
        7. Value
        8. Mean Reversion
        9. Quality

        Args:
            dataframe: OHLCV DataFrame for the symbol
            pair: Trading pair name
            btc_dataframe: BTC OHLCV for market factor (optional)

        Returns:
            DataFrame with 9 factor columns
        """
        # Return empty factors if no calculator available
        if self._factor_calculator_class is None:
            return pd.DataFrame(
                0.0,
                index=dataframe.index,
                columns=[
                    'market', 'momentum', 'reversal', 'volatility',
                    'liquidity', 'size', 'value', 'mean_reversion', 'quality'
                ]
            )

        try:
            calculator = self.get_factor_calculator()
            factor_df = calculator.calculate_factor_loadings(
                dataframe,
                pair,
                btc_dataframe=btc_dataframe,
            )
            return factor_df

        except Exception as e:
            self.logger.error(f"Error calculating factors: {e}")
            # Return empty factors
            return pd.DataFrame(
                0.0,
                index=dataframe.index,
                columns=[
                    'market', 'momentum', 'reversal', 'volatility',
                    'liquidity', 'size', 'value', 'mean_reversion', 'quality'
                ]
            )

    # ========== Utility Methods ==========

    def list_alphas(self) -> List[str]:
        """List all available alpha names."""
        return sorted(self.alpha_classes.keys())

    def list_alpha_101(self) -> List[str]:
        """List Alpha 101 alphas."""
        return sorted([a for a in self.alpha_classes.keys() if 'alpha_101' in a])

    def list_alpha_191(self) -> List[str]:
        """List Alpha 191 alphas - DISABLED (only using Alpha 101)."""
        return []  # Alpha 191 disabled, only using Alpha 101

    def get_alpha_info(self, alpha_name: str) -> Dict:
        """
        Get information about an alpha.

        Args:
            alpha_name: Alpha name

        Returns:
            Dict with alpha information
        """
        if alpha_name not in self.alpha_classes:
            return {'error': f'Unknown alpha: {alpha_name}'}

        cls = self.alpha_classes[alpha_name]
        instance = self.get_alpha_instance(alpha_name)

        return {
            'name': alpha_name,
            'class': cls.__name__,
            'neutralizer_type': getattr(instance, 'neutralizer_type', 'unknown'),
            'decay_period': getattr(instance, 'decay_period', 3),
        }


# ========== Standalone Testing ==========

if __name__ == '__main__':
    # Test the adapter
    adapter = AlphaAdapter()

    print(f"Loaded {len(adapter.list_alphas())} alphas")
    print(f"Alpha 101: {len(adapter.list_alpha_101())}")
    print(f"Alpha 191: {len(adapter.list_alpha_191())}")

    # Print first 10 alphas
    print("\nFirst 10 alphas:")
    for name in adapter.list_alphas()[:10]:
        info = adapter.get_alpha_info(name)
        print(f"  {name}: neutralizer={info.get('neutralizer_type')}")
