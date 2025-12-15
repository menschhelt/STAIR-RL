"""
HierarchicalActorCritic State Builder

Converts simple (market_state, portfolio_state) inputs to state_dict format
required by HierarchicalActorCritic.

This adapter allows gradual migration from basic networks to hierarchical architecture.
"""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F
import pandas as pd


class HierarchicalStateBuilder:
    """
    Convert simple market/portfolio states to HierarchicalActorCritic state_dict.

    This builder handles the conversion from the current training format:
        - market_state: (B, N, state_dim)
        - portfolio_state: (B, portfolio_dim)

    To the format required by HierarchicalActorCritic:
        state_dict: {
            # Local Features (per-asset, with N dimension):
            'alphas': (B, T, N, 101),                  # Alpha101 only: alpha_000~alpha_100
            'ohlcv_seq': (B, T, N, 288, 5),           # OHLCV sequences
            'portfolio_state': (B, portfolio_dim),     # Portfolio state

            # Global Features (market-wide, NO N dimension):
            'news_embedding': (B, T, 768),             # GDELT market-wide
            'social_embedding': (B, T, 768),           # Nostr market-wide
            'has_social_signal': (B, T, 1),           # Social signal availability
            'global_features': (B, T, 28),            # 23 macro + 5 Fama-French factors
        }
    """

    def __init__(
        self,
        n_assets: int,
        n_alphas: int = 101,
        temporal_window: int = 5,  # Reduced from 20 for memory efficiency
        gdelt_embeddings_path: Optional[str] = None,
        nostr_embeddings_path: Optional[str] = None,
        ohlcv_data_dir: Optional[str] = None,
        ohlcv_lookback: int = 72,  # Reduced from 288 (6h instead of 24h) for memory efficiency
        alpha_cache_dir: Optional[str] = None,
        macro_data_dir: Optional[str] = None,
        use_normalized_alphas: bool = True,
        device: str = 'cpu',
    ):
        """
        Initialize state builder.

        Args:
            n_assets: Number of assets
            n_alphas: Number of alpha factors (default: 101 for Alpha101 only: alpha_000~alpha_100)
            temporal_window: Temporal window size T (default: 20)
            gdelt_embeddings_path: Path to GDELT HDF5 file (optional)
            nostr_embeddings_path: Path to Nostr HDF5 file (optional)
            ohlcv_data_dir: Path to OHLCV parquet files directory (optional)
            ohlcv_lookback: Number of 5-minute candles per sequence (default: 288 = 1 day)
            alpha_cache_dir: Path to alpha cache directory (optional, auto-detected if None)
            macro_data_dir: Path to macro data directory (optional, auto-detected if None)
            use_normalized_alphas: Whether to use normalized alphas from alpha_cache_normalized/ (default: True)
            device: Device for tensors
        """
        self.n_assets = n_assets
        self.n_alphas = n_alphas
        self.temporal_window = temporal_window
        self.ohlcv_lookback = ohlcv_lookback
        self.device = device
        self.use_normalized_alphas = use_normalized_alphas

        # Auto-detect alpha cache directory if not provided
        if alpha_cache_dir is None:
            from config.settings import DATA_DIR
            if use_normalized_alphas:
                self.alpha_cache_dir = DATA_DIR / 'features' / 'alpha_cache_normalized'
            else:
                self.alpha_cache_dir = DATA_DIR / 'features' / 'alpha_cache'
        else:
            from pathlib import Path
            self.alpha_cache_dir = Path(alpha_cache_dir)

        # Initialize embedding loader if paths provided
        if gdelt_embeddings_path and nostr_embeddings_path:
            from agents.embedding_loader import EmbeddingLoader
            self.embedding_loader = EmbeddingLoader(
                gdelt_path=gdelt_embeddings_path,
                nostr_path=nostr_embeddings_path,
                device=device,
            )
        else:
            self.embedding_loader = None

        # Initialize OHLCV builder if path provided
        if ohlcv_data_dir:
            from features.ohlcv_processor import OHLCVSequenceBuilder
            self.ohlcv_builder = OHLCVSequenceBuilder(
                data_dir=ohlcv_data_dir,
                lookback=ohlcv_lookback,
            )
        else:
            self.ohlcv_builder = None

        # Initialize MacroDataLoader
        from pathlib import Path
        from config.settings import DATA_DIR
        if macro_data_dir is None:
            macro_data_dir = DATA_DIR / 'macro'
        else:
            macro_data_dir = Path(macro_data_dir)

        # Only initialize if directory exists
        if macro_data_dir.exists():
            from features.macro_loader import MacroDataLoader
            self.macro_loader = MacroDataLoader(
                macro_data_dir=macro_data_dir,
                cache_months=24,  # Increased from 12 for better caching
            )
        else:
            self.macro_loader = None

        # Initialize AlphaLoader for Alpha101 factors
        if self.alpha_cache_dir.exists():
            from agents.alpha_loader import AlphaLoader
            self.alpha_loader = AlphaLoader(
                alpha_cache_dir=self.alpha_cache_dir,
                n_alphas=n_alphas,
                device=device,
            )
        else:
            self.alpha_loader = None

        # Symbol list for alpha loading (set by environment or training loop)
        self._current_symbols: Optional[List[str]] = None

    def set_symbols(self, symbols: List[str]) -> None:
        """
        Set the current symbol list for alpha loading.

        Args:
            symbols: List of symbol names (e.g., ['BTCUSDT', 'ETHUSDT', ...])
        """
        self._current_symbols = symbols

    def preload_alphas(self) -> None:
        """Preload all alpha files into memory for faster access."""
        if self.alpha_loader is not None:
            self.alpha_loader.preload_all_alphas()

    def build_state_dict(
        self,
        market_state: torch.Tensor,
        portfolio_state: torch.Tensor,
        timestamps: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build state_dict from simple market/portfolio states.

        Args:
            market_state: (B, N, state_dim) - Current market features
            portfolio_state: (B, portfolio_dim) - Portfolio state
            timestamps: Optional list of ISO timestamps for embedding lookup

        Returns:
            state_dict with required fields for HierarchicalActorCritic

        Notes:
            - For mock data (state_dim=36), we pad to 292 alphas
            - News/social embeddings loaded from HDF5 if paths provided
            - Temporal dimension is created by repeating current state
            - Global features are zeros (will be extracted from real data later)
        """
        B, N, state_dim = market_state.shape
        device = market_state.device

        # ======================
        # 1. Alphas (B, T, N, 101)
        # ======================
        # Filter valid timestamps (not empty strings or NaT)
        valid_ts_for_alphas = []
        if timestamps is not None:
            for ts in timestamps[-self.temporal_window:]:
                ts_str = str(ts).strip() if ts else ''
                if ts_str and ts_str.upper() != 'NAT':
                    parsed = pd.Timestamp(ts)
                    if not pd.isnull(parsed):
                        valid_ts_for_alphas.append(parsed)

        if self.alpha_loader is not None and self._current_symbols is not None and len(valid_ts_for_alphas) > 0:
            # Load actual Alpha101 factors
            import numpy as np

            # Use filtered valid timestamps
            ts_list = valid_ts_for_alphas

            # Use fast method if preloaded, otherwise fall back to slow method
            if self.alpha_loader.has_slot_mapping():
                # Best: slot-aware lookup (dynamic universe support)
                alpha_np = self.alpha_loader.get_alphas_for_slots_fast(ts_list)  # (T, N_slots, 101)
            elif self.alpha_loader.is_preloaded():
                # Fallback: get all preloaded symbols (may cause dimension mismatch!)
                alpha_np = self.alpha_loader.get_alphas_fast(ts_list)  # (T, N_all, 101)
                # Slice to N assets if needed
                if alpha_np.shape[1] > N:
                    alpha_np = alpha_np[:, :N, :]
            else:
                alpha_np = self.alpha_loader.get_alphas_for_symbols(
                    symbols=self._current_symbols[:N],
                    timestamps=ts_list,
                )  # (T, N, 101)

            # Convert to tensor and add batch dimension
            alphas = torch.from_numpy(alpha_np).to(device=device, dtype=torch.float32)
            alphas = alphas.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, T, N, 101)

            # Pad temporal dimension if needed
            if alphas.shape[1] < self.temporal_window:
                pad_size = self.temporal_window - alphas.shape[1]
                alphas = F.pad(alphas, (0, 0, 0, 0, pad_size, 0))  # Pad T dimension
        else:
            # Fallback: use market_state features (padded to n_alphas)
            alphas_current = market_state  # (B, N, state_dim)

            if state_dim < self.n_alphas:
                # Pad to n_alphas
                alphas_padded = F.pad(alphas_current, (0, self.n_alphas - state_dim))
            elif state_dim > self.n_alphas:
                # Truncate
                alphas_padded = alphas_current[..., :self.n_alphas]
            else:
                alphas_padded = alphas_current

            # Create temporal dimension by repeating current state
            alphas = alphas_padded.unsqueeze(1).repeat(1, self.temporal_window, 1, 1)  # (B, T, N, 101)

        # ======================
        # 2. News Embeddings - Market-Wide (B, T, 768) [NO N dimension]
        # ======================
        # Pre-compute temporal_timestamps ONCE for all downstream uses
        temporal_timestamps = None
        temporal_timestamps_pd = None  # pd.Timestamp objects for macro lookup

        if timestamps is not None and len(timestamps) > 0:
            # Use the last timestamp as current time
            current_time = pd.Timestamp(timestamps[-1])

            # OPTIMIZED: Use numpy for timestamp computation (36x faster than pd.Timedelta loop)
            import numpy as np
            base_np = np.datetime64(current_time.value, 'ns')
            offsets_ns = np.arange(self.temporal_window - 1, -1, -1) * 5 * 60 * int(1e9)  # minutes to ns
            timestamps_np = base_np - offsets_ns.astype('timedelta64[ns]')

            # Convert to ISO strings for embedding lookup
            temporal_timestamps = [pd.Timestamp(ts).isoformat() for ts in timestamps_np]

            # Keep pd.Timestamp objects for macro lookup (avoid re-parsing)
            temporal_timestamps_pd = [pd.Timestamp(ts) for ts in timestamps_np]

        if self.embedding_loader is not None and temporal_timestamps is not None:

            # Get embeddings for all assets (used for Top 20 pooling)
            asset_indices = list(range(N))

            # Get GDELT embeddings (market-wide): (T, 768)
            # Use fast method if available (300x faster)
            if hasattr(self.embedding_loader, 'get_gdelt_embeddings_marketwide_fast'):
                news_emb_single = self.embedding_loader.get_gdelt_embeddings_marketwide_fast(
                    temporal_timestamps, asset_indices
                )
            else:
                news_emb_single = self.embedding_loader.get_gdelt_embeddings_marketwide(
                    temporal_timestamps, asset_indices
                )

            # Expand batch dimension: (1, T, 768)
            news_embedding = news_emb_single.unsqueeze(0)

            # Repeat for batch size: (B, T, 768)
            news_embedding = news_embedding.repeat(B, 1, 1)
        else:
            # Fallback: zeros (mock data or no embeddings)
            news_embedding = torch.zeros(
                B, self.temporal_window, 768,
                device=device,
                dtype=torch.float32
            )

        # ======================
        # 3. Social Embeddings - Market-Wide (B, T, 768) [NO N dimension]
        # ======================
        if self.embedding_loader is not None and timestamps is not None:
            # Load actual Nostr embeddings (market-wide, use same temporal_timestamps)
            # Use fast method if available (300x faster)
            if hasattr(self.embedding_loader, 'get_nostr_embeddings_marketwide_fast'):
                social_emb_single = self.embedding_loader.get_nostr_embeddings_marketwide_fast(
                    temporal_timestamps, asset_indices
                )
            else:
                social_emb_single = self.embedding_loader.get_nostr_embeddings_marketwide(
                    temporal_timestamps, asset_indices
                )

            # Expand batch dimension: (1, T, 768)
            social_embedding = social_emb_single.unsqueeze(0)

            # Repeat for batch size: (B, T, 768)
            social_embedding = social_embedding.repeat(B, 1, 1)
        else:
            # Fallback: zeros
            social_embedding = torch.zeros(
                B, self.temporal_window, 768,
                device=device,
                dtype=torch.float32
            )

        # ======================
        # 4. Social Signal Availability - Market-Wide (B, T, 1) [NO N dimension]
        # ======================
        if self.embedding_loader is not None and timestamps is not None:
            # Get actual social signal mask (market-wide)
            # Use fast method if available (300x faster)
            if hasattr(self.embedding_loader, 'get_social_signal_mask_marketwide_fast'):
                has_signal_single = self.embedding_loader.get_social_signal_mask_marketwide_fast(
                    temporal_timestamps, asset_indices
                )
            else:
                has_signal_single = self.embedding_loader.get_social_signal_mask_marketwide(
                    temporal_timestamps, asset_indices
                )

            # Expand batch dimension: (1, T, 1)
            has_social_signal = has_signal_single.unsqueeze(0)

            # Repeat for batch size: (B, T, 1)
            has_social_signal = has_social_signal.repeat(B, 1, 1)
        else:
            # Fallback: zeros
            has_social_signal = torch.zeros(
                B, self.temporal_window, 1,
                device=device,
                dtype=torch.float32
            )

        # ======================
        # 5. OHLCV Sequences (B, T, N, 288, 5)
        # ======================
        if self.ohlcv_builder is not None and timestamps is not None:
            # Load actual OHLCV sequences
            # We need to load sequences for [t-T+1, ..., t]
            import numpy as np

            # Use the last timestamp as current time
            current_time = pd.Timestamp(timestamps[-1])

            # Compute target dates for each temporal step
            temporal_dates = []
            for i in range(self.temporal_window):
                # Go back in time (5-minute intervals)
                ts = current_time - pd.Timedelta(minutes=5 * (self.temporal_window - 1 - i))
                temporal_dates.append(ts.date())

            # TODO: Get actual symbol names for asset_indices
            # For now, we assume market_state has a 'symbols' attribute or we use placeholder
            # This needs to be passed from the environment or data source
            # Placeholder: Use generic symbols (this should be replaced with actual symbols)
            symbols = [f'ASSET{i}' for i in range(N)]  # Placeholder

            # Load sequences for all (T, N) combinations
            ohlcv_list = []
            for target_date in temporal_dates:
                asset_sequences = []
                for symbol in symbols:
                    try:
                        seq = self.ohlcv_builder.load_5min_sequence(
                            symbol=symbol,
                            target_date=target_date,
                            lookback=self.ohlcv_lookback,
                        )
                        asset_sequences.append(seq)
                    except Exception:
                        # Fallback to zeros if loading fails
                        asset_sequences.append(
                            np.zeros((self.ohlcv_lookback, 5), dtype=np.float32)
                        )

                # Stack assets: (N, 288, 5)
                t_sequences = np.stack(asset_sequences, axis=0)
                ohlcv_list.append(t_sequences)

            # Stack temporal: (T, N, 288, 5)
            ohlcv_array = np.stack(ohlcv_list, axis=0)

            # Convert to tensor: (T, N, 288, 5)
            ohlcv_single = torch.tensor(ohlcv_array, dtype=torch.float32, device=device)

            # Expand batch dimension: (1, T, N, 288, 5)
            ohlcv_seq = ohlcv_single.unsqueeze(0)

            # Repeat for batch size: (B, T, N, 288, 5)
            ohlcv_seq = ohlcv_seq.repeat(B, 1, 1, 1, 1)
        else:
            # Fallback: zeros (mock data or no OHLCV builder)
            ohlcv_seq = torch.zeros(
                B, self.temporal_window, N, self.ohlcv_lookback, 5,
                device=device,
                dtype=torch.float32
            )

        # ======================
        # 6. Global Features (B, T, 28) - 23 macro + 5 Fama-French
        # ======================
        # OPTIMIZED: Use pre-computed temporal_timestamps_pd and batch lookup

        if self.macro_loader is not None and temporal_timestamps_pd is not None:
            try:
                # Use batch lookup if preloaded (much faster than individual calls)
                if self.macro_loader.is_preloaded() and hasattr(self.macro_loader, 'get_features_batch'):
                    # FAST PATH: Single batch lookup (T lookups in one call)
                    global_features_np = self.macro_loader.get_features_batch(temporal_timestamps_pd)
                    global_features_single = torch.from_numpy(global_features_np)
                elif self.macro_loader.is_preloaded():
                    # MEDIUM PATH: Individual fast lookups
                    global_features_list = []
                    for ts in temporal_timestamps_pd:
                        macro_feat = self.macro_loader.get_features_fast(ts)
                        global_features_list.append(torch.from_numpy(macro_feat))
                    global_features_single = torch.stack(global_features_list)
                else:
                    # SLOW PATH: Individual slow lookups (parquet reads)
                    global_features_list = []
                    for ts in temporal_timestamps_pd:
                        macro_feat = self.macro_loader.get_global_features(ts)
                        global_features_list.append(torch.from_numpy(macro_feat))
                    global_features_single = torch.stack(global_features_list)

                # Expand batch dimension: (1, T, n_features)
                global_features = global_features_single.unsqueeze(0)

                # Repeat for batch size: (B, T, n_features)
                global_features = global_features.repeat(B, 1, 1).to(device)

            except Exception as e:
                import logging
                logging.warning(f"Failed to load macro features: {e}")
                n_global_features = self.macro_loader.n_features if self.macro_loader else 6
                global_features = torch.zeros(
                    B, self.temporal_window, n_global_features,
                    device=device,
                    dtype=torch.float32
                )
        else:
            # Fallback: zeros (mock data or no macro loader)
            # Default to 23 features if macro_loader is initialized, else 6 for backward compat
            n_global_features = self.macro_loader.n_features if self.macro_loader else 6
            global_features = torch.zeros(
                B, self.temporal_window, n_global_features,
                device=device,
                dtype=torch.float32
            )

        # ======================
        # 7. Portfolio State (B, portfolio_dim)
        # ======================
        # Pass through unchanged
        # Expected format: (B, 22) with N+2 dimensions:
        #   - Current weights: N
        #   - Cash position: 1
        #   - Total equity: 1

        return {
            'alphas': alphas,
            'news_embedding': news_embedding,
            'social_embedding': social_embedding,
            'has_social_signal': has_social_signal,
            'ohlcv_seq': ohlcv_seq,
            'global_features': global_features,
            'portfolio_state': portfolio_state,
        }

    def validate_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Validate that state_dict has correct shapes.

        Args:
            state_dict: State dictionary to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            B = state_dict['portfolio_state'].shape[0]
            T = self.temporal_window
            N = self.n_assets

            # Determine expected global features dimension
            n_global_features = self.macro_loader.n_features if self.macro_loader else 28

            expected_shapes = {
                # Local Features (per-asset, with N dimension):
                'alphas': (B, T, N, self.n_alphas),          # (B, T, N, 101)
                'portfolio_state': (B,),                      # Variable last dimension

                # Global Features (market-wide, NO N dimension):
                'news_embedding': (B, T, 768),                # Market-wide GDELT
                'social_embedding': (B, T, 768),              # Market-wide Nostr
                'has_social_signal': (B, T, 1),              # Market-wide signal mask
                'global_features': (B, T, n_global_features), # 23 macro + 5 FF = 28
            }

            for key, expected_shape in expected_shapes.items():
                if key not in state_dict:
                    print(f"Missing key: {key}")
                    return False

                actual_shape = state_dict[key].shape
                if key == 'portfolio_state':
                    # Only check batch dimension for portfolio_state
                    if actual_shape[0] != B:
                        print(f"{key}: Expected batch size {B}, got {actual_shape[0]}")
                        return False
                else:
                    if actual_shape != expected_shape:
                        print(f"{key}: Expected {expected_shape}, got {actual_shape}")
                        return False

            return True

        except Exception as e:
            print(f"Validation error: {e}")
            return False
