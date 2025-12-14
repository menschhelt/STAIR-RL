"""
State Builder - Constructs 2D State Matrix for RL Agent.

Updated design with PCA-compressed alphas + has_signal flag (36 total features):

State Matrix Shape = (N_assets, D_features) = (20, 36)

Features:
1. PCA-Compressed Alphas (20): Principal components from 300+ alpha factors
2. Risk Factors (4): beta_MKT, beta_SMB, beta_MOM, alpha_resid
3. Market Micro (3): momentum, volatility, liquidity
4. Private State (3): weight, sentiment, has_signal
   - has_signal: Binary flag (0/1) indicating whether sentiment data is available
   - Before 2022-12-01, Nostr data is sparse, so has_signal=0
5. Global Features (6): CMKT, CSMB, CMOM, risk_free_rate, yield_spread, VIX

Portfolio State (22-dim, separate from features):
- Current weights (20)
- leverage_ratio (1): Σ|w_i| / target_leverage
- cash_ratio (1): available_cash / NAV

Signal Flagging Strategy:
- If timestamp < NOSTR_ACTIVE_DATE (2022-12-01) OR no sentiment data:
  - sentiment = 0.0 (neutral)
  - has_signal = 0 (off state)
- Otherwise:
  - sentiment = actual_score
  - has_signal = 1 (on state)

This allows the RL agent to learn to IGNORE the sentiment feature when has_signal=0.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from config.settings import DATA_DIR


class StateBuilder:
    """
    Constructs 2D State Matrix for RL training/inference.

    Output: (N_assets, D_features) at each timestamp
    Default: (20, 35) - 20 assets with 35 features each

    This is the "observation" that the RL agent sees at each step.
    """

    # Feature dimensions (updated for PCA compression + has_signal)
    PCA_ALPHA_DIM = 20       # PCA-compressed alpha factors
    RISK_FACTOR_DIM = 4      # Factor loadings
    MARKET_MICRO_DIM = 3     # Price/volume features
    PRIVATE_STATE_DIM = 3    # Portfolio-specific features (weight, sentiment, has_signal)
    GLOBAL_FEATURE_DIM = 6   # Market-wide features (broadcast)

    LOCAL_FEATURE_DIM = PCA_ALPHA_DIM + RISK_FACTOR_DIM + MARKET_MICRO_DIM + PRIVATE_STATE_DIM  # 30
    TOTAL_FEATURE_DIM = LOCAL_FEATURE_DIM + GLOBAL_FEATURE_DIM  # 36

    # Feature names - ordered by category
    PCA_FEATURE_NAMES = [f'alpha_PC{i+1}' for i in range(PCA_ALPHA_DIM)]

    RISK_FEATURE_NAMES = [
        'beta_MKT',      # Market factor loading
        'beta_SMB',      # Size factor loading
        'beta_MOM',      # Momentum factor loading
        'alpha_resid',   # Residual alpha (key signal!)
    ]

    MICRO_FEATURE_NAMES = [
        'momentum',      # Price momentum (30d)
        'volatility',    # Return volatility (30d)
        'liquidity',     # Volume z-score
    ]

    PRIVATE_FEATURE_NAMES = [
        'weight',        # Current portfolio weight
        'sentiment',     # Asset-level sentiment
        'has_signal',    # Signal availability flag (0=no data, 1=data available)
    ]

    # Nostr became active from late 2022 - before this, sentiment data is sparse
    NOSTR_ACTIVE_DATE = pd.Timestamp("2022-12-01", tz='UTC')

    LOCAL_FEATURE_NAMES = (
        PCA_FEATURE_NAMES +
        RISK_FEATURE_NAMES +
        MICRO_FEATURE_NAMES +
        PRIVATE_FEATURE_NAMES
    )

    GLOBAL_FEATURE_NAMES = [
        'CMKT',          # Market factor return
        'CSMB',          # Size factor return
        'CMOM',          # Momentum factor return
        'risk_free_rate', # Fed Funds Rate (FRED: DFF)
        'yield_spread',  # 10Y-2Y Treasury spread
        'VIX',           # Market volatility index
    ]

    # Portfolio state names (separate from observation features)
    PORTFOLIO_STATE_NAMES = [
        *[f'weight_{i}' for i in range(20)],
        'leverage_ratio',
        'cash_ratio',
    ]

    def __init__(
        self,
        n_assets: int = 20,
        lookback_period: int = 60,
        normalize: bool = True,
        pca_compressor: Optional['AlphaPCACompressor'] = None,
        target_leverage: float = 2.0,
    ):
        """
        Initialize State Builder.

        Args:
            n_assets: Number of assets in universe (default: 20)
            lookback_period: Days for factor loading calculation
            normalize: Whether to z-score normalize features
            pca_compressor: Pre-fitted PCA compressor for alpha compression
            target_leverage: Target leverage for leverage_ratio calculation
        """
        self.n_assets = n_assets
        self.lookback_period = lookback_period
        self.normalize = normalize
        self.pca_compressor = pca_compressor
        self.target_leverage = target_leverage

        # State cache
        self.loading_matrix_cache: Dict[str, Dict[str, float]] = {}
        self.factor_returns_cache: Dict[pd.Timestamp, Dict[str, float]] = {}

        # Output directory
        self.state_dir = DATA_DIR / 'features' / 'states'
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_logging()

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

    def set_pca_compressor(self, compressor: 'AlphaPCACompressor'):
        """Set PCA compressor after initialization."""
        self.pca_compressor = compressor

    def build_state(
        self,
        timestamp: pd.Timestamp,
        universe: Dict[int, Optional[str]],
        ohlcv_data: Dict[str, pd.DataFrame],
        loading_matrix: Dict[str, Dict[str, float]],
        factor_returns: Dict[str, float],
        macro_data: pd.DataFrame,
        alpha_features: Optional[Dict[str, np.ndarray]] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
        portfolio_weights: Optional[Dict[str, float]] = None,
        nav: Optional[float] = None,
        margin_used: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build 2D state matrix at a single timestamp.

        Args:
            timestamp: Current timestamp
            universe: {slot: symbol} mapping
            ohlcv_data: {symbol: OHLCV DataFrame}
            loading_matrix: {symbol: {factor: loading}}
            factor_returns: {factor: return} for current period
            macro_data: DataFrame with macro indicators (VIX, yields, etc.)
            alpha_features: {symbol: alpha_vector} pre-computed alphas
            sentiment_data: DataFrame with sentiment scores
            portfolio_weights: {symbol: weight} current portfolio
            nav: Current Net Asset Value (for cash_ratio)
            margin_used: Current margin used (for cash_ratio)

        Returns:
            Tuple of:
            - State matrix of shape (n_assets, total_features)
            - Portfolio state of shape (22,) for actor input
        """
        state = np.zeros((self.n_assets, self.TOTAL_FEATURE_DIM))

        # Get global features
        global_features = self._extract_global_features(
            factor_returns, macro_data, timestamp
        )

        # Build state for each slot
        for slot in range(1, self.n_assets + 1):
            symbol = universe.get(slot)

            if symbol and symbol in ohlcv_data:
                # Get alpha features for this symbol
                symbol_alphas = alpha_features.get(symbol) if alpha_features else None

                # Local features
                local_features = self._extract_local_features(
                    symbol=symbol,
                    ohlcv=ohlcv_data.get(symbol, pd.DataFrame()),
                    loadings=loading_matrix.get(symbol, {}),
                    factor_returns=factor_returns,
                    alpha_features=symbol_alphas,
                    sentiment=sentiment_data,
                    portfolio_weight=portfolio_weights.get(symbol, 0.0) if portfolio_weights else 0.0,
                    timestamp=timestamp,
                )
            else:
                # Empty slot - fill with neutral values
                local_features = self._get_neutral_local_features()

            # Combine local and global features
            state[slot - 1, :self.LOCAL_FEATURE_DIM] = local_features
            state[slot - 1, self.LOCAL_FEATURE_DIM:] = global_features

        # Z-score normalization (cross-sectional for local, temporal for global)
        if self.normalize:
            state = self._normalize_state(state)

        # Build portfolio state
        portfolio_state = self._build_portfolio_state(
            portfolio_weights=portfolio_weights,
            universe=universe,
            nav=nav,
            margin_used=margin_used,
        )

        return state, portfolio_state

    def _extract_local_features(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        loadings: Dict[str, float],
        factor_returns: Dict[str, float],
        alpha_features: Optional[np.ndarray],
        sentiment: Optional[pd.DataFrame],
        portfolio_weight: float,
        timestamp: pd.Timestamp,
    ) -> np.ndarray:
        """Extract local (per-asset) features including PCA-compressed alphas."""
        features = np.zeros(self.LOCAL_FEATURE_DIM)
        idx = 0

        # 1. PCA-Compressed Alphas (20 features)
        if alpha_features is not None and self.pca_compressor is not None:
            try:
                pca_features = self.pca_compressor.transform(alpha_features)
                if pca_features.ndim == 1 and len(pca_features) == self.PCA_ALPHA_DIM:
                    features[idx:idx + self.PCA_ALPHA_DIM] = pca_features
            except Exception as e:
                # PCA not fitted or transform failed - use zeros
                pass
        elif alpha_features is not None and len(alpha_features) >= self.PCA_ALPHA_DIM:
            # No PCA compressor - use first 20 raw alphas
            features[idx:idx + self.PCA_ALPHA_DIM] = alpha_features[:self.PCA_ALPHA_DIM]
        idx += self.PCA_ALPHA_DIM

        # 2. Risk Factors (4 features)
        # beta_MKT
        features[idx] = loadings.get('CMKT', loadings.get('beta_MKT', 0.0))
        idx += 1
        # beta_SMB
        features[idx] = loadings.get('CSMB', loadings.get('beta_SMB', 0.0))
        idx += 1
        # beta_MOM
        features[idx] = loadings.get('CMOM', loadings.get('beta_MOM', 0.0))
        idx += 1

        # alpha_resid: R - sum(beta * factor_return)
        if not ohlcv.empty and 'close' in ohlcv.columns:
            try:
                if isinstance(ohlcv.index, pd.DatetimeIndex):
                    loc = ohlcv.index.get_indexer([timestamp], method='nearest')[0]
                    if loc > 0 and loc < len(ohlcv):
                        actual_return = (ohlcv.iloc[loc]['close'] / ohlcv.iloc[loc-1]['close']) - 1
                        expected_return = sum(
                            loadings.get(f, 0.0) * factor_returns.get(f, 0.0)
                            for f in ['CMKT', 'CSMB', 'CMOM']
                        )
                        features[idx] = actual_return - expected_return
            except Exception:
                features[idx] = 0.0
        idx += 1

        # 3. Market Micro (3 features)
        # Momentum (30-day return)
        if not ohlcv.empty and 'close' in ohlcv.columns and len(ohlcv) >= 30:
            features[idx] = (ohlcv['close'].iloc[-1] / ohlcv['close'].iloc[-30]) - 1
        idx += 1

        # Volatility (30-day return std, annualized)
        if not ohlcv.empty and 'close' in ohlcv.columns and len(ohlcv) >= 30:
            returns = ohlcv['close'].pct_change().dropna().iloc[-30:]
            if len(returns) > 0:
                features[idx] = returns.std() * np.sqrt(365)
        idx += 1

        # Liquidity (volume z-score)
        if not ohlcv.empty and 'volume' in ohlcv.columns and len(ohlcv) >= 30:
            recent_vol = ohlcv['volume'].iloc[-30:]
            if recent_vol.std() > 0:
                features[idx] = (recent_vol.iloc[-1] - recent_vol.mean()) / recent_vol.std()
        idx += 1

        # 4. Private State (3 features)
        # Portfolio weight
        features[idx] = portfolio_weight
        idx += 1

        # Sentiment + has_signal (Signal Flagging Strategy)
        # Check if timestamp is before Nostr active date (data is sparse before 2022-12)
        is_before_active_date = timestamp < self.NOSTR_ACTIVE_DATE

        sentiment_score = 0.0
        has_signal = 0.0

        if not is_before_active_date and sentiment is not None and not sentiment.empty:
            if 'symbol' in sentiment.columns:
                sym_sentiment = sentiment[sentiment['symbol'] == symbol]
                if not sym_sentiment.empty:
                    bullish = sym_sentiment.get('bullish_weighted', sym_sentiment.get('positive_weighted', 0))
                    bearish = sym_sentiment.get('bearish_weighted', sym_sentiment.get('negative_weighted', 0))
                    if isinstance(bullish, pd.Series):
                        bullish = bullish.iloc[-1] if len(bullish) > 0 else 0
                    if isinstance(bearish, pd.Series):
                        bearish = bearish.iloc[-1] if len(bearish) > 0 else 0
                    sentiment_score = float(bullish) - float(bearish)
                    has_signal = 1.0  # Data available

        # Sentiment score (0 if no signal)
        features[idx] = sentiment_score
        idx += 1

        # has_signal flag (0=no data/before active date, 1=data available)
        features[idx] = has_signal
        idx += 1

        return features

    def _extract_global_features(
        self,
        factor_returns: Dict[str, float],
        macro_data: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> np.ndarray:
        """Extract global (market-wide) features."""
        features = np.zeros(self.GLOBAL_FEATURE_DIM)

        # Factor returns
        features[0] = factor_returns.get('CMKT', 0.0)
        features[1] = factor_returns.get('CSMB', 0.0)
        features[2] = factor_returns.get('CMOM', 0.0)

        # Macro features from macro_data
        if not macro_data.empty:
            # Risk-free rate (Fed Funds Rate)
            dff = macro_data[macro_data['indicator_name'] == 'DFF']
            if not dff.empty:
                features[3] = dff['value'].iloc[-1] / 100  # Convert to decimal

            # Yield spread (10Y - 2Y)
            t10y = macro_data[macro_data['indicator_name'].isin(['^TNX', 'DGS10'])]
            t2y = macro_data[macro_data['indicator_name'].isin(['^FVX', 'DGS2'])]
            if not t10y.empty and not t2y.empty:
                features[4] = (t10y['value'].iloc[-1] - t2y['value'].iloc[-1]) / 100

            # VIX
            vix = macro_data[macro_data['indicator_name'] == '^VIX']
            if not vix.empty:
                features[5] = vix['value'].iloc[-1] / 100  # Normalize by 100

        return features

    def _build_portfolio_state(
        self,
        portfolio_weights: Optional[Dict[str, float]],
        universe: Dict[int, Optional[str]],
        nav: Optional[float],
        margin_used: Optional[float],
    ) -> np.ndarray:
        """
        Build portfolio state vector for actor input.

        Returns:
            Portfolio state of shape (22,):
            - weights (20): Current portfolio weights per slot
            - leverage_ratio (1): Σ|w_i| / target_leverage
            - cash_ratio (1): available_cash / NAV
        """
        portfolio_state = np.zeros(22)

        # Extract weights by slot
        if portfolio_weights:
            for slot in range(1, self.n_assets + 1):
                symbol = universe.get(slot)
                if symbol and symbol in portfolio_weights:
                    portfolio_state[slot - 1] = portfolio_weights[symbol]

        # Leverage ratio: Σ|w_i| / target_leverage
        gross_exposure = np.abs(portfolio_state[:self.n_assets]).sum()
        portfolio_state[20] = gross_exposure / self.target_leverage if self.target_leverage > 0 else 0.0

        # Cash ratio: (NAV - margin_used) / NAV
        if nav is not None and nav > 0:
            margin = margin_used if margin_used is not None else 0.0
            portfolio_state[21] = (nav - margin) / nav
        else:
            portfolio_state[21] = 1.0  # 100% cash if no NAV

        return portfolio_state

    def _get_neutral_local_features(self) -> np.ndarray:
        """Get neutral values for empty slots."""
        features = np.zeros(self.LOCAL_FEATURE_DIM)
        # PCA features: zeros
        # Risk factors: beta_MKT = 1 (market neutral), others = 0
        features[self.PCA_ALPHA_DIM] = 1.0  # beta_MKT = 1
        return features

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Z-score normalize the state matrix.

        Local features: Cross-sectional normalization (across assets)
        Global features: No normalization (already on same scale)
        """
        normalized = state.copy()

        # Cross-sectional z-score for local features
        for j in range(self.LOCAL_FEATURE_DIM):
            col = state[:, j]
            valid_mask = ~np.isnan(col) & (col != 0)

            if valid_mask.sum() > 1:
                mean = col[valid_mask].mean()
                std = col[valid_mask].std()
                if std > 1e-8:
                    normalized[:, j] = (col - mean) / std
                    # Clip extreme values
                    normalized[:, j] = np.clip(normalized[:, j], -3, 3)

        return normalized

    def build_state_sequence(
        self,
        start_date: datetime,
        end_date: datetime,
        universe_history: Dict[date, Dict[int, Optional[str]]],
        ohlcv_dict: Dict[str, pd.DataFrame],
        loading_matrix_history: Dict[pd.Timestamp, Dict[str, Dict[str, float]]],
        factor_returns_history: Dict[pd.Timestamp, Dict[str, float]],
        macro_data: pd.DataFrame,
        alpha_features_history: Optional[Dict[pd.Timestamp, Dict[str, np.ndarray]]] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
        interval: str = '1d',
    ) -> Tuple[np.ndarray, List[np.ndarray], List[pd.Timestamp]]:
        """
        Build state sequence for a date range.

        Args:
            start_date: Start of period
            end_date: End of period
            universe_history: {date: {slot: symbol}}
            ohlcv_dict: {symbol: OHLCV DataFrame}
            loading_matrix_history: {timestamp: {symbol: loadings}}
            factor_returns_history: {timestamp: factor_returns}
            macro_data: Macro indicators DataFrame
            alpha_features_history: {timestamp: {symbol: alpha_vector}}
            sentiment_data: Sentiment DataFrame
            interval: Data interval

        Returns:
            Tuple of (states array (T, N, D), portfolio_states list, timestamps list)
        """
        self.logger.info(f"Building state sequence from {start_date} to {end_date}")

        # Generate timestamps
        timestamps = pd.date_range(
            start=start_date,
            end=end_date,
            freq=interval,
            tz=timezone.utc,
        )

        states = []
        portfolio_states = []
        valid_timestamps = []

        for ts in timestamps:
            ts_date = ts.date()

            # Get universe for this date
            universe = universe_history.get(ts_date, {})
            if not universe:
                continue

            # Get loading matrix (use nearest)
            loading_matrix = self._get_nearest_value(loading_matrix_history, ts)
            if not loading_matrix:
                loading_matrix = {}

            # Get factor returns
            factor_returns = self._get_nearest_value(factor_returns_history, ts)
            if not factor_returns:
                factor_returns = {}

            # Get alpha features
            alpha_features = None
            if alpha_features_history:
                alpha_features = self._get_nearest_value(alpha_features_history, ts)

            # Build state
            state, portfolio_state = self.build_state(
                timestamp=ts,
                universe=universe,
                ohlcv_data=ohlcv_dict,
                loading_matrix=loading_matrix,
                factor_returns=factor_returns,
                macro_data=macro_data,
                alpha_features=alpha_features,
                sentiment_data=sentiment_data,
                portfolio_weights=None,  # No portfolio yet
                nav=None,
                margin_used=None,
            )

            states.append(state)
            portfolio_states.append(portfolio_state)
            valid_timestamps.append(ts)

        if not states:
            return np.array([]), [], []

        states_array = np.stack(states, axis=0)

        self.logger.info(
            f"Built state sequence: shape={states_array.shape} "
            f"(T={len(valid_timestamps)}, N={self.n_assets}, D={self.TOTAL_FEATURE_DIM})"
        )

        return states_array, portfolio_states, valid_timestamps

    def _get_nearest_value(
        self,
        history: Dict[pd.Timestamp, Any],
        target: pd.Timestamp,
    ) -> Optional[Any]:
        """Get value from history dict nearest to target timestamp."""
        if not history:
            return None

        timestamps = list(history.keys())
        # Find nearest timestamp <= target
        valid = [ts for ts in timestamps if ts <= target]
        if not valid:
            return None

        nearest = max(valid)
        return history[nearest]

    def save_states(
        self,
        states: np.ndarray,
        portfolio_states: List[np.ndarray],
        timestamps: List[pd.Timestamp],
        name: str,
    ):
        """Save state sequence to disk."""
        np.savez_compressed(
            self.state_dir / f"{name}.npz",
            states=states,
            portfolio_states=np.array(portfolio_states),
            timestamps=np.array([ts.isoformat() for ts in timestamps]),
            local_features=self.LOCAL_FEATURE_NAMES,
            global_features=self.GLOBAL_FEATURE_NAMES,
            portfolio_features=self.PORTFOLIO_STATE_NAMES,
        )
        self.logger.info(f"Saved states to {self.state_dir / name}.npz")

    def load_states(self, name: str) -> Tuple[np.ndarray, List[np.ndarray], List[pd.Timestamp]]:
        """Load state sequence from disk."""
        path = self.state_dir / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"States not found: {path}")

        data = np.load(path, allow_pickle=True)
        states = data['states']
        portfolio_states = list(data['portfolio_states'])
        timestamps = [pd.Timestamp(ts) for ts in data['timestamps']]

        return states, portfolio_states, timestamps

    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        return self.LOCAL_FEATURE_NAMES + self.GLOBAL_FEATURE_NAMES

    def get_feature_info(self) -> Dict[str, Any]:
        """Get detailed feature information."""
        return {
            'total_dim': self.TOTAL_FEATURE_DIM,
            'local_dim': self.LOCAL_FEATURE_DIM,
            'global_dim': self.GLOBAL_FEATURE_DIM,
            'pca_dim': self.PCA_ALPHA_DIM,
            'risk_dim': self.RISK_FACTOR_DIM,
            'micro_dim': self.MARKET_MICRO_DIM,
            'private_dim': self.PRIVATE_STATE_DIM,
            'portfolio_dim': len(self.PORTFOLIO_STATE_NAMES),
            'feature_names': self.get_feature_names(),
            'portfolio_state_names': self.PORTFOLIO_STATE_NAMES,
        }


# ========== Utility Functions ==========

def compute_leverage_ratio(weights: np.ndarray, target_leverage: float) -> float:
    """
    Compute leverage ratio for state inclusion.

    leverage_ratio = Σ|w_i| / target_leverage

    Args:
        weights: Current portfolio weights
        target_leverage: Target leverage (e.g., 2.0 = 200%)

    Returns:
        Leverage utilization ratio (0 to 1+ if over-leveraged)
    """
    gross_exposure = np.abs(weights).sum()
    return gross_exposure / target_leverage if target_leverage > 0 else 0.0


def compute_cash_ratio(nav: float, margin_used: float) -> float:
    """
    Compute cash ratio for state inclusion.

    cash_ratio = (NAV - margin_used) / NAV

    Args:
        nav: Current Net Asset Value
        margin_used: Current margin utilized

    Returns:
        Available cash as fraction of NAV (0 to 1)
    """
    if nav <= 0:
        return 0.0
    return max(0.0, (nav - margin_used) / nav)


# ========== Standalone Testing ==========

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from datetime import datetime, timezone

    # Generate sample data
    np.random.seed(42)

    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=60,
        freq='D'
    )

    # Sample universe
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
               'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT']

    universe = {i+1: sym for i, sym in enumerate(symbols)}
    for i in range(len(symbols), 20):
        universe[i+1] = None

    # Sample OHLCV data
    def generate_ohlcv(base_price: float) -> pd.DataFrame:
        n = len(dates)
        returns = np.random.randn(n) * 0.03
        close = base_price * np.cumprod(1 + returns)
        return pd.DataFrame({
            'open': close * (1 + np.random.randn(n) * 0.01),
            'high': close * (1 + np.abs(np.random.randn(n) * 0.02)),
            'low': close * (1 - np.abs(np.random.randn(n) * 0.02)),
            'close': close,
            'volume': np.abs(np.random.randn(n) * 1000 + 5000),
        }, index=dates)

    ohlcv_data = {s: generate_ohlcv(100 + np.random.rand() * 1000) for s in symbols}

    # Sample loading matrix
    loading_matrix = {
        s: {
            'CMKT': 0.8 + np.random.rand() * 0.4,
            'CSMB': np.random.randn() * 0.3,
            'CMOM': np.random.randn() * 0.2,
        }
        for s in symbols
    }

    # Sample factor returns
    factor_returns = {
        'CMKT': 0.02,
        'CSMB': 0.005,
        'CMOM': 0.01,
    }

    # Sample macro data
    macro_data = pd.DataFrame({
        'indicator_name': ['DFF', '^TNX', '^FVX', '^VIX'],
        'value': [5.25, 4.5, 4.0, 18.0],
    })

    # Sample alpha features (simulated PCA-ready)
    alpha_features = {
        s: np.random.randn(300)  # 300 raw alpha features
        for s in symbols
    }

    # Sample portfolio weights
    portfolio_weights = {
        'BTCUSDT': 0.3,
        'ETHUSDT': -0.2,
        'SOLUSDT': 0.15,
        'BNBUSDT': 0.1,
    }

    # Test state building
    builder = StateBuilder(n_assets=20, target_leverage=2.0)
    timestamp = dates[-1]

    state, portfolio_state = builder.build_state(
        timestamp=timestamp,
        universe=universe,
        ohlcv_data=ohlcv_data,
        loading_matrix=loading_matrix,
        factor_returns=factor_returns,
        macro_data=macro_data,
        alpha_features=alpha_features,
        sentiment_data=None,
        portfolio_weights=portfolio_weights,
        nav=100000,
        margin_used=50000,
    )

    print("\n" + "=" * 60)
    print("STATE BUILDER TEST (Updated 36-dim with has_signal)")
    print("=" * 60)
    print(f"\nState shape: {state.shape}")
    print(f"Portfolio state shape: {portfolio_state.shape}")

    info = builder.get_feature_info()
    print(f"\nFeature breakdown:")
    print(f"  PCA alphas: {info['pca_dim']}")
    print(f"  Risk factors: {info['risk_dim']}")
    print(f"  Market micro: {info['micro_dim']}")
    print(f"  Private state: {info['private_dim']} (weight, sentiment, has_signal)")
    print(f"  Global features: {info['global_dim']}")
    print(f"  Total: {info['total_dim']}")

    print(f"\nPortfolio state:")
    print(f"  leverage_ratio: {portfolio_state[20]:.4f}")
    print(f"  cash_ratio: {portfolio_state[21]:.4f}")

    print("\n" + "-" * 40)
    print("Sample state (first 3 assets):")
    print("-" * 40)

    all_features = builder.get_feature_names()
    for slot in range(3):
        sym = universe.get(slot + 1, 'N/A')
        print(f"\nSlot {slot + 1} ({sym}):")
        # Print select features only (updated indices for 36-dim)
        print(f"  alpha_PC1: {state[slot, 0]:.4f}")
        print(f"  alpha_PC20: {state[slot, 19]:.4f}")
        print(f"  beta_MKT: {state[slot, 20]:.4f}")
        print(f"  alpha_resid: {state[slot, 23]:.4f}")
        print(f"  momentum: {state[slot, 24]:.4f}")
        print(f"  weight: {state[slot, 27]:.4f}")
        print(f"  sentiment: {state[slot, 28]:.4f}")
        print(f"  has_signal: {state[slot, 29]:.4f}")  # New!
        print(f"  CMKT (global): {state[slot, 30]:.4f}")
        print(f"  VIX (global): {state[slot, 35]:.4f}")
