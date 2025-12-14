"""
Markowitz Mean-Variance Optimization Benchmark.

Classic portfolio optimization:
max μ'w - (λ/2) w'Σw
s.t. sum(w) = 1, w >= 0 (for long-only)

References:
- Markowitz, H. (1952). "Portfolio Selection"
- Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix"
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal
import logging
import warnings

from .base_benchmark import BaseBenchmark, BenchmarkConfig

logger = logging.getLogger(__name__)

# Try to import cvxpy for optimization
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("cvxpy not installed. Markowitz optimization will use fallback method.")


class MarkowitzBenchmark(BaseBenchmark):
    """
    Mean-Variance Optimization (MVO) Portfolio Strategy.

    Objective: max E[R] - (λ/2) Var[R]
    Equivalent to: max μ'w - (λ/2) w'Σw

    where:
    - μ: Expected returns vector (estimated from historical data)
    - Σ: Covariance matrix (estimated from historical data)
    - λ: Risk aversion parameter
    - w: Portfolio weights

    Options:
    - long_only: If True, w >= 0
    - shrinkage: Use Ledoit-Wolf shrinkage for covariance estimation
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        risk_aversion: float = 1.0,
        long_only: bool = True,
        shrinkage: bool = True,
        min_history: int = 60,
    ):
        """
        Initialize Markowitz MVO benchmark.

        Args:
            config: Benchmark configuration
            risk_aversion: Lambda parameter (higher = more conservative)
            long_only: If True, enforce non-negative weights
            shrinkage: If True, use Ledoit-Wolf shrinkage for covariance
            min_history: Minimum observations required for estimation
        """
        super().__init__(config)
        self.name = "Markowitz (MV)"
        self.risk_aversion = risk_aversion
        self.long_only = long_only
        self.shrinkage = shrinkage
        self.min_history = min_history

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute mean-variance optimal portfolio.

        Args:
            timestamp: Current timestamp
            prices: Current asset prices (n_assets,)
            features: Dictionary with 'returns' key (T, n_assets)
            current_weights: Current portfolio weights (not used)

        Returns:
            weights: MVO optimal weights (n_assets,)
        """
        n_assets = self.config.n_assets

        # Get historical returns
        returns = features.get('returns', None)

        if returns is None or len(returns) < self.min_history:
            logger.warning(f"Insufficient history at {timestamp}, using equal weight")
            return np.ones(n_assets) * self.config.target_leverage / n_assets

        # Ensure correct shape
        if returns.shape[1] < n_assets:
            padding = np.zeros((returns.shape[0], n_assets - returns.shape[1]))
            returns = np.hstack([returns, padding])
        elif returns.shape[1] > n_assets:
            returns = returns[:, :n_assets]

        # Estimate expected returns and covariance
        mu = self._estimate_expected_returns(returns)
        sigma = self._estimate_covariance(returns)

        # Solve optimization problem
        if CVXPY_AVAILABLE:
            weights = self._solve_mvo_cvxpy(mu, sigma)
        else:
            weights = self._solve_mvo_analytical(mu, sigma)

        # Scale to target leverage
        gross_exposure = np.abs(weights).sum()
        if gross_exposure > 0:
            weights = weights / gross_exposure * self.config.target_leverage

        return weights

    def _estimate_expected_returns(self, returns: np.ndarray) -> np.ndarray:
        """
        Estimate expected returns from historical data.

        Uses simple mean. Could be extended to use:
        - CAPM expected returns
        - Shrinkage estimators
        - Factor model predictions
        """
        return np.mean(returns, axis=0)

    def _estimate_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Estimate covariance matrix from historical data.

        Uses Ledoit-Wolf shrinkage if enabled for better
        conditioning and out-of-sample performance.
        """
        if self.shrinkage:
            return self._ledoit_wolf_shrinkage(returns)
        else:
            return np.cov(returns.T)

    def _ledoit_wolf_shrinkage(self, returns: np.ndarray) -> np.ndarray:
        """
        Ledoit-Wolf shrinkage estimator for covariance matrix.

        Shrinks sample covariance toward a structured target
        (scaled identity matrix).

        Reference: Ledoit & Wolf (2004)
        """
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            lw.fit(returns)
            return lw.covariance_
        except ImportError:
            # Fallback to simple shrinkage
            n, p = returns.shape
            sample_cov = np.cov(returns.T)

            # Target: scaled identity
            mu = np.trace(sample_cov) / p
            target = mu * np.eye(p)

            # Shrinkage intensity (simplified)
            shrinkage = 0.2

            return (1 - shrinkage) * sample_cov + shrinkage * target

    def _solve_mvo_cvxpy(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Solve MVO using cvxpy convex optimization.

        Problem:
        max μ'w - (λ/2) w'Σw
        s.t. sum(|w|) <= L (leverage constraint)
             w >= 0 (if long_only)
        """
        n = len(mu)
        w = cp.Variable(n)

        # Objective: maximize return - risk_aversion * variance
        ret = mu @ w
        risk = cp.quad_form(w, sigma)
        objective = cp.Maximize(ret - (self.risk_aversion / 2) * risk)

        # Constraints
        constraints = []

        if self.long_only:
            constraints.append(w >= 0)
            constraints.append(cp.sum(w) <= self.config.target_leverage)
        else:
            # Long-short: constraint on gross exposure
            constraints.append(cp.norm(w, 1) <= self.config.target_leverage)

        # Individual position limits
        constraints.append(w <= self.config.max_single_weight)
        if not self.long_only:
            constraints.append(w >= -self.config.max_single_weight)

        # Solve
        prob = cp.Problem(objective, constraints)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=cp.OSQP, verbose=False)

            if prob.status in ['optimal', 'optimal_inaccurate']:
                return w.value
            else:
                logger.warning(f"MVO solver status: {prob.status}, using fallback")
                return self._solve_mvo_analytical(mu, sigma)

        except Exception as e:
            logger.warning(f"MVO solver failed: {e}, using fallback")
            return self._solve_mvo_analytical(mu, sigma)

    def _solve_mvo_analytical(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Analytical solution for unconstrained MVO.

        w* = (1/λ) Σ^(-1) μ

        With normalization for leverage constraint.
        """
        n = len(mu)

        try:
            # Add small regularization for numerical stability
            sigma_reg = sigma + 1e-6 * np.eye(n)
            sigma_inv = np.linalg.inv(sigma_reg)

            # Unconstrained optimal weights
            w_star = (1 / self.risk_aversion) * sigma_inv @ mu

            # For long-only, clip negative weights
            if self.long_only:
                w_star = np.maximum(w_star, 0)

            # Normalize to target leverage
            gross = np.abs(w_star).sum()
            if gross > 0:
                w_star = w_star / gross * self.config.target_leverage

            return w_star

        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is singular, using equal weight")
            weights = np.ones(n) / n * self.config.target_leverage
            return weights


class MinVarianceBenchmark(MarkowitzBenchmark):
    """
    Minimum Variance Portfolio.

    Special case of MVO where we only minimize variance
    without considering expected returns.

    min w'Σw
    s.t. sum(w) = 1

    Often performs well out-of-sample because it doesn't
    rely on return estimates (which are noisy).
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None, **kwargs):
        # Very high risk aversion effectively ignores returns
        super().__init__(config, risk_aversion=1000.0, **kwargs)
        self.name = "Minimum Variance (MinVar)"


class MaxSharpeRatioBenchmark(BaseBenchmark):
    """
    Maximum Sharpe Ratio Portfolio (Tangency Portfolio).

    Maximize risk-adjusted return:
    max (μ'w - r_f) / √(w'Σw)

    where r_f is the risk-free rate.

    This is equivalent to MVO with a specific risk aversion
    that depends on the risk-free rate and market conditions.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        risk_free_rate: float = 0.0,
        long_only: bool = True,
        shrinkage: bool = True,
    ):
        super().__init__(config)
        self.name = "Max Sharpe Ratio"
        self.risk_free_rate = risk_free_rate
        self.long_only = long_only
        self.shrinkage = shrinkage

    def compute_weights(
        self,
        timestamp: pd.Timestamp,
        prices: np.ndarray,
        features: Dict[str, np.ndarray],
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute maximum Sharpe ratio portfolio.
        """
        n_assets = self.config.n_assets
        returns = features.get('returns', None)

        if returns is None or len(returns) < 60:
            return np.ones(n_assets) * self.config.target_leverage / n_assets

        mu = np.mean(returns, axis=0) - self.risk_free_rate
        sigma = np.cov(returns.T) if not self.shrinkage else self._ledoit_wolf(returns)

        if not CVXPY_AVAILABLE:
            # Fallback: analytical tangency portfolio
            try:
                sigma_inv = np.linalg.inv(sigma + 1e-6 * np.eye(n_assets))
                w = sigma_inv @ mu
                if self.long_only:
                    w = np.maximum(w, 0)
                w = w / np.abs(w).sum() * self.config.target_leverage
                return w
            except:
                return np.ones(n_assets) * self.config.target_leverage / n_assets

        # Solve using second-order cone programming
        w = cp.Variable(n_assets)
        ret = mu @ w
        risk = cp.norm(cp.psd_wrap(sigma) @ w, 2)

        constraints = [risk <= 1]  # Normalize risk
        if self.long_only:
            constraints.append(w >= 0)
        constraints.append(cp.norm(w, 1) <= self.config.target_leverage)

        prob = cp.Problem(cp.Maximize(ret), constraints)

        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status == 'optimal':
                return w.value
        except:
            pass

        return np.ones(n_assets) * self.config.target_leverage / n_assets

    def _ledoit_wolf(self, returns):
        """Simple Ledoit-Wolf shrinkage."""
        try:
            from sklearn.covariance import LedoitWolf
            return LedoitWolf().fit(returns).covariance_
        except:
            return np.cov(returns.T)
