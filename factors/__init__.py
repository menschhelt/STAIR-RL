"""
Factors Module - Fama-French style factor system for cryptocurrency.

This module implements:
- Loading Matrix: B_i for each asset's factor exposures
- Crypto Factors: CMKT, CSMB, CMOM, CVOL, CLIQ (Liu et al. 2019)
- Characteristic Engine: Individual asset characteristics for sorting
- Portfolio Formation: Quintile/top-bottom portfolio construction

Key Formula:
    R_i = B_i * F + epsilon_i

Where:
    R_i = Asset i's return
    B_i = Factor loadings (beta coefficients)
    F = Factor returns
    epsilon_i = Residual alpha (unexplained return)
"""

from .loading_matrix import LoadingMatrixCalculator
from .characteristic_engine import CharacteristicEngine
from .portfolio_formation import PortfolioFormationEngine
from .crypto_factors import CryptoFactorEngine

__all__ = [
    'LoadingMatrixCalculator',
    'CharacteristicEngine',
    'PortfolioFormationEngine',
    'CryptoFactorEngine',
]
