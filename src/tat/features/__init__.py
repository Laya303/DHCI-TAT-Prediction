"""
Feature Engineering Components for Pharmacy TAT Optimization

Comprehensive feature engineering toolkit for medication preparation turnaround time
analysis and healthcare workflow optimization. Provides specialized processors for
categorical variables, laboratory values, operational metrics, and temporal patterns
specifically designed for pharmacy operations and clinical decision-making workflows.
"""

# Core feature engineering components for pharmacy TAT analysis
from .categoricals import CategoricalEncoder
from .cleaners import Cleaner  
from .labs import LabProcessor
from .operational import OperationalEngineer

# Temporal feature engineering submodule (imported as namespace)
from . import temporal

# Module version for healthcare analytics pipeline tracking
__version__ = "1.0.0"

# Public API for feature engineering in pharmacy TAT analysis
__all__ = [
    'CategoricalEncoder',
    'Cleaner',
    'LabProcessor', 
    'OperationalEngineer',
    'temporal',
    '__version__',
]
