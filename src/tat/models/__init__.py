"""
TAT Prediction Models for Pharmacy Workflow Optimization

Comprehensive machine learning model library for medication preparation turnaround time
prediction and healthcare workflow optimization. Provides production-ready regression
models, ensemble methods, and automated hyperparameter optimization specifically designed
for pharmacy operations and clinical decision-making environments.

Key Components:
- Base Models: Abstract interfaces for TAT prediction model development
- Regression Models: Specialized regressors for medication preparation time prediction
- Ensemble Methods: Advanced stacking and voting approaches for improved accuracy
- Model Factory: Automated model creation and configuration management
- Hyperparameter Optimization: Automated tuning using Optuna for optimal performance
- Training Orchestration: End-to-end training pipeline for production deployment

Model Types:
- RidgeTATRegressor: Regularized linear regression optimized for TAT prediction
- RandomForestTATRegressor: Tree ensemble for non-linear TAT pattern recognition
- XGBoostTATRegressor: Gradient boosting for high-performance TAT forecasting
- StackingTATRegressor: Advanced ensemble combining multiple model strengths

Usage Example:
    from tat.models import TATModelFactory, XGBoostTATRegressor, OptunaTATOptimizer
    
    # Automated model creation
    factory = TATModelFactory()
    model = factory.create_regression_model('xgboost')
    
    # Manual model configuration with optimization
    xgb_model = XGBoostTATRegressor()
    optimizer = OptunaTATOptimizer(random_state=42)
    best_params = optimizer.optimize_regression_model(xgb_model, X_train, y_train, X_val, y_val, n_trials=100)
    
    # Production model training with optimized parameters
    optimized_model = factory.create_regression_model('xgboost', **best_params)
    optimized_model.fit(X_train, y_train)
"""

# Base model interfaces and abstract classes
from .base import BaseTATModel, BaseRegressionTATModel

# Regression model implementations
from .linear_model import RidgeTATRegressor
from .random_forest_model import RandomForestTATRegressor  
from .xgboost_model import XGBoostTATRegressor
from .ensemble_model import StackingTATRegressor# Model creation and training infrastructure
from .factory import TATModelFactory

# Hyperparameter optimization and tuning
from .hyperparameter_optimizer import OptunaTATOptimizer

# Model utilities and helper functions
from .model_utils import XGBoostCompatibilityHandler
# Note: model_utils also contains RidgeTATRegressor but we import from linear_model for consistency

# Module version for MLOps tracking and model lifecycle management
__version__ = "1.0.0"

# Public API for TAT prediction modeling in pharmacy analytics
__all__ = [
    # Base model interfaces
    'BaseTATModel',
    'BaseRegressionTATModel',
    
    # Regression model implementations  
    # TAT Regression Models    # Model creation and training
    'TATModelFactory',
    'TATTrainingOrchestrator',
    
    # Hyperparameter optimization
    'OptunaTATOptimizer',
    
    # Utilities
    'XGBoostCompatibilityHandler',
    
    # Module metadata
    '__version__',
]
