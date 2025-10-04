"""
Healthcare Analytics Model Utilities for Healthcare TAT Prediction

Comprehensive utility functions and compatibility handlers supporting robust
healthcare model deployment, XGBoost version management, and linear regression
capabilities across diverse production environments and healthcare IT infrastructure.

Key Components:
- XGBoost compatibility handling for version-agnostic deployment
- Ridge regression for interpretable TAT prediction modeling
- Healthcare validation utilities ensuring clinical data quality
- Model evaluation metrics optimized for TAT prediction accuracy
- Production deployment utilities supporting MLOps workflows

"""
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# XGBoost imports with fallback handling for deployment environments
try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None
    XGBRegressor = None

# Scikit-learn imports for Ridge regression and utilities
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import base model for Ridge regression implementation
from .base import BaseRegressionTATModel

logger = logging.getLogger(__name__)

class XGBoostCompatibilityHandler:
    """
    XGBoost Version Compatibility Handler for Healthcare Deployment
    
    Manages XGBoost version differences and provides consistent API across
    diverse healthcare environments supporting robust TAT prediction deployment.
    Handles early stopping, validation, callbacks, and parameter compatibility
    issues across different XGBoost versions in production healthcare systems.
    
    XGBoost Version Management:
    - Modern callback system detection (XGBoost >= 1.6) for advanced training workflows
    - Legacy parameter compatibility ensuring deployment across diverse healthcare IT environments
    - Graceful degradation supporting fallback to standard fitting when validation fails
    - Version-specific capability mapping enabling optimized training strategy selection
    - Error handling and logging supporting production deployment troubleshooting and monitoring
    
    Attributes:
        xgboost_available: Boolean indicating XGBoost installation availability
        version: XGBoost version string for compatibility decision-making
        capabilities: Dictionary of version-specific capabilities and features
    
    Example:
        For version-agnostic XGBoost deployment:
        ```python
        # Initialize compatibility handler for robust deployment
        handler = XGBoostCompatibilityHandler()
        
        # Check capabilities before training
        if handler.capabilities['early_stopping']:
            model = handler.fit_with_validation(
                xgb_model, X_train, y_train, X_val, y_val
            )
        ```
    """
    
    def __init__(self):
        """
        Initialize XGBoost compatibility handler with comprehensive version detection.
        
        Performs automatic capability assessment supporting optimal training strategy
        selection and robust deployment across diverse healthcare IT environments.
        """
        self.xgboost_available = XGBOOST_AVAILABLE
        self.version = xgb.__version__ if XGBOOST_AVAILABLE else None
        self.capabilities = self._detect_capabilities()
        
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available - TAT prediction models requiring XGBoost will fail")
        else:
            logger.info(f"XGBoost compatibility handler initialized: version {self.version}")
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """
        Detect XGBoost version capabilities for healthcare deployment optimization.
        
        Performs comprehensive capability assessment enabling version-appropriate
        training strategy selection and optimal healthcare deployment configuration.
        
        Returns:
            Dict[str, Any]: Version-specific capabilities for compatibility handling
            including early stopping, callbacks, validation, and advanced features.
        """
        if not self.xgboost_available:
            return {
                'early_stopping': False,
                'callbacks': False,
                'validation': False,
                'version': None,
                'deployment_ready': False
            }
        
        capabilities = {
            'version': self.version,
            'early_stopping': True,
            'validation': True,
            'deployment_ready': True
        }
        
        # Detect modern callback system availability (XGBoost >= 1.6)
        try:
            from xgboost.callback import EarlyStopping
            capabilities['callbacks'] = True
            capabilities['modern_callbacks'] = True
            logger.debug("Modern XGBoost callback system detected")
        except ImportError:
            capabilities['callbacks'] = False
            capabilities['modern_callbacks'] = False
            capabilities['legacy_early_stopping'] = True
            logger.debug("Legacy XGBoost early stopping detected")
        
        # Detect additional advanced features for comprehensive capability mapping
        try:
            # Check for SHAP integration capability
            import shap
            capabilities['shap_integration'] = True
        except ImportError:
            capabilities['shap_integration'] = False
        
        return capabilities
    
    def fit_with_validation(self, model: XGBRegressor, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame, 
                          y_val: pd.Series, early_stopping_rounds: int = 10) -> XGBRegressor:
        """
        Fit XGBoost model with validation using version-appropriate training methods.
        
        Provides robust XGBoost training with early stopping across diverse healthcare
        environments supporting optimal model performance and deployment reliability.
        Automatically selects modern callback system or legacy parameters based on
        version detection ensuring consistent training behavior across IT infrastructures.
        
        Healthcare Training Optimization:
        - Version-appropriate early stopping ensuring optimal model performance across deployments
        - Validation monitoring supporting model quality assessment and overfitting prevention
        - Graceful fallback handling ensuring training completion even with validation failures
        - Comprehensive error handling supporting production deployment reliability
        - Training logging enabling audit trails and troubleshooting support workflows
        
        Training Strategy Selection:
        - Modern callbacks: Advanced early stopping with save_best functionality (XGBoost >= 1.6)
        - Legacy parameters: Traditional early_stopping_rounds parameter for compatibility
        - Fallback training: Standard fitting without validation when early stopping fails
        - Error recovery: Comprehensive exception handling ensuring training completion
        - Production logging: Detailed training status reporting for healthcare deployment monitoring
        
        Args:
            model: XGBoost regressor to train with healthcare-optimized configuration
            X_train: Training feature matrix with clinical and operational variables
            y_train: Training target variable (TAT in minutes) for prediction modeling
            X_val: Validation feature matrix for early stopping and performance monitoring
            y_val: Validation target variable for model quality assessment
            early_stopping_rounds: Patience rounds for early stopping optimization
            
        Returns:
            XGBRegressor: Fitted XGBoost model ready for healthcare TAT prediction
            with validated training completion and optimal performance characteristics.
        
        Raises:
            ImportError: XGBoost not available preventing TAT prediction model training
        
        Example:
            For robust XGBoost training with validation:
            ```python
            # Train with version-appropriate validation
            handler = XGBoostCompatibilityHandler()
            trained_model = handler.fit_with_validation(
                xgb_model, X_train, y_train, X_val, y_val
            )
            ```
        """
        if not self.xgboost_available:
            raise ImportError("XGBoost not available for healthcare TAT prediction training")
        
        try:
            if self.capabilities.get('modern_callbacks', False):
                # Modern XGBoost with advanced callback system for optimal training
                from xgboost.callback import EarlyStopping
                
                logger.debug("Using modern XGBoost callback system for training")
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[EarlyStopping(rounds=early_stopping_rounds, save_best=True)],
                    verbose=False
                )
                logger.info("XGBoost training completed with modern callbacks and early stopping")
                
            elif self.capabilities.get('legacy_early_stopping', False):
                # Legacy XGBoost with traditional early_stopping_rounds parameter
                logger.debug("Using legacy XGBoost early stopping for training")
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False
                )
                logger.info("XGBoost training completed with legacy early stopping")
                
            else:
                # Standard XGBoost fitting with validation set but no early stopping
                logger.debug("Using standard XGBoost training with validation monitoring")
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                logger.info("XGBoost training completed with validation monitoring")
                
        except Exception as e:
            logger.warning(f"XGBoost validation training failed: {str(e)}, falling back to standard fit")
            # Robust fallback to standard fitting ensuring training completion
            try:
                model.fit(X_train, y_train)
                logger.info("XGBoost training completed with fallback standard fitting")
            except Exception as fallback_error:
                logger.error(f"XGBoost fallback training failed: {str(fallback_error)}")
                raise RuntimeError(f"XGBoost training failed completely: {str(fallback_error)}")
        
        return model
    
    def validate_model(self, model: XGBRegressor) -> Dict[str, Any]:
        """
        Validate XGBoost model for healthcare deployment readiness.
        
        Performs comprehensive model validation ensuring deployment safety and
        performance characteristics meeting Healthcare TAT prediction requirements.
        
        Args:
            model: Trained XGBoost model for validation assessment
            
        Returns:
            Dict[str, Any]: Comprehensive validation results including deployment readiness,
            model characteristics, and healthcare compliance assessment.
        """
        validation_results = {
            'model_type': 'XGBoost',
            'xgboost_version': self.version,
            'deployment_ready': False,
            'validation_passed': False,
            'issues': []
        }
        
        try:
            # Check model training completion
            if hasattr(model, 'feature_importances_'):
                validation_results['feature_importance_available'] = True
            else:
                validation_results['issues'].append("Feature importance not available")
            
            # Check model attributes for healthcare deployment
            if hasattr(model, 'n_estimators') and model.n_estimators > 0:
                validation_results['ensemble_configured'] = True
                validation_results['n_estimators'] = model.n_estimators
            else:
                validation_results['issues'].append("Invalid ensemble configuration")
            
            # Validate healthcare-appropriate parameters
            if hasattr(model, 'max_depth') and 3 <= model.max_depth <= 10:
                validation_results['interpretability_preserved'] = True
            else:
                validation_results['issues'].append("Tree depth may compromise interpretability")
            
            # Overall validation assessment
            if len(validation_results['issues']) == 0:
                validation_results['validation_passed'] = True
                validation_results['deployment_ready'] = True
            
        except Exception as e:
            validation_results['issues'].append(f"Validation error: {str(e)}")
        
        return validation_results

