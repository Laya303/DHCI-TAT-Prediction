"""
Base classes for TAT prediction models.

Provides abstract interfaces and common functionality for medication
preparation turnaround time prediction models.
- MLOps-ready save/load functionality enabling automated deployment and monitoring
- Clinical  Numerical Features aligned with pharmacy workflow optimization objectives
- Error handling and validation ensuring robust operation in healthcare environments

Note:
    Essential foundation for healthcare TAT prediction system supporting pharmacy
    workflow optimization and medication preparation efficiency through standardized
    healthcare analytics model architecture enabling clinical operations excellence.
"""
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

class BaseTATModel(ABC):
    """
    Abstract Base Class for Healthcare TAT Prediction Models
    
    Comprehensive foundation class enforcing healthcare-specific requirements and
    providing standardized functionality for medication preparation turnaround time
    prediction models. Essential for healthcare pharmacy workflow optimization
    supporting consistent model development and clinical operations excellence.
    
    Healthcare Model Architecture:
    - Abstract interface ensuring clinical requirements compliance across model types
    - Healthcare metadata management supporting audit trails and regulatory documentation
    - Clinical performance evaluation optimized for 60-minute TAT threshold assessment
    - Production lifecycle management with MLOps integration and deployment readiness
    - Standardized error handling ensuring robust operation in healthcare environments
    
    Clinical Requirements:
    - Mandatory fit/predict interface supporting consistent healthcare analytics workflows
    - Healthcare-specific hyperparameter spaces ensuring clinical interpretability
    - 60-minute threshold accuracy evaluation supporting pharmacy quality standards
    - Clinical metadata preservation enabling audit trails and regulatory compliance
    - Production-ready save/load functionality supporting automated deployment pipelines
    
    Args:
        random_state: Reproducibility seed ensuring consistent healthcare model results
                     across training runs and deployment environments for validation.
    
    Attributes:
        model: Underlying ML model instance fitted for TAT prediction
        is_fitted: Training status flag ensuring prediction safety in healthcare context
        metadata: Healthcare-specific model information supporting clinical documentation
        
    Example:
        For healthcare TAT model development:
        ```python
        # Custom model inheriting healthcare base class
        class CustomTATModel(BaseTATModel):
            def fit(self, X, y):
                # Healthcare-specific training logic
                return self
                
            def predict(self, X):
                # Clinical prediction implementation
                return predictions
        ```
    
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Healthcare TAT Prediction Model Foundation
        
        Establishes base healthcare analytics model with clinical metadata and
        reproducibility configuration supporting healthcare pharmacy workflow
        optimization and production deployment requirements.
        
        Args:
            random_state: Reproducibility seed for consistent healthcare model results
                         ensuring validation and compliance across training and deployment.
        
        Note:
            Essential for healthcare analytics quality assurance ensuring healthcare
            pharmacy workflow optimization through consistent model initialization
            supporting medication preparation efficiency and clinical operations excellence.
        """
        self.random_state = random_state
        self.model = None  # Underlying ML model instance - set by subclass implementation
        self.is_fitted = False  # Training status flag - prevents unsafe prediction calls
        
        # Healthcare-specific metadata for clinical documentation and audit trails
        self.metadata = {
            'model_type': self.__class__.__name__,  # Model class for clinical documentation
            'healthcare_context': 'TAT Prediction',  # Clinical application context
            'clinical_objective': '60-minute threshold optimization',  # Primary quality goal
            'random_state': random_state  # Reproducibility configuration
        }
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseTATModel':
        """
        Train TAT Prediction Model on Healthcare Data
        
        Abstract method enforcing standardized training interface across all healthcare
        TAT prediction models. Essential for healthcare pharmacy workflow optimization
        ensuring consistent model development supporting clinical operations excellence.
        
        Args:
            X: Healthcare feature matrix containing clinical, operational, and temporal
               variables supporting comprehensive medication preparation workflow analysis.
            y: TAT target variable in minutes or binary 60-minute threshold classification
               supporting pharmacy quality monitoring and workflow optimization objectives.
            
        Returns:
            BaseTATModel: Fitted model instance ready for healthcare TAT prediction
            and clinical decision support with validated training completion status.
        
        Raises:
            NotImplementedError: Subclass must implement healthcare-specific training logic
            ValueError: Invalid healthcare data compromising clinical model development

        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate TAT Predictions for Pharmacy Workflow Optimization
        
        Abstract method enforcing standardized prediction interface across all healthcare
        TAT models. Essential for healthcare pharmacy operations supporting consistent
        clinical decision-making and workflow optimization through reliable predictions.
        
        Args:
            X: Healthcare feature matrix for TAT prediction containing clinical,
               operational, and temporal variables supporting pharmacy workflow analysis.
            
        Returns:
            np.ndarray: TAT predictions in minutes or threshold probabilities supporting
            pharmacy workflow optimization and clinical decision-making requirements.
        
        Raises:
            NotImplementedError: Subclass must implement healthcare-specific prediction logic
            ValueError: Model not fitted or invalid input compromising healthcare safety
        """
        pass
    
    @abstractmethod
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """
        Define Healthcare-Appropriate Hyperparameter Search Spaces
        
        Abstract method enforcing clinical interpretability requirements across all
        healthcare TAT models. Essential for healthcare pharmacy workflow optimization
        ensuring hyperparameter optimization aligned with clinical needs and constraints.
        
        Returns:
            Dict[str, Any]: Hyperparameter search space dictionary defining clinical
            optimization ranges supporting healthcare analytics and workflow optimization.
        
        Raises:
            NotImplementedError: Subclass must define healthcare-specific parameter ranges
        
        Example:
            For healthcare hyperparameter definition:
            ```python
            def get_hyperparameter_space(self):
                return {
                    'max_depth': [3, 5, 7],  # Clinical interpretability
                    'n_estimators': [50, 100, 200],  # Performance balance
                    'learning_rate': [0.01, 0.1, 0.2]  # Convergence control
                }
            ```

        """
        pass
    
    def evaluate_healthcare_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate Model Performance Using Healthcare-Specific Clinical Metrics
        
        Comprehensive performance evaluation optimized for healthcare pharmacy
        workflow optimization and 60-minute TAT threshold compliance requirements.
        Focuses on clinically relevant metrics supporting pharmacy operations excellence
        and patient care throughput enhancement through evidence-based assessment.
        
        Args:
            y_true: Actual TAT values in minutes from medication preparation workflow
            y_pred: Model predictions in minutes supporting clinical decision-making
            
        Returns:
            Dict[str, float]: Comprehensive healthcare  Numerical Features including
            clinical accuracy, threshold compliance, and workflow optimization indicators.
        
        Raises:
            ValueError: Mismatched array lengths compromising healthcare evaluation integrity
        
        Example:
            For clinical performance assessment:
            ```python
            # Evaluate model on healthcare test data
            metrics = model.evaluate_healthcare_metrics(y_test, predictions)
            
            print(f"Clinical Accuracy: {metrics['MAE']:.1f} minutes MAE")
            print(f"Threshold Compliance: {metrics['threshold_60min_accuracy']:.1f}%")
            print(f"Healthcare Score: {metrics['healthcare_score']:.1f}/100")
            ```
        
        """
        # Validate input arrays for healthcare safety and analysis integrity
        if len(y_true) != len(y_pred):
            raise ValueError("Prediction and true value arrays must have same length")
        
        # Core prediction accuracy metrics for clinical interpretation
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Root mean square error in minutes
        mae = mean_absolute_error(y_true, y_pred)  # Mean absolute error in minutes  
        median_ae = np.median(np.abs(y_true - y_pred))  # Robust accuracy measure
        
        # Healthcare-specific performance indicators for workflow optimization
        within_10min = np.mean(np.abs(y_true - y_pred) <= 10) * 100  # Excellent accuracy band
        within_30min = np.mean(np.abs(y_true - y_pred) <= 30) * 100  # Acceptable accuracy band
        
        # Critical 60-minute threshold accuracy for healthcare operations compliance
        threshold_60min_accuracy = np.mean((y_true > 60) == (y_pred > 60)) * 100
        
        return {
            'RMSE': rmse,  # Prediction accuracy in minutes
            'MAE': mae,  # Mean absolute prediction error
            'MedianAE': median_ae,  # Robust accuracy indicator
            'within_10min_pct': within_10min,  # Excellent prediction percentage
            'within_30min_pct': within_30min,  # Acceptable prediction percentage
            'threshold_60min_accuracy': threshold_60min_accuracy,  # Quality compliance accuracy
            'healthcare_score': (within_30min + threshold_60min_accuracy) / 2  # Composite performance
        }
    
    def save_model(self, filepath: Path) -> None:
        """
        Save Trained Healthcare Model with Clinical Metadata for MLOps Deployment
        
        Production-ready model persistence supporting healthcare healthcare analytics
        deployment and MLOps integration. Preserves clinical metadata and training status
        enabling automated deployment, monitoring, and audit trail requirements for
        pharmacy workflow optimization and regulatory compliance.
        
        Args:
            filepath: Model save location supporting MLOps deployment and artifact management
            
        Raises:
            ValueError: Unfitted model preventing unsafe healthcare deployment
            IOError: File system issues affecting healthcare MLOps integration
        
        Example:
            For healthcare model deployment preparation:
            ```python
            # Train model for production deployment
            model.fit(X_train, y_train)
            
            # Save for MLOps integration
            model.save_model(Path("models/tat_predictor_v1.joblib"))
            ```
        
        """
        # Validate model training status for safe healthcare deployment
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Ensure directory exists for healthcare MLOps artifact management
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model with comprehensive healthcare metadata for audit trails
        save_dict = {
            'model': self.model,  # Trained model instance
            'metadata': self.metadata,  # Healthcare clinical context
            'is_fitted': self.is_fitted  # Training status validation
        }
        
        # Persist model with healthcare metadata for production deployment
        joblib.dump(save_dict, filepath)
        logger.info(f"TAT model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Path) -> 'BaseTATModel':
        """
        Load Trained Healthcare Model with Validation for Production Deployment
        
        Production-ready model loading supporting healthcare healthcare analytics
        deployment and MLOps integration. Validates model integrity and reconstructs
        clinical metadata enabling safe healthcare deployment with audit trail preservation
        and regulatory compliance for pharmacy workflow optimization.
        
        Args:
            filepath: Model file location for healthcare artifact loading and validation
            
        Returns:
            BaseTATModel: Loaded model instance ready for healthcare TAT prediction
            with preserved clinical metadata and validated training status.
        
        Raises:
            FileNotFoundError: Model artifact not accessible for healthcare deployment
            ValueError: Invalid model data compromising healthcare analytics integrity
        
        Example:
            For production healthcare model deployment:
            ```python
            # Load trained model for healthcare deployment
            model = BaseTATModel.load_model(Path("models/tat_predictor_v1.joblib"))
            
            # Use for clinical TAT prediction
            predictions = model.predict(X_new)
            ```
    
        """
        # Validate model file accessibility for healthcare deployment
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model artifact with healthcare metadata preservation
        save_dict = joblib.load(filepath)
        
        # Reconstruct model instance with clinical context preservation
        instance = cls(random_state=save_dict['metadata']['random_state'])
        instance.model = save_dict['model']  # Trained model instance
        instance.metadata = save_dict['metadata']  # Healthcare clinical context
        instance.is_fitted = save_dict['is_fitted']  # Training status validation
        
        logger.info(f"TAT model loaded from {filepath}")
        return instance


class BaseRegressionTATModel(BaseTATModel):
    """
    Healthcare Regression Base Class for Continuous TAT Prediction
    
    Specialized foundation class for regression-based medication preparation TAT
    prediction supporting healthcare pharmacy workflow optimization. Handles
    continuous TAT prediction with healthcare-optimized target transformation
    enabling accurate workflow timing analysis and bottleneck identification.
    
    Target Transformation:
    - Log1p transformation handling skewed TAT distributions common in healthcare
    - Inverse transformation ensuring predictions return to interpretable minutes scale
    - Distribution normalization supporting improved regression model performance
    - Healthcare data characteristics accommodation for robust prediction accuracy
    - Clinical interpretability preservation through proper scaling and transformation
    
    Args:
        random_state: Reproducibility seed ensuring consistent healthcare regression results
        
    Attributes:
        target_transform: Transformation strategy for skewed healthcare TAT data
        
    Example:
        For healthcare TAT regression model:
        ```python
        # Custom regression model for TAT prediction
        class XGBoostTATRegressor(BaseRegressionTATModel):
            def fit(self, X, y):
                # Transform target for skewed data
                y_transformed = self._transform_target(y)
                # Fit regression model
                return self
        ```
    
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Healthcare Regression TAT Model with Target Transformation
        
        Establishes regression-specific healthcare analytics model with target
        transformation configuration optimized for skewed TAT distributions.
        Supporting healthcare pharmacy workflow optimization through continuous
        prediction capability and clinical interpretability requirements.
        
        Args:
            random_state: Reproducibility seed ensuring consistent healthcare regression
                         results across training runs and deployment environments.
    
        """
        super().__init__(random_state)
        # Default transformation for skewed healthcare TAT data distributions
        self.target_transform = 'log1p'  # Log1p handles zero values common in healthcare
    
    def _transform_target(self, y: pd.Series) -> pd.Series:
        """
        Apply Target Transformation for Skewed Healthcare TAT Data
        
        Transforms target variable to handle skewed TAT distributions common in
        healthcare data. Essential for healthcare pharmacy workflow optimization
        supporting improved regression model performance through appropriate data
        preprocessing and distribution normalization for accurate predictions.
        
        Healthcare Target Transformation:
        - Log1p transformation accommodating skewed TAT distributions in healthcare data
        - Zero value handling ensuring robust transformation for all TAT ranges
        - Distribution normalization improving regression model training and convergence
        - Clinical data characteristics accommodation supporting accurate model development
        - Performance optimization through appropriate target scaling and preprocessing
        
        Args:
            y: Original TAT target values in minutes from medication preparation workflow
            
        Returns:
            pd.Series: Transformed target values optimized for regression model training
            supporting improved convergence and prediction accuracy.
        """
        if self.target_transform == 'log1p':
            return np.log1p(y)  # Log1p transformation for skewed healthcare data
        return y  # Return original values if no transformation specified
    
    def _inverse_transform_target(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse Transform Predictions to Original Clinical TAT Scale
        
        Converts transformed predictions back to interpretable minutes scale for
        clinical decision-making. Essential for healthcare pharmacy workflow
        optimization ensuring TAT predictions maintain clinical interpretability
        and actionable insights for healthcare stakeholders and operations teams.
        
        Healthcare Inverse Transformation:
        - Expm1 inverse transformation returning predictions to minutes scale
        - Clinical interpretability restoration ensuring actionable healthcare insights
        - Original scale preservation supporting pharmacy workflow decision-making
        - Healthcare stakeholder communication through interpretable prediction values
        - Production deployment preparation with clinical scale consistency
        
        Args:
            y_transformed: Model predictions in transformed scale from regression training
            
        Returns:
            np.ndarray: TAT predictions in original minutes scale supporting clinical
            decision-making and pharmacy workflow optimization requirements.
        
        Note:
            Critical for clinical interpretation supporting healthcare pharmacy workflow
            optimization through interpretable TAT predictions enabling medication preparation
            efficiency and healthcare operations excellence via actionable insights.
        """
        if self.target_transform == 'log1p':
            return np.expm1(y_transformed)  # Inverse log1p transformation to minutes
        return y_transformed  # Return predictions if no transformation applied
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate TAT Predictions in Original Clinical Minutes Scale
        
        Produces continuous TAT predictions in interpretable minutes supporting
        healthcare pharmacy workflow optimization and clinical decision-making.
        Handles target transformation ensuring predictions maintain clinical
        interpretability for healthcare stakeholders and operations excellence.
        
        Args:
            X: Healthcare feature matrix containing clinical and operational variables
               supporting comprehensive medication preparation workflow analysis.
            
        Returns:
            np.ndarray: TAT predictions in minutes scale supporting clinical decision-making
            and pharmacy workflow optimization through interpretable continuous forecasts.
        
        Raises:
            ValueError: Unfitted model preventing unsafe healthcare prediction operation
        
        Example:
            For clinical TAT prediction:
            ```python
            # Generate interpretable TAT predictions
            tat_predictions = model.predict(X_test)
            
            # Clinical interpretation in minutes
            print(f"Average predicted TAT: {tat_predictions.mean():.1f} minutes")
            ```

        """
        # Validate model training status for safe healthcare prediction
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get transformed predictions from trained regression model
        y_pred_transformed = self.model.predict(X)
        
        # Transform back to original clinical TAT scale (minutes) for interpretation
        return self._inverse_transform_target(y_pred_transformed)