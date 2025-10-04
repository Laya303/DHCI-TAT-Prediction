"""
Healthcare Analytics Ensemble Model for Medication Preparation TAT Prediction

Advanced stacking ensemble implementation supporting healthcare pharmacy workflow
optimization and 60-minute TAT threshold compliance initiatives. Combines multiple
base learners with meta-learning architecture enabling robust medication preparation
turnaround time prediction across diverse clinical scenarios and operational conditions.

Key Features:
- Multi-algorithm ensemble combining Ridge, Random Forest, and XGBoost for robustness
- Healthcare-optimized stacking architecture supporting diverse clinical scenarios
- Clinical interpretability through base model importance and meta-learner analysis
- Production-ready ensemble configuration with MLOps integration capabilities
- 60-minute TAT threshold optimization through ensemble diversity and accuracy

Clinical Applications:
- Robust TAT prediction across diverse medication preparation scenarios and complexities
- Multi-perspective bottleneck analysis through complementary algorithmic approaches
- Pharmacy operations optimization via ensemble-based predictive analytics
- Clinical decision support with improved prediction confidence through model diversity
- Healthcare quality monitoring with enhanced accuracy for 60-minute threshold compliance

"""
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .base import BaseRegressionTATModel
from .model_utils import XGBoostCompatibilityHandler

logger = logging.getLogger(__name__)

class StackingTATRegressor(BaseRegressionTATModel):
    """
    Healthcare Stacking Ensemble for Advanced TAT Prediction
    
    Comprehensive stacking ensemble regressor optimized for healthcare medication
    preparation TAT prediction and pharmacy workflow optimization. Combines diverse
    base learners with meta-learning architecture providing superior prediction
    accuracy and robustness across varied clinical scenarios and operational conditions.
    
    Healthcare Ensemble Architecture:
    - Ridge regression base: Linear interpretability for clinical coefficient analysis
    - Random Forest base: Non-linear pattern capture and feature interaction modeling
    - XGBoost base: Gradient boosting for complex healthcare workflow relationships
    - Ridge meta-learner: Optimal base model combination through regularized weighting
    - Cross-validation stacking: Robust ensemble training preventing overfitting
    
    Args:
        random_state: Reproducibility seed ensuring consistent ensemble results
        **kwargs: Ensemble configuration parameters including:
                 - meta_alpha: Meta-learner regularization for ensemble combination
                 - xgb_n_estimators: XGBoost tree count for complex pattern modeling
                 - rf_n_estimators: Random Forest tree count for ensemble diversity
                 - ridge_alpha: Ridge regularization for linear interpretability
                 - cv_folds: Cross-validation folds for robust ensemble training
    
    Attributes:
        base_models: List of (name, model) tuples for ensemble base learners
        meta_learner: Ridge regression meta-learner for optimal combination
        ensemble_params: Configuration parameters for healthcare ensemble optimization
        compatibility_handler: XGBoost version compatibility management
    
    Example:
        For advanced TAT prediction ensemble:
        ```python
        # Initialize healthcare-optimized ensemble
        ensemble = StackingTATRegressor(
            random_state=42,
            meta_alpha=1.0,
            xgb_n_estimators=100,
            rf_n_estimators=100
        )
        
        # Train on medication preparation data
        ensemble.fit(X_train, y_train)
        
        # Generate robust TAT predictions
        predictions = ensemble.predict(X_test)
        ```

    """
    
    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize Healthcare Stacking Ensemble with Clinical Optimization
        
        Establishes advanced ensemble architecture optimized for healthcare
        medication preparation TAT prediction and pharmacy workflow optimization.
        Configures base learners and meta-learner with healthcare-specific parameters
        ensuring clinical interpretability and production deployment readiness.
        
        Healthcare Ensemble Initialization:
        - Base regression model foundation with ensemble-specific target transformation
        - XGBoost compatibility management ensuring robust cross-platform deployment
        - Healthcare-optimized ensemble parameter extraction and validation
        - Base model architecture construction with clinical interpretability focus
        - Meta-learner configuration for optimal algorithmic combination
        
        Ensemble Configuration Components:
        - Meta-learner regularization supporting stable ensemble combination
        - Base model parameters optimized for healthcare TAT prediction scenarios
        - Cross-validation setup ensuring robust ensemble training and validation
        - Production deployment preparation with MLOps integration capabilities
        - Clinical metadata management supporting audit trails and documentation
        
        Args:
            random_state: Reproducibility seed for consistent ensemble behavior
            **kwargs: Ensemble configuration including meta_alpha, xgb_n_estimators,
                     rf_n_estimators, ridge_alpha, and cv_folds parameters.
        
        """
        super().__init__(random_state)
        
        # XGBoost compatibility handler for robust cross-platform deployment
        self.compatibility_handler = XGBoostCompatibilityHandler()
        
        # Extract healthcare-optimized ensemble parameters with clinical defaults
        self.ensemble_params = {
            'meta_alpha': kwargs.get('meta_alpha', 1.0),          # Meta-learner regularization
            'xgb_n_estimators': kwargs.get('xgb_n_estimators', 100),  # XGBoost complexity
            'rf_n_estimators': kwargs.get('rf_n_estimators', 100),    # Random Forest diversity
            'ridge_alpha': kwargs.get('ridge_alpha', 10.0),           # Ridge regularization
            'cv_folds': kwargs.get('cv_folds', 5)                     # Cross-validation robustness
        }
        
        # Build healthcare-optimized ensemble components for TAT prediction
        self.base_models = self._build_base_models()
        self.meta_learner = self._build_meta_learner()
        
        # Initialize stacking regressor with healthcare configuration
        self.model = StackingRegressor(
            estimators=self.base_models,         # Diverse base learner ensemble
            final_estimator=self.meta_learner,   # Ridge meta-learner for combination
            cv=self.ensemble_params['cv_folds'], # Cross-validation for robustness
            n_jobs=-1                            # Parallel processing for efficiency
        )
        
        # Update healthcare metadata with ensemble-specific information
        self.metadata.update({
            'algorithm': 'Stacking Ensemble Regression',
            'base_models': [name for name, _ in self.base_models],
            'meta_learner': 'Ridge',
            'healthcare_optimization': 'TAT prediction with ensemble robustness',
            'interpretability': 'Limited - ensemble black box with base model insights',
            'clinical_focus': '60-minute TAT threshold optimization',
            'ensemble_diversity': 'Linear + Tree-based + Gradient boosting'
        })
    
    def _build_base_models(self) -> List[Tuple[str, Any]]:
        """
        Build Healthcare-Optimized Base Models for Stacking Ensemble
        
        Constructs diverse base learners optimized for healthcare medication
        preparation TAT prediction supporting varied clinical scenarios and
        operational conditions. Each base model provides unique perspective
        on healthcare workflow patterns enabling robust ensemble predictions.
        
        Healthcare Base Model Architecture:
        - Ridge regression: Linear interpretability for clinical coefficient analysis
        - Random Forest: Non-linear pattern capture with feature interaction modeling
        - XGBoost: Advanced gradient boosting for complex healthcare relationships
        - Healthcare parameter optimization: Clinical interpretability and performance balance
        - Production configuration: Efficient deployment with parallel processing support
        
        Base Model Optimization:
        - Ridge regularization preventing overfitting in healthcare feature spaces
        - Random Forest depth and sampling controls for diverse tree generation
        - XGBoost learning rate and tree parameters for stable healthcare predictions
        - Cross-platform compatibility ensuring robust production deployment
        - Parallel processing configuration maximizing training and inference efficiency
        
        Returns:
            List[Tuple[str, Any]]: Base model ensemble with healthcare-optimized
            configurations supporting diverse TAT prediction scenarios and clinical needs.
        
        """
        # Base Ridge regression with healthcare-appropriate regularization
        ridge_base = Ridge(
            alpha=self.ensemble_params['ridge_alpha'],  # Regularization for stability
            random_state=self.random_state              # Reproducible clinical results
        )
        
        # Base Random Forest with ensemble diversity optimization
        rf_base = RandomForestRegressor(
            n_estimators=self.ensemble_params['rf_n_estimators'],  # Tree diversity
            max_depth=6,                    # Clinical interpretability balance
            min_samples_split=5,            # Robust split requirements
            random_state=self.random_state, # Consistent healthcare results
            n_jobs=-1                       # Parallel processing efficiency
        )
        
        # Base XGBoost with healthcare-optimized gradient boosting parameters
        xgb_params = {
            'n_estimators': self.ensemble_params['xgb_n_estimators'],  # Boosting rounds
            'max_depth': 4,                 # Tree depth for interpretability
            'learning_rate': 0.1,           # Conservative learning for stability
            'random_state': self.random_state,  # Reproducible training
            'n_jobs': -1                    # Parallel processing support
        }
        
        # Add advanced tree method if supported for efficiency
        try:
            xgb_params['tree_method'] = 'hist'  # Histogram-based tree construction
        except:
            pass  # Fallback to default method for compatibility
        
        xgb_base = XGBRegressor(**xgb_params)
        
        return [
            ('ridge', ridge_base),    # Linear interpretability
            ('rf', rf_base),          # Non-linear pattern capture
            ('xgb', xgb_base)         # Gradient boosting complexity
        ]
    
    def _build_meta_learner(self) -> Ridge:
        """
        Build Meta-Learner for Optimal Base Model Combination
        
        Constructs Ridge regression meta-learner optimized for combining base model
        predictions in healthcare healthcare TAT prediction system. Provides
        regularized weighting of diverse algorithmic perspectives ensuring stable
        ensemble predictions and clinical interpretability of model contributions.
        
        Healthcare Meta-Learning Features:
        - Ridge regularization preventing overfitting in base model combination
        - Linear combination enabling interpretable base model contribution analysis
        - Healthcare parameter optimization balancing performance and interpretability
        - Production stability ensuring consistent ensemble behavior across deployments
        - Clinical audit support through transparent combination coefficient analysis
        
        Meta-Learner Configuration:
        - Alpha regularization: Stable combination preventing base model dominance
        - Random state: Reproducible ensemble behavior for clinical validation
        - Linear architecture: Interpretable base model contribution weights
        - Coefficient accessibility: Clinical insight into algorithmic contributions
        - Production readiness: Efficient inference supporting real-time predictions
        
        Returns:
            Ridge: Configured meta-learner for optimal base model combination
            supporting stable ensemble predictions and clinical interpretability.
        
        """
        return Ridge(
            alpha=self.ensemble_params['meta_alpha'],  # Regularization for stable combination
            random_state=self.random_state             # Reproducible meta-learning
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_data: Optional[tuple] = None) -> 'StackingTATRegressor':
        """
        Train Stacking Ensemble Model on Healthcare TAT Data
        
        Comprehensive ensemble training optimized for healthcare medication
        preparation TAT prediction and pharmacy workflow optimization. Trains
        diverse base learners and meta-learner using cross-validation stacking
        ensuring robust predictions across varied clinical scenarios.
        
        Healthcare Ensemble Training:
        - Target transformation handling skewed TAT distributions common in healthcare
        - Cross-validation stacking preventing overfitting and ensuring robustness
        - Base model diversity training capturing complementary healthcare patterns
        - Meta-learner optimization for optimal algorithmic combination
        - Training metadata preservation supporting audit trails and documentation
        
        Training Workflow Components:
        - Skewed target transformation: Log1p handling for healthcare TAT distributions
        - Stacking regressor training: Cross-validated base model and meta-learner fitting
        - Metadata management: Training statistics and configuration preservation
        - Production preparation: Model state management for healthcare deployment
        - Clinical validation: Training completion status for safe prediction operation
        
        Args:
            X: Healthcare feature matrix containing clinical, operational, and temporal
               variables supporting comprehensive medication preparation workflow analysis.
            y: TAT target variable in minutes supporting continuous prediction and
               60-minute threshold compliance assessment for pharmacy quality monitoring.
            validation_data: Optional (X_val, y_val) tuple - maintained for API
                           consistency but not used in stacking ensemble training.
            
        Returns:
            StackingTATRegressor: Fitted ensemble model ready for healthcare TAT
            prediction and clinical decision support with validated training completion.
        
        Example:
            For healthcare ensemble training:
            ```python
            # Train ensemble on medication preparation data
            ensemble.fit(X_train, y_train)
            
            # Validate training completion
            assert ensemble.is_fitted
            print(f"Base models: {ensemble.metadata['base_models']}")
            ```
        """
        # Transform target variable for skewed healthcare TAT distribution
        y_transformed = self._transform_target(y)
        
        # Fit the stacking ensemble model with cross-validation robustness
        self.model.fit(X, y_transformed)
        
        # Set training completion status for safe healthcare prediction operation
        self.is_fitted = True
        
        # Store comprehensive training metadata for clinical documentation
        self.metadata.update({
            'training_samples': len(X),                          # Dataset size
            'feature_count': X.shape[1],                         # Feature dimensionality
            'target_transform': self.target_transform,           # Transformation method
            'ensemble_params': self.ensemble_params,             # Configuration
            'base_model_count': len(self.base_models),           # Ensemble diversity
            'training_completed': True,                          # Validation status
            'clinical_deployment_ready': True                    # Production readiness
        })

        logger.info(f"Stacking ensemble trained: {len(self.base_models)} base models, {len(X):,} samples")
        return self
    
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """
        Define Healthcare-Appropriate Hyperparameter Ranges for Ensemble Optimization
        
        Provides comprehensive hyperparameter search space optimized for healthcare
        medication preparation TAT prediction and clinical interpretability requirements.
        Focuses on meta-learner regularization and key base model parameters ensuring
        healthcare deployment readiness and 60-minute threshold compliance optimization.
        
        Healthcare Hyperparameter Optimization:
        - Meta-learner regularization: Alpha ranges ensuring stable ensemble combination
        - Base model complexity: Tree counts balanced for performance and interpretability
        - Clinical interpretability: Parameter ranges maintaining healthcare transparency
        - Performance boundaries: Optimization space supporting 60-minute TAT accuracy
        - Production efficiency: Parameter ranges enabling real-time deployment
        
        Hyperparameter Categories:
        - Meta-learning parameters: Alpha regularization for stable base model combination
        - XGBoost configuration: Tree count and complexity for gradient boosting optimization
        - Random Forest setup: Estimator count for ensemble diversity and pattern capture
        - Ridge regularization: Alpha values for linear interpretability and stability
        - Cross-validation: Fold configuration for robust ensemble training validation
        
        Returns:
            Dict[str, Any]: Comprehensive hyperparameter search space with healthcare
            optimization ranges supporting clinical interpretability and deployment needs.
        
        Example:
            For ensemble hyperparameter optimization:
            ```python
            # Get healthcare-optimized search space
            param_space = ensemble.get_hyperparameter_space()
            
            # Use with Optuna for clinical optimization
            study = optuna.create_study()
            study.optimize(objective, n_trials=100)
            ```
        """
        return {
            'meta_alpha': ('float', 0.1, 100.0, 'log'),        # Meta-learner regularization
            'xgb_n_estimators': ('int', 50, 150),              # XGBoost complexity control
            'rf_n_estimators': ('int', 50, 150),               # Random Forest diversity
            'ridge_alpha': ('float', 1.0, 100.0, 'log')       # Ridge regularization
        }
    
    def get_base_model_importance(self) -> Dict[str, Any]:
        """
        Extract Feature Importance from Base Models for Clinical Insights
        
        Provides comprehensive feature importance analysis from interpretable base
        models supporting healthcare pharmacy workflow optimization and bottleneck
        identification. Extracts complementary perspectives from diverse algorithms
        enabling clinical understanding of medication preparation TAT drivers.
        
        Healthcare Feature Importance Analysis:
        - Tree-based importance: Random Forest and XGBoost feature ranking for pattern analysis
        - Linear coefficients: Ridge regression weights for direct clinical interpretation
        - Multi-algorithm perspective: Diverse algorithmic views on healthcare workflow factors
        - Clinical interpretability: Feature importance with healthcare context and meaning
        - Bottleneck identification: Key factors affecting medication preparation efficiency
        
        Base Model Importance Extraction:
        - Random Forest: Feature importance scores from tree-based ensemble splits
        - XGBoost: Gradient boosting importance from tree construction patterns
        - Ridge: Linear coefficients indicating direct feature impact on TAT
        - Feature name mapping: Clinical variable identification for stakeholder communication
        - Algorithmic diversity: Complementary perspectives on healthcare workflow drivers
        
        Returns:
            Dict[str, Any]: Comprehensive feature importance results from interpretable
            base models with clinical context supporting pharmacy workflow optimization.
        
        Raises:
            ValueError: Unfitted ensemble preventing unsafe importance extraction
        
        Example:
            For clinical feature importance analysis:
            ```python
            # Extract multi-algorithm importance
            importance = ensemble.get_base_model_importance()
            
            # Analyze Random Forest patterns
            rf_importance = importance['rf']['values']
            top_features = np.argsort(rf_importance)[-5:]
            ```
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to extract base model importance")
        
        importance_results = {}
        
        # Extract feature importance from each interpretable base model
        for name, _ in self.base_models:
            # Access fitted base model from stacking ensemble
            base_model = getattr(self.model, f'named_estimators_')[name]
            
            if hasattr(base_model, 'feature_importances_'):
                # Tree-based models (Random Forest, XGBoost) with feature importance
                importance_results[name] = {
                    'type': 'feature_importance',                  # Importance type
                    'values': base_model.feature_importances_,     # Importance scores
                    'feature_names': self.model.feature_names_in_, # Clinical variables
                    'algorithm': 'tree_based',                     # Method category
                    'clinical_interpretation': 'Gini importance from tree splits'
                }
            elif hasattr(base_model, 'coef_'):
                # Linear models (Ridge) with coefficient interpretation
                importance_results[name] = {
                    'type': 'coefficients',                        # Coefficient type
                    'values': base_model.coef_,                    # Linear weights
                    'feature_names': self.model.feature_names_in_, # Clinical variables
                    'algorithm': 'linear',                         # Method category
                    'clinical_interpretation': 'Direct linear impact on TAT'
                }
        
        logger.info(f"Feature importance extracted from {len(importance_results)} base models")
        return importance_results
    
    def get_meta_learner_coefficients(self) -> pd.DataFrame:
        """
        Extract Meta-Learner Coefficients Showing Base Model Contributions
        
        Provides comprehensive analysis of base model contributions to ensemble
        predictions supporting healthcare pharmacy workflow optimization and
        clinical interpretability. Reveals algorithmic weighting patterns enabling
        understanding of ensemble decision-making and model contribution hierarchy.
        
        Healthcare Meta-Learning Analysis:
        - Base model contribution weights: Linear coefficients showing algorithmic influence
        - Contribution ranking: Relative importance of diverse base learners in ensemble
        - Clinical interpretability: Understanding which algorithms drive TAT predictions
        - Ensemble transparency: Insight into meta-learner combination strategies
        - Production monitoring: Base model contribution tracking for ensemble stability
        
        Meta-Learner Coefficient Analysis:
        - Linear combination weights: Ridge regression coefficients for base model contributions
        - Absolute importance ranking: Magnitude-based contribution assessment
        - Algorithmic contribution hierarchy: Understanding ensemble decision patterns
        - Clinical audit support: Transparent ensemble behavior for healthcare validation
        - Production stability insights: Base model weighting for deployment monitoring
        
        Returns:
            pd.DataFrame: Comprehensive meta-learner coefficient analysis with base
            model contributions, rankings, and clinical interpretability insights.
        
        Raises:
            ValueError: Unfitted ensemble preventing unsafe coefficient extraction
            AttributeError: Meta-learner without coefficient attributes
        
        Example:
            For ensemble contribution analysis:
            ```python
            # Extract meta-learner insights
            contributions = ensemble.get_meta_learner_coefficients()
            
            # Analyze algorithmic contributions
            print("Base Model Contributions:")
            print(contributions[['base_model', 'meta_coefficient', 'contribution_rank']])
            ```
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to extract meta-learner coefficients")
        
        if hasattr(self.model.final_estimator_, 'coef_'):
            # Extract base model names for coefficient mapping
            base_model_names = [name for name, _ in self.base_models]
            
            # Create comprehensive coefficient analysis DataFrame
            coef_df = pd.DataFrame({
                'base_model': base_model_names,                                      # Algorithm names
                'meta_coefficient': self.model.final_estimator_.coef_,              # Linear weights
                'abs_coefficient': np.abs(self.model.final_estimator_.coef_),       # Magnitude
                'contribution_rank': np.argsort(-np.abs(self.model.final_estimator_.coef_)) + 1,  # Ranking
                'clinical_interpretation': [
                    'Linear interpretability for clinical coefficients',
                    'Non-linear pattern capture and feature interactions', 
                    'Gradient boosting for complex healthcare relationships'
                ]
            }).sort_values('abs_coefficient', ascending=False)
            
            logger.info(f"Meta-learner coefficients extracted: {len(coef_df)} base models analyzed")
            return coef_df
        else:
            raise AttributeError("Meta-learner does not have coefficient attributes")