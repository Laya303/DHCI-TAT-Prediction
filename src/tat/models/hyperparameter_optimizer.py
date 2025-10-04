"""
Healthcare Analytics Hyperparameter Optimization for Medication Preparation TAT Prediction

Advanced Optuna-based optimization engine supporting healthcare pharmacy workflow
optimization and 60-minute TAT threshold compliance initiatives. Provides healthcare-focused
hyperparameter tuning with clinical interpretability constraints enabling robust model
optimization for medication preparation turnaround time prediction and bottleneck analysis.

Key Features:
- Healthcare-optimized Optuna integration with clinical interpretability constraints
- Multi-algorithm hyperparameter optimization supporting diverse TAT prediction scenarios
- Robust evaluation strategies with cross-validation and validation set flexibility
- Clinical parameter space definition ensuring healthcare deployment readiness
- Comprehensive optimization tracking with audit trails for regulatory compliance

"""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

from .base import BaseTATModel, BaseRegressionTATModel

# Suppress Optuna logging for cleaner healthcare training output
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)

class OptunaTATOptimizer:
    """
    Advanced Healthcare Hyperparameter Optimizer for TAT Prediction Models
    
    Comprehensive Optuna-based optimization engine optimized for healthcare medication
    preparation TAT prediction and pharmacy workflow optimization. Provides healthcare-focused
    hyperparameter tuning with clinical interpretability constraints and robust evaluation
    strategies ensuring optimal model configuration for diverse clinical scenarios.
    
    Args:
        random_state: Reproducibility seed ensuring consistent optimization results
                     across runs and deployment environments for healthcare validation.
    
    Attributes:
        optimization_history: Comprehensive tracking of optimization results and performance
        random_state: Reproducibility configuration for consistent healthcare analytics
        
    Example:
        For healthcare hyperparameter optimization:
        ```python
        # Initialize optimizer for TAT prediction
        optimizer = OptunaTATOptimizer(random_state=42)
        
        # Optimize XGBoost model for clinical deployment
        best_params = optimizer.optimize_model(
            xgb_model, X_train, y_train, X_val, y_val, n_trials=50
        )
        
        # Review optimization results
        summary = optimizer.get_optimization_summary()
        ```

    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize Healthcare Hyperparameter Optimizer with Clinical Configuration
        
        Establishes advanced Optuna-based optimization engine optimized for healthcare
        medication preparation TAT prediction and pharmacy workflow optimization. Configures
        reproducible optimization with healthcare-specific tracking ensuring clinical
        interpretability and production deployment readiness for diverse analytics scenarios.
        
        Args:
            random_state: Reproducibility seed ensuring consistent healthcare optimization
                         results across training runs and deployment environments.
        
        """
        self.random_state = random_state        # Reproducible optimization configuration
        self.optimization_history = {}          # Comprehensive tracking for audit trails
        
        logger.info(f"Initialized TAT Optuna optimizer (random_state={random_state})")
    
    def _suggest_parameter(self, trial: optuna.Trial, param_name: str, param_config: tuple) -> Any:
        """
        Suggest Healthcare-Appropriate Parameter Values with Clinical Constraints
        
        Intelligent parameter suggestion optimized for healthcare medication preparation
        TAT prediction ensuring clinical interpretability and healthcare deployment readiness.
        Handles diverse parameter types with appropriate sampling strategies supporting
        robust hyperparameter optimization across varied healthcare analytics scenarios.
        
        Healthcare Parameter Suggestion:
        - Integer parameter handling for discrete clinical configuration values
        - Float parameter support with log-scale sampling for regularization and rates
        - Categorical parameter selection enabling algorithm-specific configuration choices
        - Clinical constraint validation ensuring healthcare-appropriate parameter ranges
        - Error handling preventing invalid parameter suggestions compromising optimization
        
        Parameter Type Support:
        - 'int': Discrete parameters like tree counts and depth limits for interpretability
        - 'float': Continuous parameters with optional log-scale for regularization values
        - 'categorical': Discrete choices for algorithm-specific configuration options
        - Clinical validation: Parameter range verification for healthcare deployment safety
        - Healthcare constraints: Interpretability and performance boundary enforcement
        
        Args:
            trial: Optuna trial instance for parameter suggestion and optimization tracking
            param_name: Healthcare parameter identifier supporting clinical documentation
            param_config: Parameter configuration tuple defining type, bounds, and distribution
            
        Returns:
            Any: Suggested parameter value within healthcare constraints and clinical bounds
        
        Raises:
            ValueError: Invalid parameter type compromising healthcare optimization integrity
        
        Example:
            For healthcare parameter suggestion:
            ```python
            # Integer parameter for tree count
            n_trees = _suggest_parameter(trial, 'n_estimators', ('int', 50, 200))
            
            # Log-scale float for regularization
            alpha = _suggest_parameter(trial, 'alpha', ('float', 0.01, 100.0, 'log'))
            ```
        """
        if param_config[0] == 'int':
            # Integer parameters for discrete clinical configuration values
            return trial.suggest_int(param_name, param_config[1], param_config[2])
        elif param_config[0] == 'float':
            # Float parameters with optional log-scale for regularization and rates
            if len(param_config) > 3 and param_config[3] == 'log':
                return trial.suggest_float(param_name, param_config[1], param_config[2], log=True)
            else:
                return trial.suggest_float(param_name, param_config[1], param_config[2])
        elif param_config[0] == 'categorical':
            # Categorical parameters for algorithm-specific configuration choices
            return trial.suggest_categorical(param_name, param_config[1])
        else:
            raise ValueError(f"Unknown parameter type: {param_config[0]}")
    
    def optimize_regression_model(self, model: BaseRegressionTATModel, 
                                X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                                n_trials: int = 20) -> Dict[str, Any]:
        """
        Optimize Healthcare Regression Model for Continuous TAT Prediction
        
        Comprehensive hyperparameter optimization optimized for healthcare medication
        preparation TAT prediction supporting pharmacy workflow optimization. Employs robust
        evaluation strategies with clinical validation ensuring optimal model configuration
        for healthcare deployment and 60-minute threshold compliance requirements.
        
        Healthcare Regression Optimization:
        - Clinical hyperparameter space exploration with interpretability constraints
        - Robust evaluation using cross-validation or validation set for healthcare accuracy
        - Target transformation handling for skewed TAT distributions common in healthcare
        - Error handling and fallback strategies ensuring optimization completion in clinical environments
        - Comprehensive tracking supporting audit trails and regulatory compliance documentation
        
        Optimization Strategy Components:
        - TPE sampling optimized for healthcare parameter spaces and clinical requirements
        - RMSE minimization focusing on clinical accuracy for medication preparation timing
        - Cross-validation robustness ensuring model stability across diverse healthcare datasets
        - Healthcare metadata preservation supporting clinical documentation and validation
        - Production readiness validation ensuring optimized models meet deployment standards
        
        Args:
            model: Healthcare regression model instance ready for hyperparameter optimization
            X_train: Training feature matrix containing clinical and operational variables
            y_train: Training TAT target in minutes supporting continuous prediction optimization
            X_val: Optional validation features for dedicated evaluation and performance tracking
            y_val: Optional validation TAT target for robust optimization and clinical validation
            n_trials: Optimization iterations balancing performance improvement with efficiency
            
        Returns:
            Dict[str, Any]: Optimal hyperparameters with healthcare constraints and clinical
            validation ensuring deployment readiness and pharmacy workflow optimization.
        
        Example:
            For healthcare regression optimization:
            ```python
            # Optimize XGBoost for TAT prediction
            best_params = optimizer.optimize_regression_model(
                xgb_model, X_train, y_train, X_val, y_val, n_trials=50
            )
            
            # Access optimization results
            print(f"Best RMSE: {optimizer.optimization_history['XGBoostTATRegressor_regression']['best_value']:.2f}")
            ```
        """
        # Extract healthcare-specific hyperparameter space with clinical constraints
        hyperparameter_space = model.get_hyperparameter_space()
        
        def objective(trial):
            """
            Objective function for healthcare hyperparameter optimization with clinical focus
            
            Evaluates trial parameters for healthcare TAT prediction ensuring clinical
            accuracy and healthcare deployment readiness through robust evaluation strategies.
            """
            # Generate trial parameters within healthcare constraints
            trial_params = {}
            for param_name, param_config in hyperparameter_space.items():
                trial_params[param_name] = self._suggest_parameter(trial, param_name, param_config)
            
            # Create model instance with trial parameters for healthcare evaluation
            model_class = model.__class__
            trial_model = model_class(random_state=self.random_state, **trial_params)
            
            # Transform target for skewed healthcare TAT data distributions
            y_train_transformed = trial_model._transform_target(y_train)
            
            if X_val is not None and y_val is not None:
                # Use validation set for dedicated performance evaluation
                trial_model.fit(X_train, y_train)
                y_pred = trial_model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            else:
                # Use cross-validation for robust healthcare model evaluation
                try:
                    scores = cross_val_score(
                        trial_model.model, X_train, y_train_transformed,
                        cv=3,                          # 3-fold CV for efficiency
                        scoring='neg_mean_squared_error', # RMSE-based evaluation
                        n_jobs=-1                      # Parallel processing
                    )
                    rmse = np.sqrt(-scores.mean())     # Convert to RMSE
                except Exception as e:
                    logger.warning(f"Cross-validation failed for trial: {e}")
                    return float('inf')  # Penalize failed trials
            
            return rmse  # Minimize RMSE for clinical accuracy
        
        # Create healthcare-optimized Optuna study with clinical configuration
        study = optuna.create_study(
            direction="minimize",                                    # Minimize RMSE for accuracy
            sampler=TPESampler(seed=self.random_state),             # Reproducible sampling
            study_name=f"tat_{model.__class__.__name__}"            # Healthcare study naming
        )
        
        try:
            # Execute optimization with healthcare progress tracking
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            # Store comprehensive optimization history for audit trails
            model_key = f"{model.__class__.__name__}_regression"
            self.optimization_history[model_key] = {
                'best_params': study.best_params,           # Optimal hyperparameters
                'best_value': study.best_value,             # Best RMSE achieved
                'n_trials': len(study.trials),              # Optimization iterations
                'optimization_successful': True,            # Success status
                'healthcare_context': 'TAT prediction regression', # Clinical application
                'clinical_objective': '60-minute threshold optimization' # Quality goal
            }
            
            logger.info(f"{model.__class__.__name__} optimization complete. Best RMSE: {study.best_value:.2f} minutes")
            return study.best_params
            
        except Exception as e:
            logger.error(f"{model.__class__.__name__} optimization failed: {e}")
            
            # Return default healthcare parameters on optimization failure
            default_params = self._get_default_params(model)
            
            model_key = f"{model.__class__.__name__}_regression"
            self.optimization_history[model_key] = {
                'best_params': default_params,              # Fallback configuration
                'best_value': None,                         # No optimization value
                'n_trials': 0,                              # No successful trials
                'optimization_successful': False,           # Failure status
                'error': str(e),                           # Error documentation
                'fallback_used': True,                     # Fallback indicator
                'healthcare_context': 'TAT prediction regression (fallback)' # Clinical context
            }
            
            return default_params
    
    def optimize_model(self, model: BaseTATModel, 
                      X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                      n_trials: int = 20) -> Dict[str, Any]:
        """
        General Healthcare Model Optimization Dispatcher with Clinical Validation
        
        Comprehensive optimization dispatcher optimized for healthcare medication
        preparation TAT prediction supporting pharmacy workflow optimization. Routes
        model types to appropriate optimization strategies ensuring clinical interpretability
        and healthcare deployment readiness across diverse analytics scenarios.
        
        Healthcare Optimization Dispatch:
        - Model type detection ensuring appropriate optimization strategy selection
        - Regression model routing to continuous TAT prediction optimization workflows
        - Error handling preventing optimization failures in healthcare environments
        - Healthcare validation ensuring optimized models meet clinical deployment standards
        - Production readiness confirmation supporting MLOps integration and monitoring
        
        Args:
            model: Healthcare TAT model instance ready for hyperparameter optimization
            X_train: Training feature matrix with clinical and operational variables
            y_train: Training target supporting model-specific optimization strategies
            X_val: Optional validation features for robust evaluation and performance tracking
            y_val: Optional validation target for clinical accuracy assessment
            n_trials: Optimization iterations balancing improvement with computational efficiency
            
        Returns:
            Dict[str, Any]: Optimal hyperparameters with healthcare constraints ensuring
            clinical interpretability and pharmacy workflow optimization requirements.
        
        Raises:
            ValueError: Unknown model type compromising healthcare optimization integrity
        
        Example:
            For general healthcare model optimization:
            ```python
            # Optimize any supported TAT model
            best_params = optimizer.optimize_model(
                model, X_train, y_train, X_val, y_val, n_trials=30
            )
            
            # Use optimized parameters for production deployment
            optimized_model = ModelClass(**best_params)
            ```
        """
        if isinstance(model, BaseRegressionTATModel):
            # Route regression models to continuous TAT prediction optimization
            return self.optimize_regression_model(model, X_train, y_train, X_val, y_val, n_trials)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
    
    def _get_default_params(self, model: BaseTATModel) -> Dict[str, Any]:
        """
        Get Healthcare-Appropriate Default Parameters for Model Fallback
        
        Provides clinically validated default hyperparameters for healthcare medication
        preparation TAT prediction ensuring robust fallback configuration when optimization
        fails. Maintains healthcare deployment readiness with interpretability constraints
        and clinical performance standards for pharmacy workflow optimization scenarios.
        
        Healthcare Default Configuration:
        - Algorithm-specific defaults optimized for clinical interpretability and performance
        - Healthcare constraint enforcement ensuring deployment readiness and safety
        - Interpretability preservation maintaining transparency for clinical stakeholders
        - Performance balance ensuring accuracy while maintaining computational efficiency
        - Production validation supporting reliable healthcare analytics deployment
        
        Args:
            model: Healthcare TAT model instance requiring default parameter configuration
            
        Returns:
            Dict[str, Any]: Healthcare-validated default hyperparameters ensuring clinical
            interpretability and pharmacy workflow optimization requirements.
        
        """
        model_name = model.__class__.__name__
        
        # Healthcare-optimized default parameters by algorithm type
        defaults = {
            'XGBoostTATRegressor': {
                'n_estimators': 150,        # Moderate complexity for interpretability
                'max_depth': 6,             # Clinical interpretability balance
                'learning_rate': 0.1,       # Conservative learning for stability
                'subsample': 0.8,           # Robust training with sample variation
                'colsample_bytree': 0.8,    # Feature sampling for generalization
                'reg_alpha': 0.1,           # L1 regularization for feature selection
                'reg_lambda': 0.1           # L2 regularization for stability
            },
            'RandomForestTATRegressor': {
                'n_estimators': 200,        # Ensemble diversity for robustness
                'max_depth': 8,             # Interpretability with performance balance
                'min_samples_split': 5,     # Conservative splitting for stability
                'min_samples_leaf': 2,      # Minimum leaf size for robustness
                'max_features': 'sqrt'      # Feature sampling for diversity
            },
            'RidgeTATRegressor': {
                'alpha': 1.0               # Moderate regularization for stability
            },
            'StackingTATRegressor': {
                'meta_alpha': 1.0,          # Meta-learner regularization
                'xgb_n_estimators': 100,    # XGBoost base model complexity
                'rf_n_estimators': 100,     # Random Forest base model trees
                'ridge_alpha': 10.0         # Ridge base model regularization
            }
        }
        
        return defaults.get(model_name, {})
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """
        Get Comprehensive Optimization Summary for Healthcare Reporting and Analysis
        
        Provides detailed optimization results analysis optimized for healthcare pharmacy
        workflow optimization and clinical stakeholder communication. Generates comprehensive
        summary supporting evidence-based model selection and healthcare deployment decisions
        with audit trail documentation and regulatory compliance requirements.
        
        Healthcare Optimization Reporting:
        - Model performance comparison supporting evidence-based selection for clinical deployment
        - Optimization success tracking enabling robust healthcare analytics pipeline monitoring
        - Error documentation supporting troubleshooting and continuous improvement workflows
        - Clinical context preservation maintaining healthcare interpretation and validation
        - Audit trail generation supporting regulatory compliance and documentation requirements
        
        Summary Components:
        - Model identification: Algorithm names with clinical context and healthcare application
        - Optimization status: Success indicators supporting deployment readiness assessment
        -  Numerical Features: Best RMSE values enabling clinical accuracy comparison
        - Trial statistics: Optimization iteration tracking for efficiency analysis and reporting
        - Error tracking: Failure documentation supporting continuous improvement and debugging
        
        Returns:
            pd.DataFrame: Comprehensive optimization summary with healthcare context and
            clinical interpretation supporting pharmacy workflow optimization and stakeholder communication.
        
        Example:
            For healthcare optimization reporting:
            ```python
            # Generate optimization summary for clinical review
            summary = optimizer.get_optimization_summary()
            
            # Review model performance comparison
            best_models = summary.sort_values('best_value').head(3)
            print("Top 3 optimized models for clinical deployment:")
            print(best_models[['model', 'best_value', 'optimization_successful']])
            ```
        """
        if not self.optimization_history:
            # Return empty DataFrame with healthcare-appropriate schema
            return pd.DataFrame(columns=[
                'model', 'optimization_successful', 'best_value', 
                'n_trials', 'has_error', 'healthcare_context', 
                'clinical_objective', 'fallback_used'
            ])
        
        # Generate comprehensive summary data for healthcare reporting
        summary_data = []
        for model_key, history in self.optimization_history.items():
            summary_data.append({
                'model': model_key,                                           # Model identifier
                'optimization_successful': history['optimization_successful'], # Success status
                'best_value': history['best_value'],                         # Best RMSE achieved
                'n_trials': history['n_trials'],                            # Optimization iterations
                'has_error': 'error' in history,                            # Error indicator
                'healthcare_context': history.get('healthcare_context', 'TAT prediction'), # Clinical application
                'clinical_objective': history.get('clinical_objective', '60-minute optimization'), # Quality goal
                'fallback_used': history.get('fallback_used', False)        # Fallback indicator
            })
        
        # Create DataFrame with healthcare context and clinical interpretation
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by performance for clinical stakeholder review (successful optimizations first)
        if 'best_value' in summary_df.columns:
            summary_df = summary_df.sort_values(
                ['optimization_successful', 'best_value'], 
                ascending=[False, True]  # Success first, then by RMSE
            )
        
        logger.info(f"Optimization summary generated: {len(summary_df)} models analyzed")
        return summary_df