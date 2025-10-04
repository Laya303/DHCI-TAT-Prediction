"""
Model factory for TAT prediction model creation and training.

Provides centralized model instantiation, hyperparameter optimization,
and training orchestration for various ML algorithms.
- Healthcare metadata management enabling audit trails and regulatory compliance
- Model versioning and artifact management supporting production healthcare deployment
- Clinical validation workflows ensuring model safety and accuracy for patient care

Note:
    Essential for healthcare pharmacy workflow optimization providing comprehensive
    model development infrastructure supporting medication preparation efficiency and
    clinical operations excellence through advanced healthcare analytics orchestration.
"""
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from pathlib import Path
import joblib

from .base import BaseTATModel, BaseRegressionTATModel
from .xgboost_model import XGBoostTATRegressor
from .random_forest_model import RandomForestTATRegressor
from .linear_model import RidgeTATRegressor
from .ensemble_model import StackingTATRegressor
from .hyperparameter_optimizer import OptunaTATOptimizer

logger = logging.getLogger(__name__)

class TATModelFactory:
    """
    Healthcare Model Factory for TAT Prediction System
    
    Comprehensive model instantiation factory optimized for healthcare medication
    preparation TAT prediction and pharmacy workflow optimization. Provides centralized
    model creation with healthcare-specific configurations ensuring clinical
    interpretability and production deployment readiness for diverse analytics scenarios.
    
    Example:
        For healthcare model instantiation:
        ```python
        # Create XGBoost model for TAT prediction
        xgb_model = TATModelFactory.create_regression_model(
            'xgboost', 
            n_estimators=100,
            max_depth=6
        )
        
        # Create ensemble model for robust predictions
        ensemble = TATModelFactory.create_regression_model('stacking')
        ```
    
 
    """
    
    # Healthcare-optimized model registry for TAT prediction
    REGRESSION_MODELS = {
        'ridge': RidgeTATRegressor,              # Linear interpretability
        'xgboost': XGBoostTATRegressor,          # Gradient boosting complexity
        'random_forest': RandomForestTATRegressor, # Ensemble tree-based learning
        'stacking': StackingTATRegressor         # Multi-algorithm ensemble
    }
    
    @classmethod
    def create_regression_model(cls, model_type: str, **kwargs) -> BaseRegressionTATModel:
        """
        Create Healthcare Regression Model for Continuous TAT Prediction
        
        Instantiates healthcare-optimized regression models for healthcare medication
        preparation TAT prediction supporting pharmacy workflow optimization. Provides
        standardized model creation with clinical parameter validation and healthcare-specific
        configuration ensuring production deployment readiness and interpretability.
        
        Healthcare Model Creation:
        - Model type validation ensuring healthcare-appropriate algorithm selection
        - Clinical parameter configuration supporting interpretability and performance requirements
        - Healthcare metadata initialization enabling audit trails and documentation
        - Production deployment preparation with MLOps integration capabilities
        - Error handling ensuring robust model instantiation in healthcare environments
        
        Supported Healthcare Models:
        - 'ridge': Linear regression with regularization for clinical interpretability
        - 'xgboost': Gradient boosting for complex healthcare workflow pattern modeling
        - 'random_forest': Tree ensemble for robust prediction with feature interactions
        - 'stacking': Multi-algorithm ensemble for superior prediction accuracy and robustness
        
        Args:
            model_type: Healthcare model algorithm selection ('ridge', 'xgboost', 
                       'random_forest', 'stacking') optimized for clinical scenarios.
            **kwargs: Model-specific healthcare parameters supporting clinical optimization
                     and interpretability requirements for pharmacy workflow analysis.
            
        Returns:
            BaseRegressionTATModel: Configured healthcare regression model ready for
            TAT prediction training with clinical validation and production deployment.
        
        Raises:
            ValueError: Invalid model type compromising healthcare analytics integrity
        
        Example:
            For clinical TAT prediction model creation:
            ```python
            # Create interpretable Ridge model for clinical review
            ridge_model = TATModelFactory.create_regression_model(
                'ridge', 
                alpha=10.0,
                random_state=42
            )
            
            # Create ensemble model for production deployment
            ensemble_model = TATModelFactory.create_regression_model('stacking')
            ```
        
        """
        if model_type not in cls.REGRESSION_MODELS:
            available = ', '.join(cls.REGRESSION_MODELS.keys())
            raise ValueError(f"Unknown regression model: {model_type}. Available: {available}")
        
        # Instantiate healthcare-optimized model class with clinical configuration
        model_class = cls.REGRESSION_MODELS[model_type]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> Dict[str, List[str]]:
        """
        Get Available Healthcare Model Types for TAT Prediction
        
        Provides comprehensive listing of available healthcare analytics models
        supporting healthcare pharmacy workflow optimization and clinical
        decision-making. Essential for healthcare stakeholder communication and
        model selection workflows enabling evidence-based algorithm choice.
        
        Healthcare Model Categories:
        - Regression models: Continuous TAT prediction supporting precise workflow timing
        - Clinical optimization: Models optimized for healthcare interpretability requirements
        - Production deployment: Algorithms suitable for real-time healthcare analytics
        - Stakeholder communication: Model types with clear clinical interpretation capabilities
        
        Returns:
            Dict[str, List[str]]: Comprehensive model registry with healthcare categories
            supporting clinical algorithm selection and pharmacy workflow optimization.
        
        Example:
            For healthcare model selection workflow:
            ```python
            # Get available models for clinical team review
            available = TATModelFactory.get_available_models()
            
            print("Available TAT prediction models:")
            for category, models in available.items():
                print(f"  {category}: {', '.join(models)}")
            ```

        """
        return {
            'regression': list(cls.REGRESSION_MODELS.keys())  # Continuous TAT prediction models
        }


class TATTrainingOrchestrator:
    """
    Comprehensive Healthcare Training Orchestrator for TAT Prediction Models
    
    Advanced training pipeline management optimized for healthcare medication
    preparation TAT prediction and pharmacy workflow optimization. Orchestrates complete
    model development lifecycle including data splitting, hyperparameter optimization,
    training, evaluation, and feature importance analysis for clinical decision support.
    
    Args:
        scaling_strategy: Training strategy selection ('tree', 'linear', 'mixed') optimizing
                         for healthcare interpretability and performance requirements.
        random_state: Reproducibility seed ensuring consistent healthcare analytics results
                     across training runs and deployment environments for validation.
    
    Attributes:
        trained_models: Registry of trained healthcare models with clinical metadata
        training_results: Comprehensive training outcomes with  Numerical Features
        optimizer: Hyperparameter optimization engine with healthcare constraints
        
    Example:
        For comprehensive healthcare model training:
        ```python
        # Initialize training orchestrator for pharmacy optimization
        orchestrator = TATTrainingOrchestrator(
            scaling_strategy='mixed',
            random_state=42
        )
        
        # Train complete model suite with clinical validation
        results = orchestrator.train_all_models(X, y_reg, y_clf)
        
        # Get best model for production deployment
        best_model = orchestrator.get_best_model()
        ```

    """
    
    def __init__(self, scaling_strategy: str = "mixed", random_state: int = 42):
        """
        Initialize Healthcare Training Orchestrator with Clinical Configuration
        
        Establishes comprehensive training pipeline optimized for healthcare
        medication preparation TAT prediction and pharmacy workflow optimization.
        Configures strategy-based model selection and healthcare-specific validation
        ensuring clinical interpretability and production deployment readiness.
        
        Healthcare Training Initialization:
        - Strategy-based model selection ensuring clinical interpretability and performance balance
        - Reproducibility configuration supporting consistent healthcare analytics validation
        - Hyperparameter optimizer initialization with healthcare constraints and clinical focus
        - Training registry setup enabling comprehensive model management and audit trails
        - Production preparation supporting MLOps integration and automated deployment workflows
        
        Args:
            scaling_strategy: Training approach selection optimizing for healthcare scenarios:
                            - 'tree': Tree-based models for complex pattern recognition
                            - 'linear': Linear models for clinical interpretability
                            - 'mixed': Comprehensive approach with ensemble capabilities
            random_state: Reproducibility seed ensuring consistent training across environments
        
        """
        self.scaling_strategy = scaling_strategy  # Strategy for healthcare model selection
        self.random_state = random_state          # Reproducible analytics configuration
        self.trained_models = {}                  # Healthcare model registry
        self.training_results = {}                # Comprehensive training outcomes
        
        # Initialize hyperparameter optimizer with healthcare constraints
        self.optimizer = OptunaTATOptimizer(random_state=random_state)
        
        logger.info(f"Initialized TAT training orchestrator ({scaling_strategy} strategy)")
    
    def create_train_test_splits(self, X: pd.DataFrame, y_reg: pd.Series, y_clf: pd.Series,
                           test_size: float = 0.2) -> Dict[str, Any]:
        """
        Create Healthcare-Optimized Train/Test Splits with Clinical Validation
        
        Generates stratified data splits optimized for healthcare medication preparation
        TAT analysis ensuring representative healthcare dataset validation. Provides robust
        splitting with clinical class balance validation supporting accurate model training
        and evaluation for pharmacy workflow optimization and 60-minute threshold compliance.
        
        Healthcare Data Splitting:
        - Stratified splitting ensuring representative TAT threshold class distribution
        - Clinical validation checking minimum class sizes for robust healthcare analytics
        - Fallback splitting handling edge cases in healthcare data distribution patterns
        - Class balance verification ensuring both threshold categories in train/test sets
        - Error handling supporting robust operation with diverse healthcare dataset characteristics
        
        Args:
            X: Healthcare feature matrix containing clinical, operational, and temporal variables
            y_reg: Continuous TAT target in minutes for regression model training and evaluation
            y_clf: Binary TAT threshold target (>60 minutes) for stratified splitting validation
            test_size: Proportion of data reserved for testing (default: 0.2 for 80/20 split)
            
        Returns:
            Dict[str, Any]: Comprehensive data splits with healthcare validation including
            training and testing sets for both regression and classification targets.
        
        Raises:
            ValueError: Insufficient data diversity compromising healthcare model validation
        
        Example:
            For healthcare data splitting workflow:
            ```python
            # Create validated healthcare data splits
            splits = orchestrator.create_train_test_splits(X, y_reg, y_clf)
            
            # Verify clinical representation
            print(f"Training samples: {len(splits['X_train']):,}")
            print(f"Test samples: {len(splits['X_test']):,}")
            ```
        """
        from sklearn.model_selection import train_test_split
        
        # Analyze clinical class distribution for healthcare validation
        class_counts = y_clf.value_counts()
        logger.info(f"Overall class distribution: {class_counts.to_dict()}")
        
        # Validate sufficient class diversity for healthcare analytics
        if len(class_counts) < 2:
            raise ValueError(f"Cannot create stratified split with only one class: {class_counts.to_dict()}")
        
        # Ensure minimum samples per class for stable healthcare model training
        min_class_size = min(class_counts.values)
        if min_class_size < 10:
            logger.warning(f"Very small minority class ({min_class_size} samples) - consider collecting more data")
        
        try:
            # Perform stratified splitting ensuring clinical representation
            X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
                X, y_reg, y_clf,
                test_size=test_size,           # Test proportion for validation
                random_state=self.random_state, # Reproducible splitting
                stratify=y_clf,                # Maintain class balance
                shuffle=True                   # Randomize before splitting
            )
            
            # Verify clinical representation in both splits
            train_counts = pd.Series(y_clf_train).value_counts()
            test_counts = pd.Series(y_clf_test).value_counts()
            
            logger.info(f"Training set class distribution: {train_counts.to_dict()}")
            logger.info(f"Test set class distribution: {test_counts.to_dict()}")
            
            # Validate successful stratification for healthcare analytics
            if len(train_counts) < 2 or len(test_counts) < 2:
                raise ValueError("Stratified split failed to maintain both classes in train/test sets")
            
            return {
                'X_train': X_train, 'X_test': X_test,                    # Feature splits
                'y_reg_train': y_reg_train, 'y_reg_test': y_reg_test,    # Regression targets
                'y_clf_train': y_clf_train, 'y_clf_test': y_clf_test      # Classification targets
            }
            
        except ValueError as e:
            if "stratify" in str(e).lower():
                logger.error(f"Stratified split failed: {e}")
                logger.info("Falling back to random split without stratification")
                
                # Fallback: random split for problematic healthcare datasets
                X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
                    X, y_reg, y_clf,
                    test_size=test_size,           # Maintain test proportion
                    random_state=self.random_state, # Reproducible results
                    shuffle=True                   # Randomized splitting
                )
                
                return {
                    'X_train': X_train, 'X_test': X_test,
                    'y_reg_train': y_reg_train, 'y_reg_test': y_reg_test,
                    'y_clf_train': y_clf_train, 'y_clf_test': y_clf_test
                }
            else:
                raise e
    
    def train_model_with_optimization(self, model_type: str,
                                    splits: Dict[str, Any], n_trials: int = 20) -> BaseTATModel:
        """
        Train Healthcare Model with Comprehensive Optimization and Clinical Analysis
        
        Comprehensive model training workflow optimized for healthcare medication
        preparation TAT prediction including hyperparameter optimization, feature importance
        analysis, and clinical evaluation. Provides complete model development supporting
        pharmacy workflow optimization and healthcare stakeholder decision-making requirements.
        
        Args:
            model_type: Healthcare algorithm selection supporting clinical interpretability
                       and performance requirements for medication preparation analysis.
            splits: Validated healthcare data splits ensuring representative training/testing
                   with clinical class balance and statistical validity for model development.
            n_trials: Hyperparameter optimization iterations balancing performance improvement
                     with computational efficiency for healthcare production deployment.
            
        Returns:
            BaseTATModel: Fully trained and optimized healthcare model with clinical
            validation and feature importance analysis ready for production deployment.
        
        Example:
            For comprehensive healthcare model training:
            ```python
            # Train optimized XGBoost model with clinical analysis
            xgb_model = orchestrator.train_model_with_optimization(
                'xgboost', 
                splits, 
                n_trials=50
            )
            
            # Access clinical insights
            results = orchestrator.training_results['xgboost']
            top_features = results['metrics']['top_clinical_features']
            ```
    
        """
        from ..analysis.feature_importance import FeatureImportanceAnalyzer
        
        # Create healthcare model instance with default configuration
        model = TATModelFactory.create_regression_model(model_type)
        y_train, y_test = splits['y_reg_train'], splits['y_reg_test']
        
        # Optimize hyperparameters with healthcare constraints and clinical focus
        best_params = self.optimizer.optimize_model(
            model, splits['X_train'], y_train, splits['X_test'], y_test, n_trials
        )
        
        # Create optimized model instance with best healthcare parameters
        optimized_model = TATModelFactory.create_regression_model(model_type, **best_params)
        
        # Train final model with validation data integration for robustness
        optimized_model.fit(
            splits['X_train'], y_train,
            validation_data=(splits['X_test'], y_test)  # Validation for training monitoring
        )
        
        # Evaluate clinical performance using healthcare-specific metrics
        predictions = optimized_model.predict(splits['X_test'])
        metrics = optimized_model.evaluate_healthcare_metrics(y_test, predictions)

        # Generate comprehensive feature importance analysis for clinical insights
        try:
            # Initialize feature importance analyzer with healthcare context
            logger.info(f"Generating feature importance analysis for {model_type} model")
            analyzer = FeatureImportanceAnalyzer(optimized_model.model, splits['X_train'])
            importance_results = analyzer.analyze_importance(splits['X_test'])

            # Integrate clinical insights into metrics
            metrics.update({
                'feature_importance': importance_results,                      # Complete analysis
                'top_clinical_features': importance_results.get('top_features', [])[:10],  # Key factors
                'clinical_insights': importance_results.get('clinical_insights', [])       # Actionable insights
            })
            
            logger.info(f"{model_type} feature importance analysis completed successfully")
            
        except Exception as e:
            logger.warning(f"Feature importance analysis failed for {model_type}: {e}")
            # Provide fallback structure for consistent interface
            metrics.update({
                'feature_importance': {'error': str(e)},    # Error documentation
                'top_clinical_features': [],                # Empty list fallback
                'clinical_insights': []                     # Empty insights fallback
            })

        # Store comprehensive training results with healthcare metadata
        model_key = f"{model_type}"
        self.trained_models[model_key] = optimized_model
        self.training_results[model_key] = {
            'model': optimized_model,       # Trained model instance
            'metrics': metrics,             # Clinical performance evaluation
            'best_params': best_params      # Optimized hyperparameters
        }
        
        logger.info(f"Trained {model_type} (RMSE: {metrics.get('RMSE', 'N/A'):.2f})")
        return optimized_model
    
    def _train_single_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series, n_trials: int = 20) -> Dict[str, Any]:
        """
        Train Single Healthcare Model with Optimization and Analysis
        
        Internal method for training individual healthcare models with hyperparameter
        optimization and clinical evaluation. Provides standardized training workflow
        supporting healthcare pharmacy workflow optimization requirements.
        
        Args:
            model_type: Healthcare algorithm selection for TAT prediction
            X_train: Training features for healthcare model development
            y_train: Training targets for TAT prediction modeling
            X_test: Testing features for clinical validation
            y_test: Testing targets for performance evaluation
            n_trials: Hyperparameter optimization iterations
            
        Returns:
            Dict[str, Any]: Training result with model, metrics, and optimized parameters
        """
        splits = {
            'X_train': X_train, 'X_test': X_test,
            'y_reg_train': y_train, 'y_reg_test': y_test
        }
        
        model = self.train_model_with_optimization(model_type, splits, n_trials)
        
        return {
            'model': model,
            'metrics': self.training_results[model_type]['metrics'],
            'optimized_params': self.training_results[model_type]['best_params']
        }
    
    def analyze_feature_importance(self, model_type: str) -> Dict[str, float]:
        """
        Analyze Feature Importance for Clinical Insights
        
        Extracts feature importance analysis from trained healthcare models supporting
        healthcare pharmacy workflow optimization and bottleneck identification.
        Provides clinical insights for healthcare stakeholder decision-making.
        
        Args:
            model_type: Trained healthcare model for importance analysis
            
        Returns:
            Dict[str, float]: Feature importance scores for clinical interpretation
        """
        if model_type not in self.training_results:
            raise ValueError(f"Model {model_type} not found in training results")
        
        metrics = self.training_results[model_type]['metrics']
        
        if 'feature_importance' in metrics and 'importance_scores' in metrics['feature_importance']:
            return metrics['feature_importance']['importance_scores']
        else:
            # Fallback: try to get importance from the trained model directly
            model = self.training_results[model_type]['model']
            if hasattr(model.model, 'feature_importances_'):
                # For tree-based models
                feature_names = getattr(model, 'feature_names_', 
                                      [f'feature_{i}' for i in range(len(model.model.feature_importances_))])
                return dict(zip(feature_names, model.model.feature_importances_))
            elif hasattr(model.model, 'coef_'):
                # For linear models
                feature_names = getattr(model, 'feature_names_', 
                                      [f'feature_{i}' for i in range(len(model.model.coef_))])
                return dict(zip(feature_names, abs(model.model.coef_)))
            else:
                return {}
    
    def train_all_models(self, X: pd.DataFrame, y_reg: pd.Series, y_clf: pd.Series) -> Dict[str, Dict]:
        """
        Train Complete Healthcare TAT Model Suite with Clinical Validation
        
        Comprehensive training orchestration for healthcare medication preparation
        TAT prediction supporting pharmacy workflow optimization through diverse algorithmic
        approaches. Trains strategy-based model suite with clinical validation, feature
        importance analysis, and performance comparison enabling evidence-based model selection.
        
        Args:
            X: Healthcare feature matrix containing clinical, operational, and temporal
               variables supporting comprehensive medication preparation workflow analysis.
            y_reg: Continuous TAT target in minutes supporting precise workflow timing
                   prediction and pharmacy operations optimization requirements.
            y_clf: Binary TAT threshold target (>60 minutes) supporting quality compliance
                   assessment and stratified validation for robust model training workflows.
            
        Returns:
            Dict[str, Dict]: Comprehensive training results including model instances,
             Numerical Features, feature importance analysis, and clinical insights.
        
        Example:
            For complete healthcare model suite training:
            ```python
            # Train all models with mixed strategy
            results = orchestrator.train_all_models(X, y_reg, y_clf)
            
            # Review performance comparison
            for model_name, result in results.items():
                metrics = result['metrics']
                print(f"{model_name}: RMSE={metrics['RMSE']:.2f}")
            ```
        """
        logger.info(f"Training healthcare TAT models with {self.scaling_strategy} strategy")
        
        # Create validated healthcare data splits with clinical representation
        splits = self.create_train_test_splits(X, y_reg, y_clf)
        
        # Define model training strategy based on healthcare requirements
        models_to_train = []
        
        # Strategy-specific model training to avoid double-counting ensemble base learners
        if self.scaling_strategy == "tree":
            # Tree-based models for complex healthcare pattern recognition
            models_to_train.extend([
                'xgboost',        # Gradient boosting for complex relationships
                'random_forest'   # Ensemble learning with feature interactions
            ])
        elif self.scaling_strategy == "linear":
            # Linear models for clinical interpretability and transparency
            models_to_train.extend([
                'ridge'           # Regularized linear regression
            ])
        elif self.scaling_strategy == "mixed":
            # Comprehensive ensemble approach - single unified model
            models_to_train.append('stacking')  # Multi-algorithm ensemble (includes Ridge, XGB, RF internally)
        
        # Train complete model suite with healthcare optimization
        for model_type in models_to_train:
            try:
                self.train_model_with_optimization(model_type, splits)
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
        
        # Generate comprehensive performance comparison for healthcare stakeholders
        self._generate_comparison_report()
        
        logger.info("Healthcare TAT model training complete!")
        return self.training_results
    
    def _generate_comparison_report(self) -> None:
        """
        Generate Healthcare Performance Comparison Report for Clinical Stakeholders
        
        Comprehensive model performance analysis optimized for healthcare pharmacy
        workflow optimization and clinical decision-making. Provides healthcare stakeholder
        communication with performance rankings, clinical metrics, and actionable insights
        supporting evidence-based model selection for medication preparation efficiency.
        
        Healthcare Performance Analysis:
        - RMSE-based ranking supporting clinical accuracy assessment and model comparison
        - 60-minute threshold accuracy highlighting quality compliance performance
        - Clinical accuracy bands (within 30 minutes) supporting workflow optimization
        - Healthcare composite scoring providing stakeholder-friendly performance summary
        - Top-performing model identification enabling production deployment decisions

        """
        if not self.training_results:
            return
        
        # Extract regression models for healthcare performance comparison
        regression_results = {
            k: v for k, v in self.training_results.items() if 'RMSE' in v['metrics']
        }
        
        if regression_results:
            logger.info("\nHealthcare TAT Prediction Model Performance Summary:")
            
            # Sort by RMSE (primary clinical accuracy metric)
            sorted_models = sorted(
                regression_results.items(),
                key=lambda x: x[1]['metrics']['RMSE']
            )
            
            # Report top 3 models for healthcare stakeholder review
            for model_name, result in sorted_models[:3]:
                metrics = result['metrics']
                logger.info(
                    f"  {model_name}: RMSE={metrics['RMSE']:.2f}min, "
                    f"60min_acc={metrics.get('threshold_60min_accuracy', 0):.1f}%, "
                    f"within_30min={metrics.get('within_30min_pct', 0):.1f}%"
                )

    def save_all_models(self, output_dir: Path) -> None:
        """
        Save Complete Healthcare Model Suite with Clinical Artifacts for MLOps Deployment
        
        Comprehensive model persistence optimized for healthcare healthcare analytics
        deployment and MLOps integration. Saves trained models, feature importance analysis,
        and clinical metadata supporting automated deployment, monitoring, and audit trail
        requirements for pharmacy workflow optimization and regulatory compliance.
        
        Healthcare MLOps Artifact Management:
        - Trained model serialization with healthcare metadata and clinical context preservation
        - Feature importance analysis export enabling clinical review and bottleneck identification
        - CSV export for healthcare stakeholder accessibility and clinical interpretation
        - Training metadata preservation supporting audit trails and regulatory documentation
        - Directory organization enabling automated deployment and monitoring workflows
        
        Clinical Artifact Components:
        - Model binaries: Trained healthcare models ready for production TAT prediction
        - Feature importance: Clinical insights for pharmacy workflow optimization and review
        - Top features CSV: Healthcare stakeholder-friendly format for clinical interpretation
        - Training metadata: Comprehensive documentation supporting audit and compliance requirements
        - Healthcare context: Clinical application documentation for deployment and validation
        
        Args:
            output_dir: Healthcare artifact storage location supporting MLOps integration
                       and automated deployment workflows for clinical analytics systems.
        
        Example:
            For healthcare MLOps artifact management:
            ```python
            # Save complete model suite with clinical artifacts
            orchestrator.save_all_models(Path("models/healthcare_tat"))
            
            # Artifacts include:
            # - Individual model files (.joblib)
            # - Feature importance analysis
            # - Clinical top features (CSV)
            # - Training metadata
            ```

        """
        # Create healthcare artifact directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature importance directory for clinical review and analysis
        feature_importance_dir = output_dir / "feature_importance"
        feature_importance_dir.mkdir(exist_ok=True)
        
        # Save all trained healthcare models with clinical metadata
        for model_name, result in self.training_results.items():
            model = result['model']
            model_path = output_dir / f"{model_name}_model.joblib"
            model.save_model(model_path)
            
            # Save feature importance analysis for clinical bottleneck identification
            if 'feature_importance' in result['metrics']:
                importance_path = feature_importance_dir / f"{model_name}_feature_importance.joblib"
                joblib.dump(result['metrics']['feature_importance'], importance_path)
                
                # Export top features as CSV for healthcare stakeholder accessibility
                if 'top_features' in result['metrics']['feature_importance']:
                    top_features_df = pd.DataFrame(result['metrics']['feature_importance']['top_features'])
                    csv_path = feature_importance_dir / f"{model_name}_top_features.csv"
                    top_features_df.to_csv(csv_path, index=False)
        
        # Save comprehensive training metadata for healthcare audit trails
        metadata = {
            'scaling_strategy': self.scaling_strategy,             # Training strategy
            'random_state': self.random_state,                     # Reproducibility config
            'training_results_summary': {                          # Performance summary
                k: {
                    'metrics': {key: val for key, val in v['metrics'].items() if key != 'feature_importance'},
                    'has_feature_importance': 'feature_importance' in v['metrics']
                } for k, v in self.training_results.items()
            },
            'healthcare_context': 'TAT Prediction - Pharmacy Workflow Optimization',  # Clinical application
            'clinical_objective': '60-minute TAT threshold compliance and bottleneck identification',
            'target_stakeholders': ['Pharmacy operations', 'Clinical leadership', 'Quality improvement']
        }
        
        # Save metadata for healthcare MLOps integration and audit requirements
        metadata_path = output_dir / "training_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"All TAT models with feature importance saved to {output_dir}")
        
    def get_best_model(self) -> Optional[BaseTATModel]:
        """
        Get Best Performing Healthcare Model for Production Deployment
        
        Identifies top-performing model optimized for healthcare medication preparation
        TAT prediction and pharmacy workflow optimization. Selects based on clinical accuracy
        metrics ensuring optimal healthcare analytics performance for production deployment
        and real-time clinical decision support requirements.
        
        Healthcare Model Selection:
        - RMSE-based selection ensuring clinical accuracy optimization for TAT prediction
        - Performance validation confirming model training completion and deployment readiness
        - Healthcare context preservation maintaining clinical metadata and interpretability
        - Production deployment preparation with validated model instance and configuration
        - Clinical validation ensuring model safety and accuracy for patient care scenarios
        
        Returns:
            Optional[BaseTATModel]: Best performing healthcare model ready for production
            TAT prediction or None if no models successfully trained with valid performance.
        
        Example:
            For production healthcare model deployment:
            ```python
            # Get best model for clinical deployment
            best_model = orchestrator.get_best_model()
            
            if best_model:
                # Deploy for real-time TAT prediction
                predictions = best_model.predict(X_production)
                print(f"Model type: {best_model.metadata['algorithm']}")
            ```
        
        """
        # Filter for successfully trained models with valid  Numerical Features
        task_results = {
            k: v for k, v in self.training_results.items()
        }
        
        # Return None if no models successfully trained
        if not task_results:
            return None
        
        # Select model with lowest RMSE for optimal clinical accuracy
        best_model_name = min(
            task_results.keys(),
            key=lambda k: task_results[k]['metrics'].get('RMSE', float('inf'))
        )

        return task_results[best_model_name]['model']