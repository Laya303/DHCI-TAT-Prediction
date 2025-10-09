"""
Test Suite for Healthcare XGBoost TAT Prediction Model

Comprehensive validation ensuring XGBoost gradient boosting supports Dana Farber's
medication preparation TAT prediction with advanced pattern recognition, clinical
interpretability, and healthcare-appropriate hyperparameter optimization. Validates
gradient boosting architecture, feature importance analysis, and clinical insight
generation for pharmacy workflow optimization and complex bottleneck identification.

Test Coverage:
- XGBoostTATRegressor initialization with healthcare parameters
- Gradient boosting training with validation and early stopping support
- Feature importance extraction and clinical bottleneck identification
- Clinical insights generation with intervention recommendations
- Hyperparameter space definition for healthcare optimization  
- XGBoost compatibility handling across versions
- Edge cases and error handling for robust clinical deployment
- Production readiness validation with healthcare metadata preservation

"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, PropertyMock, patch
import tempfile
import os
from pathlib import Path
import xgboost as xgb
from xgboost import XGBRegressor

from src.tat.models.xgboost_model import XGBoostTATRegressor


class TestXGBoostInitialization:
    """Test XGBoostTATRegressor model initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default XGBoost model initialization with healthcare parameters."""
        model = XGBoostTATRegressor()
        
        # Validate healthcare-optimized default parameters
        assert model.default_params['n_estimators'] == 150
        assert model.default_params['max_depth'] == 6
        assert model.default_params['learning_rate'] == 0.1
        assert model.default_params['subsample'] == 0.8
        assert model.default_params['colsample_bytree'] == 0.8
        assert model.default_params['reg_alpha'] == 0.1
        assert model.default_params['reg_lambda'] == 0.1
        assert model.default_params['objective'] == 'reg:squarederror'
        assert model.default_params['eval_metric'] == 'rmse'
        
        # Validate XGBoost model initialization
        assert isinstance(model.model, XGBRegressor)
        assert model.model.n_estimators == 150
        assert model.model.max_depth == 6
        assert model.model.learning_rate == 0.1
        
        # Validate compatibility handler
        assert hasattr(model, 'compatibility_handler')
        assert model.compatibility_handler is not None
        
        # Validate healthcare metadata
        assert model.metadata['algorithm'] == 'XGBoost Gradient Boosting Regression'
        assert 'xgboost_version' in model.metadata
        assert 'clinical interpretability' in model.metadata['healthcare_optimization']
        assert 'Tree-based feature importance' in model.metadata['interpretability']
        
    def test_custom_parameters_initialization(self):
        """Test XGBoost initialization with custom healthcare parameters."""
        custom_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'reg_alpha': 0.2,
            'reg_lambda': 0.3
        }
        
        model = XGBoostTATRegressor(random_state=123, **custom_params)
        
        # Validate custom parameter integration
        assert model.default_params['n_estimators'] == 100
        assert model.default_params['max_depth'] == 4
        assert model.default_params['learning_rate'] == 0.05
        assert model.default_params['reg_alpha'] == 0.2
        assert model.default_params['reg_lambda'] == 0.3
        assert model.default_params['random_state'] == 123
        
        # Validate model reflects custom parameters
        assert model.model.n_estimators == 100
        assert model.model.max_depth == 4
        assert model.model.learning_rate == 0.05
        
    def test_healthcare_metadata_completeness(self):
        """Test comprehensive healthcare metadata initialization."""
        model = XGBoostTATRegressor()
        
        # Validate clinical metadata completeness
        required_metadata = [
            'algorithm', 'xgboost_version', 'healthcare_optimization', 
            'interpretability', 'clinical_advantages', 'bottleneck_identification',
            'regulatory_compliance', 'deployment_readiness', 'xgboost_capabilities'
        ]
        
        for key in required_metadata:
            assert key in model.metadata
            assert isinstance(model.metadata[key], (str, dict))
            
        # Validate XGBoost version tracking
        assert model.metadata['xgboost_version'] == xgb.__version__
        
    def test_compatibility_handler_integration(self):
        """Test XGBoost compatibility handler integration."""
        model = XGBoostTATRegressor()
        
        # Validate compatibility handler attributes
        assert hasattr(model.compatibility_handler, 'capabilities')
        assert hasattr(model.compatibility_handler, 'fit_with_validation')
        
        # Validate capabilities in metadata
        assert 'xgboost_capabilities' in model.metadata
        capabilities = model.metadata['xgboost_capabilities']
        assert isinstance(capabilities, dict)


class TestXGBoostTraining:
    """Test XGBoost gradient boosting training with healthcare data."""
    
    @pytest.fixture
    def training_data(self):
        """Generate realistic healthcare TAT training dataset."""
        np.random.seed(42)
        n_samples = 1000
        
        # Healthcare feature simulation with complex relationships
        data = {
            'medication_complexity': np.random.exponential(2.5, n_samples),
            'queue_length': np.random.poisson(6, n_samples),
            'staff_experience_years': np.random.gamma(4, 2, n_samples),
            'time_of_day': np.random.uniform(0, 24, n_samples),
            'day_of_week': np.random.randint(1, 8, n_samples),
            'preparation_urgency': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'equipment_availability': np.random.beta(4, 1, n_samples),
            'verification_required': np.random.binomial(1, 0.35, n_samples),
            'batch_size': np.random.gamma(2, 3, n_samples),
            'complexity_score': np.random.exponential(1.8, n_samples)
        }
        
        X = pd.DataFrame(data)
        
        # Complex TAT generation with non-linear relationships for XGBoost
        y = (18 + 
             X['medication_complexity'] * 10 +
             X['queue_length'] * 2.5 +
             -X['staff_experience_years'] * 0.8 +
             X['preparation_urgency'] * 4 +
             X['complexity_score'] * 3 +
             # Non-linear interactions for XGBoost to capture
             (X['medication_complexity'] * X['queue_length']) * 0.5 +
             np.random.exponential(4, n_samples))
        
        return X, pd.Series(y, name='tat_minutes')
    
    def test_successful_training_without_validation(self, training_data):
        """Test successful XGBoost training without validation data."""
        X, y = training_data
        model = XGBoostTATRegressor(n_estimators=50)  # Faster for testing
        
        # Train gradient boosting model
        fitted_model = model.fit(X, y)
        
        # Validate training completion
        assert fitted_model is model
        assert model.is_fitted
        
        # Validate training metadata
        assert model.metadata['training_samples'] == len(X)
        assert model.metadata['feature_count'] == X.shape[1]
        assert model.metadata['validation_used'] is False
        assert model.metadata['training_completed'] is True
        assert model.metadata['clinical_deployment_ready'] is True
        
        # Validate XGBoost-specific metadata
        assert model.metadata['n_estimators_trained'] == 50
        assert model.metadata['feature_importances_available'] is True
        
    def test_successful_training_with_validation(self, training_data):
        """Test successful XGBoost training with validation data and early stopping."""
        X, y = training_data
        
        # Split data for validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model = XGBoostTATRegressor(n_estimators=100)
        
        # Train with validation data
        fitted_model = model.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        # Validate training completion
        assert fitted_model is model
        assert model.is_fitted
        assert model.metadata['validation_used'] is True
        
    def test_target_transformation_during_training(self, training_data):
        """Test target transformation handling during XGBoost training."""
        X, y = training_data
        model = XGBoostTATRegressor()
        model.target_transform = 'log1p'  # Set after initialization
        
        # Test with transformed targets
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.metadata['target_transform'] == 'log1p'
        
    def test_gradient_boosting_configuration_preservation(self, training_data):
        """Test XGBoost gradient boosting configuration preservation after training."""
        X, y = training_data
        model = XGBoostTATRegressor(
            n_estimators=75, 
            max_depth=8, 
            learning_rate=0.05
        )
        
        model.fit(X, y)
        
        # Validate gradient boosting configuration preserved
        assert model.metadata['n_estimators_trained'] == 75
        assert model.metadata['max_depth_used'] == 8
        assert model.metadata['learning_rate_used'] == 0.05
        assert model.metadata['xgboost_interpretability_confirmed'] is True


class TestFeatureImportance:
    """Test XGBoost feature importance extraction for clinical insights."""
    
    @pytest.fixture
    def fitted_xgboost_model(self):
        """Create fitted XGBoost model for importance testing."""
        np.random.seed(42)
        n_samples = 600
        
        # Healthcare features with known importance patterns for XGBoost
        X = pd.DataFrame({
            'critical_workflow_factor': np.random.exponential(2, n_samples),    # High importance
            'moderate_operational_factor': np.random.gamma(2, 1, n_samples),   # Moderate importance  
            'minor_environmental_factor': np.random.uniform(0, 1, n_samples),  # Low importance
            'noise_factor': np.random.normal(0, 0.1, n_samples),              # Minimal importance
            'interaction_factor_1': np.random.exponential(1.5, n_samples),    # For interactions
            'interaction_factor_2': np.random.gamma(1.5, 1, n_samples)        # For interactions
        })
        
        # Target with clear feature relationships and interactions for XGBoost
        y = (12 + 
             X['critical_workflow_factor'] * 18 +        # Strong relationship
             X['moderate_operational_factor'] * 6 +      # Moderate relationship
             X['minor_environmental_factor'] * 2 +       # Weak relationship
             # Non-linear interaction for XGBoost to capture  
             (X['interaction_factor_1'] * X['interaction_factor_2']) * 0.8 +
             np.random.exponential(3, n_samples))
        
        model = XGBoostTATRegressor(n_estimators=80, random_state=42)
        model.fit(X, pd.Series(y))
        return model
    
    def test_feature_importance_extraction(self, fitted_xgboost_model):
        """Test comprehensive XGBoost feature importance extraction."""
        importance = fitted_xgboost_model.get_feature_importance()
        
        # Validate importance DataFrame structure
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 6  # Six features
        
        # Validate required columns
        required_columns = [
            'feature', 'importance', 'importance_pct', 'importance_rank',
            'clinical_significance', 'bottleneck_potential', 'gradient_boosting_gain'
        ]
        for col in required_columns:
            assert col in importance.columns
            
        # Validate healthcare context columns
        healthcare_columns = [
            'healthcare_context', 'clinical_objective', 
            'xgboost_confidence', 'interpretation_method'
        ]
        for col in healthcare_columns:
            assert col in importance.columns
            
    def test_gain_based_importance_calculation(self, fitted_xgboost_model):
        """Test XGBoost gain-based importance calculation and ranking."""
        importance = fitted_xgboost_model.get_feature_importance()
        
        # Validate importance ranking
        assert importance['importance_rank'].iloc[0] == 1  # Top feature
        # After sorting by importance (descending), ranks should be in order
        sorted_importance = importance.sort_values('importance_rank')
        assert sorted_importance['importance_rank'].is_monotonic_increasing
        
        # Validate percentage calculation sums to 100
        assert abs(importance['importance_pct'].sum() - 100.0) < 0.001
        assert all(importance['importance_pct'] >= 0)
        
        # Validate sorted by importance (descending)
        assert importance['importance'].is_monotonic_decreasing
        
        # Validate gradient boosting gain values
        assert all(importance['gradient_boosting_gain'] >= 0)
        np.testing.assert_array_equal(
            importance['importance'].values, 
            importance['gradient_boosting_gain'].values
        )
        
    def test_clinical_significance_categorization(self, fitted_xgboost_model):
        """Test clinical significance and bottleneck potential categorization."""
        importance = fitted_xgboost_model.get_feature_importance()
        
        # Validate clinical significance categories
        sig_categories = importance['clinical_significance'].unique()
        assert all(cat in ['High', 'Moderate', 'Low'] for cat in sig_categories)
        
        # Validate bottleneck potential categories
        bottleneck_categories = importance['bottleneck_potential'].unique()
        valid_bottlenecks = ['Critical', 'Significant', 'Moderate', 'Limited']
        assert all(cat in valid_bottlenecks for cat in bottleneck_categories)
        
    def test_xgboost_context_metadata(self, fitted_xgboost_model):
        """Test XGBoost-specific context and metadata in feature importance."""
        importance = fitted_xgboost_model.get_feature_importance()
        
        # Validate XGBoost-specific metadata
        assert importance['xgboost_confidence'].iloc[0] == 'High - Gradient boosting ensemble consensus'
        assert importance['interpretation_method'].iloc[0] == 'Gain-based importance from XGBoost trees'
        
        # Validate healthcare context consistency
        assert all(importance['healthcare_context'] == 'Medication preparation TAT prediction')
        assert all(importance['clinical_objective'] == '60-minute threshold optimization')
        
    def test_unfitted_model_importance_error(self):
        """Test feature importance extraction error on unfitted XGBoost model."""
        model = XGBoostTATRegressor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.get_feature_importance()


class TestClinicalInsights:
    """Test XGBoost clinical insights generation for healthcare stakeholders."""
    
    @pytest.fixture
    def clinical_xgboost_model(self):
        """Create XGBoost model with clinical feature patterns."""
        np.random.seed(42)
        n_samples = 900
        
        # Healthcare features with clinical relevance for gradient boosting
        X = pd.DataFrame({
            'medication_complexity': np.random.exponential(2.2, n_samples),
            'pharmacist_experience': np.random.gamma(3.5, 2, n_samples),
            'queue_depth': np.random.poisson(5, n_samples),
            'verification_steps': np.random.randint(1, 7, n_samples),
            'preparation_urgency': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'equipment_status': np.random.beta(3.5, 1, n_samples),
            'batch_complexity': np.random.gamma(2.5, 1.5, n_samples),
            'time_pressure': np.random.exponential(1.8, n_samples)
        })
        
        # Clinical TAT with complex patterns for XGBoost to capture
        y = (15 + 
             X['medication_complexity'] * 14 +          # Critical bottleneck
             X['queue_depth'] * 5 +                     # Significant factor
             X['verification_steps'] * 3.5 +            # Moderate factor
             -X['pharmacist_experience'] * 1.2 +        # Efficiency factor
             X['batch_complexity'] * 2.5 +              # Operational factor
             # Complex interactions for XGBoost
             (X['medication_complexity'] * X['time_pressure']) * 0.6 +
             np.random.exponential(3.5, n_samples))
        
        model = XGBoostTATRegressor(n_estimators=120, random_state=42)
        model.fit(X, pd.Series(y))
        return model
    
    def test_clinical_insights_generation(self, clinical_xgboost_model):
        """Test comprehensive XGBoost clinical insights generation."""
        insights = clinical_xgboost_model.get_clinical_insights()
        
        # Validate insights structure
        assert isinstance(insights, dict)
        
        # Validate required insight components
        required_components = [
            'model_type', 'clinical_objective', 'healthcare_context',
            'critical_bottlenecks', 'significant_drivers', 
            'intervention_recommendations', 'xgboost_advantages'
        ]
        for component in required_components:
            assert component in insights
            
        # Validate XGBoost-specific components
        xgboost_components = [
            'xgboost_version', 'model_parameters', 'regulatory_compliance',
            'deployment_readiness', 'clinical_safety'
        ]
        for component in xgboost_components:
            assert component in insights
            
    def test_critical_bottleneck_identification(self, clinical_xgboost_model):
        """Test critical bottleneck identification and analysis."""
        insights = clinical_xgboost_model.get_clinical_insights()
        
        critical_bottlenecks = insights['critical_bottlenecks']
        
        # Validate critical bottleneck structure
        assert isinstance(critical_bottlenecks, list)
        assert len(critical_bottlenecks) <= 5  # Top 5 maximum
        
        if critical_bottlenecks:
            bottleneck = critical_bottlenecks[0]
            
            # Validate XGBoost-specific bottleneck attributes
            required_attrs = [
                'feature', 'importance_pct', 'clinical_impact',
                'intervention_priority', 'bottleneck_type', 'evidence_strength',
                'gradient_boosting_gain'
            ]
            for attr in required_attrs:
                assert attr in bottleneck
                
            # Validate XGBoost-specific intervention priority
            assert bottleneck['intervention_priority'] == 'Immediate'
            assert bottleneck['bottleneck_type'] == 'High-impact workflow constraint'
            assert 'XGBoost ensemble consensus' in bottleneck['evidence_strength']
            
            # Validate gradient boosting gain is included
            assert 'gradient_boosting_gain' in bottleneck
            assert isinstance(bottleneck['gradient_boosting_gain'], (int, float))
            
    def test_significant_driver_analysis(self, clinical_xgboost_model):
        """Test significant driver identification and categorization."""
        insights = clinical_xgboost_model.get_clinical_insights()
        
        significant_drivers = insights['significant_drivers']
        
        # Validate significant drivers structure  
        assert isinstance(significant_drivers, list)
        assert len(significant_drivers) <= 5  # Top 5 maximum
        
        if significant_drivers:
            driver = significant_drivers[0]
            
            # Validate XGBoost-specific driver attributes
            assert 'feature' in driver
            assert 'importance_pct' in driver
            assert 'clinical_impact' in driver
            assert 'gradient_boosting_gain' in driver
            
            # Validate intervention priority
            assert driver['intervention_priority'] == 'High'
            assert driver['bottleneck_type'] == 'Secondary workflow factor'
            assert 'gradient boosting' in driver['evidence_strength']
            
    def test_xgboost_advantages_documentation(self, clinical_xgboost_model):
        """Test XGBoost-specific advantages documentation."""
        insights = clinical_xgboost_model.get_clinical_insights()
        
        advantages = insights['xgboost_advantages']
        
        # Validate advantages structure
        assert isinstance(advantages, list)
        assert len(advantages) >= 4  # Multiple advantages documented
        
        # Check for key XGBoost advantages
        advantage_text = ' '.join(advantages)
        assert 'gradient boosting' in advantage_text.lower()
        assert 'pattern recognition' in advantage_text.lower()
        assert 'regularization' in advantage_text.lower()
        assert 'interaction' in advantage_text.lower()
        
    def test_model_parameters_documentation(self, clinical_xgboost_model):
        """Test XGBoost model parameters documentation in insights."""
        insights = clinical_xgboost_model.get_clinical_insights()
        
        model_params = insights['model_parameters']
        
        # Validate model parameters structure
        assert isinstance(model_params, dict)
        
        # Validate required parameters
        required_params = [
            'n_estimators', 'max_depth', 'learning_rate',
            'regularization_alpha', 'regularization_lambda'
        ]
        for param in required_params:
            assert param in model_params
            assert isinstance(model_params[param], (int, float))
            
    def test_clinical_validation_metrics(self, clinical_xgboost_model):
        """Test clinical validation and confidence metrics."""
        insights = clinical_xgboost_model.get_clinical_insights()
        
        # Validate validation metrics
        assert 'interpretability_confidence' in insights
        assert 'clinical_validation' in insights
        assert 'total_features_analyzed' in insights
        
        # Check XGBoost-specific confidence levels
        assert 'gradient boosting' in insights['interpretability_confidence']
        assert isinstance(insights['total_features_analyzed'], int)
        assert insights['total_features_analyzed'] > 0
        
        # Validate XGBoost version tracking
        assert 'xgboost_version' in insights
        
    def test_intervention_recommendations(self, clinical_xgboost_model):
        """Test XGBoost-specific intervention recommendation generation."""
        insights = clinical_xgboost_model.get_clinical_insights()
        
        recommendations = insights['intervention_recommendations']
        
        # Validate recommendations structure
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 2  # Multiple recommendations
        
        # Validate recommendation content
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0
            
        # Check for XGBoost-specific recommendations
        rec_text = ' '.join(recommendations)
        assert 'gradient boosting' in rec_text.lower()
        assert 'shap' in rec_text.lower()  # SHAP analysis recommendation
        
    def test_unfitted_model_insights_error(self):
        """Test clinical insights error on unfitted XGBoost model."""
        model = XGBoostTATRegressor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.get_clinical_insights()


class TestHyperparameterSpace:
    """Test XGBoost hyperparameter space definition for healthcare optimization."""
    
    def test_hyperparameter_space_structure(self):
        """Test XGBoost hyperparameter space structure and ranges."""
        model = XGBoostTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Validate parameter space structure
        assert isinstance(param_space, dict)
        
        # Validate required XGBoost hyperparameters
        required_params = [
            'n_estimators', 'max_depth', 'learning_rate',
            'subsample', 'colsample_bytree', 'reg_alpha', 
            'reg_lambda', 'gamma', 'min_child_weight'
        ]
        for param in required_params:
            assert param in param_space
            
    def test_gradient_boosting_parameter_ranges(self):
        """Test XGBoost gradient boosting parameter ranges for healthcare."""
        model = XGBoostTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Validate n_estimators range for gradient boosting
        n_est = param_space['n_estimators']
        assert n_est[0] == 'int'
        assert n_est[1] == 50   # Minimum ensemble size
        assert n_est[2] == 300  # Maximum ensemble size
        
        # Validate max_depth range for interpretability
        depth = param_space['max_depth']
        assert depth[0] == 'int'
        assert depth[1] == 3  # Minimum interpretability
        assert depth[2] == 8  # Maximum complexity
        
        # Validate learning_rate range for stability
        lr = param_space['learning_rate']
        assert lr[0] == 'float'
        assert lr[1] == 0.05  # Conservative minimum
        assert lr[2] == 0.3   # Conservative maximum
        
    def test_regularization_parameter_ranges(self):
        """Test XGBoost regularization parameter ranges for healthcare."""
        model = XGBoostTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Validate subsample range for regularization
        subsample = param_space['subsample']
        assert subsample[0] == 'float'
        assert subsample[1] == 0.7  # Minimum sampling
        assert subsample[2] == 1.0  # Maximum sampling
        
        # Validate colsample_bytree range for diversity
        colsample = param_space['colsample_bytree']
        assert colsample[0] == 'float'
        assert colsample[1] == 0.7  # Minimum feature sampling
        assert colsample[2] == 1.0  # Maximum feature sampling
        
        # Validate regularization alpha range
        alpha = param_space['reg_alpha']
        assert alpha[0] == 'float'
        assert alpha[1] == 0.0  # No L1 regularization
        assert alpha[2] == 1.0  # Strong L1 regularization
        
        # Validate regularization lambda range
        lambda_param = param_space['reg_lambda']
        assert lambda_param[0] == 'float'
        assert lambda_param[1] == 0.0  # No L2 regularization
        assert lambda_param[2] == 1.0  # Strong L2 regularization
        
    def test_tree_structure_parameter_ranges(self):
        """Test XGBoost tree structure parameter ranges for clinical interpretability."""
        model = XGBoostTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Validate gamma range for minimum split loss
        gamma = param_space['gamma']
        assert gamma[0] == 'float'
        assert gamma[1] == 0.0  # No minimum split loss
        assert gamma[2] == 0.5  # Conservative maximum
        
        # Validate min_child_weight range for regularization
        min_child = param_space['min_child_weight']
        assert min_child[0] == 'int'
        assert min_child[1] == 1   # Minimum child weight
        assert min_child[2] == 10  # Conservative maximum


class TestXGBoostPrediction:
    """Test XGBoost gradient boosting prediction capabilities."""
    
    @pytest.fixture
    def prediction_xgboost_model(self):
        """Create fitted XGBoost model for prediction testing."""
        np.random.seed(42)
        n_samples = 700
        
        X = pd.DataFrame({
            'factor_1': np.random.exponential(1.8, n_samples),
            'factor_2': np.random.gamma(2.2, 1.5, n_samples), 
            'factor_3': np.random.uniform(0, 12, n_samples),
            'factor_4': np.random.normal(6, 2.5, n_samples),
            'factor_5': np.random.exponential(1.2, n_samples)
        })
        
        # Complex relationships for XGBoost to capture
        y = (10 + X['factor_1'] * 4 + X['factor_2'] * 2.5 + 
             (X['factor_1'] * X['factor_2']) * 0.3 +  # Interaction
             np.random.exponential(2.5, n_samples))
        
        model = XGBoostTATRegressor(n_estimators=60, random_state=42)
        model.fit(X, pd.Series(y))
        return model, X
    
    def test_gradient_boosting_prediction(self, prediction_xgboost_model):
        """Test XGBoost gradient boosting prediction functionality."""
        model, X = prediction_xgboost_model
        
        # Test prediction
        predictions = model.predict(X[:100])
        
        # Validate prediction output
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 100
        assert all(pred > 0 for pred in predictions)  # Positive TAT values
        
    def test_prediction_with_target_transformation(self, prediction_xgboost_model):
        """Test XGBoost prediction with target transformation."""
        model, X = prediction_xgboost_model
        
        # Enable target transformation
        model.target_transform = 'log1p'
        
        predictions = model.predict(X[:50])
        
        # Validate transformed predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 50
        assert all(pred > 0 for pred in predictions)
        
    def test_unfitted_model_prediction_error(self):
        """Test prediction error on unfitted XGBoost model."""
        model = XGBoostTATRegressor()
        X = pd.DataFrame({'feature': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)


class TestCompatibilityHandling:
    """Test XGBoost compatibility handling across versions."""
    
    def test_compatibility_handler_initialization(self):
        """Test XGBoost compatibility handler initialization."""
        model = XGBoostTATRegressor()
        
        # Validate compatibility handler exists
        assert hasattr(model, 'compatibility_handler')
        assert model.compatibility_handler is not None
        
        # Validate capabilities tracking
        capabilities = model.compatibility_handler.capabilities
        assert isinstance(capabilities, dict)
        
    def test_compatibility_handler_training(self):
        """Test XGBoost compatibility handler integration during training."""
        model = XGBoostTATRegressor()
        
        # Prepare training data
        np.random.seed(42)
        X = pd.DataFrame({'feature': np.random.normal(0, 1, 100)})
        y = pd.Series(np.random.exponential(10, 100))
        X_val = pd.DataFrame({'feature': np.random.normal(0, 1, 20)})
        y_val = pd.Series(np.random.exponential(10, 20))
        
        # Train with validation data - this should work regardless of XGBoost version
        model.fit(X, y, validation_data=(X_val, y_val))
        
        # Verify training completed successfully
        assert model.is_fitted
        assert model.metadata['validation_used'] is True
        
        # Verify compatibility handler exists and has capabilities
        assert hasattr(model, 'compatibility_handler')
        assert hasattr(model.compatibility_handler, 'capabilities')


class TestEdgeCasesAndErrorHandling:
    """Test XGBoost edge cases and error handling for robust deployment."""
    
    def test_single_feature_xgboost_training(self):
        """Test XGBoost training with single feature."""
        np.random.seed(42)
        X = pd.DataFrame({'single_feature': np.random.exponential(2.5, 200)})
        y = pd.Series(8 + X['single_feature'] * 4 + np.random.normal(0, 1.5, 200))
        
        model = XGBoostTATRegressor(n_estimators=30)
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.metadata['feature_count'] == 1
        
    def test_small_dataset_xgboost_handling(self):
        """Test XGBoost handling of small healthcare datasets."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature_2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        })
        y = pd.Series([12, 18, 24, 30, 36, 42, 48, 54, 60, 66])
        
        model = XGBoostTATRegressor(n_estimators=10)
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.metadata['training_samples'] == 10
        
    def test_extreme_tat_values_xgboost_handling(self):
        """Test XGBoost handling of extreme TAT values."""
        np.random.seed(42)
        X = pd.DataFrame({
            'normal_factor': np.random.normal(6, 2, 100),
            'extreme_factor': np.random.exponential(12, 100)
        })
        
        # Include extreme TAT values
        y_extreme = pd.Series(np.concatenate([
            np.random.exponential(18, 85),  # Normal range
            [400, 600, 900, 1200] * 3,     # Extreme values
            [0.8, 1.2, 1.5]                # Very fast
        ]))
        
        model = XGBoostTATRegressor(n_estimators=25)
        model.fit(X, y_extreme)
        
        assert model.is_fitted
        predictions = model.predict(X[:10])
        assert all(pred >= 0 for pred in predictions)
        
    def test_missing_feature_importance_attributes(self):
        """Test handling of XGBoost without feature importance."""
        model = XGBoostTATRegressor()
        
        # Create a mock that lacks the feature_importances_ attribute
        class MockXGB:
            pass
        
        model.model = MockXGB()
        model.is_fitted = True
        
        with pytest.raises(AttributeError, match="does not have feature importance"):
            model.get_feature_importance()


class TestProductionReadiness:
    """Test XGBoost production deployment readiness and healthcare compliance."""
    
    def test_xgboost_serialization_compatibility(self):
        """Test XGBoost model serialization for production deployment."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.exponential(2.2, 300),
            'feature_2': np.random.gamma(3.5, 1, 300),
            'feature_3': np.random.uniform(0, 15, 300)
        })
        y = pd.Series(12 + X['feature_1'] * 6 + X['feature_2'] * 3 + 
                     np.random.normal(0, 2.5, 300))
        
        model = XGBoostTATRegressor(n_estimators=40)
        model.fit(X, y)
        
        # Test model save/load functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'xgb_model'
            model.save_model(model_path)
            
            # Validate model file created
            assert model_path.exists()
            
            # Test model loading
            loaded_model = XGBoostTATRegressor.load_model(model_path)
            
            # Validate loaded model functionality
            original_pred = model.predict(X[:10])
            loaded_pred = loaded_model.predict(X[:10])
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)
            
    def test_healthcare_metadata_preservation(self):
        """Test healthcare metadata preservation across XGBoost model lifecycle."""
        np.random.seed(42)
        X = pd.DataFrame({'healthcare_feature': np.random.exponential(2.3, 200)})
        y = pd.Series(10 + X['healthcare_feature'] * 5 + np.random.normal(0, 1.8, 200))
        
        model = XGBoostTATRegressor(
            n_estimators=35,
            max_depth=5,
            learning_rate=0.08,
            random_state=123
        )
        model.fit(X, y)
        
        # Validate comprehensive metadata preservation
        metadata = model.metadata
        
        # XGBoost configuration preserved
        assert metadata['algorithm'] == 'XGBoost Gradient Boosting Regression'
        assert metadata['n_estimators_trained'] == 35
        assert metadata['max_depth_used'] == 5
        assert metadata['learning_rate_used'] == 0.08
        
        # Training details preserved
        assert metadata['training_samples'] == 200
        assert metadata['feature_count'] == 1
        assert metadata['training_completed'] is True
        
        # XGBoost-specific context preserved
        assert 'clinical interpretability' in metadata['healthcare_optimization']
        assert metadata['clinical_deployment_ready'] is True
        assert 'xgboost_version' in metadata
        
    def test_xgboost_consistency_validation(self):
        """Test XGBoost prediction consistency."""
        np.random.seed(42)
        X = pd.DataFrame({
            'consistent_feature': np.random.gamma(2.5, 2, 400)
        })
        y = pd.Series(14 + X['consistent_feature'] * 7 + np.random.exponential(2.8, 400))
        
        # Train multiple models with same configuration
        model1 = XGBoostTATRegressor(n_estimators=60, random_state=42)
        model2 = XGBoostTATRegressor(n_estimators=60, random_state=42)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Test prediction consistency
        test_X = X[:20]
        pred1 = model1.predict(test_X)
        pred2 = model2.predict(test_X)
        
        # With same random_state, predictions should be very close
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=8)
        
    def test_xgboost_clinical_interpretability_validation(self):
        """Test XGBoost clinical interpretability and transparency."""
        np.random.seed(42)
        X = pd.DataFrame({
            'critical_xgb_factor': np.random.exponential(3.2, 350),
            'secondary_xgb_factor': np.random.gamma(2.8, 2, 350),
            'operational_xgb_factor': np.random.uniform(1, 12, 350),
            'interaction_factor': np.random.exponential(1.8, 350)
        })
        
        # Clear clinical relationships with interactions for XGBoost
        y = (18 + 
             X['critical_xgb_factor'] * 10 +             # Strong clinical impact
             X['secondary_xgb_factor'] * 4 +             # Moderate impact  
             X['operational_xgb_factor'] * 1.5 +         # Minor impact
             # Interaction for XGBoost to capture
             (X['critical_xgb_factor'] * X['interaction_factor']) * 0.4 +
             np.random.exponential(3.2, 350))
        
        model = XGBoostTATRegressor(n_estimators=100, random_state=42)
        model.fit(X, pd.Series(y))
        
        # Validate clinical interpretability through feature importance
        importance = model.get_feature_importance()
        
        # Critical factor should have highest importance
        top_feature = importance.iloc[0]['feature']
        assert 'critical_xgb_factor' == top_feature
        
        # Generate clinical insights
        insights = model.get_clinical_insights()
        
        # Validate XGBoost clinical insight completeness
        assert len(insights['critical_bottlenecks']) > 0
        assert len(insights['intervention_recommendations']) > 0
        assert 'gradient boosting' in insights['interpretability_confidence']
        
        # Validate XGBoost-specific advantages
        assert len(insights['xgboost_advantages']) >= 4
        
    def test_xgboost_healthcare_compliance_validation(self):
        """Test XGBoost healthcare regulatory compliance features."""
        model = XGBoostTATRegressor(random_state=42)
        
        # Validate regulatory compliance attributes
        assert 'regulatory_compliance' in model.metadata
        compliance_info = model.metadata['regulatory_compliance']
        assert 'Interpretable' in compliance_info
        assert 'tree-based' in compliance_info
        
        # Validate XGBoost deployment readiness
        assert 'deployment_readiness' in model.metadata
        deployment_info = model.metadata['deployment_readiness']
        assert 'MLOps' in deployment_info
        assert 'compatibility handling' in deployment_info
        
        # Validate XGBoost version tracking
        assert 'xgboost_version' in model.metadata
        assert model.metadata['xgboost_version'] == xgb.__version__