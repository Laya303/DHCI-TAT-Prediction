"""
Test Suite for Healthcare Stacking Ensemble TAT Prediction Model

Comprehensive validation ensuring StackingTATRegressor ensemble supports Dana Farber's
medication preparation TAT prediction with multi-algorithm diversity, clinical
interpretability, and healthcare-appropriate optimization. Validates stacking
architecture, base model integration, meta-learner functionality, and ensemble
insights generation for pharmacy workflow optimization.

Test Coverage:
- StackingTATRegressor initialization with healthcare parameters
- Base model construction (Ridge, Random Forest, XGBoost)
- Meta-learner configuration and ensemble training
- Feature importance extraction from base models
- Meta-learner coefficient analysis for ensemble interpretability
- Hyperparameter space definition for healthcare optimization
- Edge cases and error handling for robust clinical deployment
- Production readiness validation with healthcare metadata preservation

Healthcare Validation:
- Clinical interpretability through base model importance analysis
- Evidence-based bottleneck identification through multi-algorithm perspectives
- Healthcare stakeholder communication through ensemble insights
- Production deployment readiness with MLOps integration capabilities
- Regulatory compliance through interpretable ensemble architecture
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, PropertyMock, patch
import tempfile
import os
from pathlib import Path
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.tat.models.ensemble_model import StackingTATRegressor


class TestEnsembleInitialization:
    """Test StackingTATRegressor ensemble initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default stacking ensemble initialization with healthcare parameters."""
        ensemble = StackingTATRegressor()
        
        # Validate ensemble parameters
        assert ensemble.ensemble_params['meta_alpha'] == 1.0
        assert ensemble.ensemble_params['xgb_n_estimators'] == 100
        assert ensemble.ensemble_params['rf_n_estimators'] == 100
        assert ensemble.ensemble_params['ridge_alpha'] == 10.0
        assert ensemble.ensemble_params['cv_folds'] == 5
        
        # Validate base models
        assert len(ensemble.base_models) == 3
        base_model_names = [name for name, _ in ensemble.base_models]
        assert 'ridge' in base_model_names
        assert 'rf' in base_model_names
        assert 'xgb' in base_model_names
        
        # Validate meta-learner
        assert isinstance(ensemble.meta_learner, Ridge)
        assert ensemble.meta_learner.alpha == 1.0
        
        # Validate stacking regressor
        assert isinstance(ensemble.model, StackingRegressor)
        assert ensemble.model.cv == 5
        
        # Validate healthcare metadata
        assert ensemble.metadata['algorithm'] == 'Stacking Ensemble Regression'
        assert ensemble.metadata['meta_learner'] == 'Ridge'
        assert len(ensemble.metadata['base_models']) == 3
        
    def test_custom_parameters_initialization(self):
        """Test stacking ensemble initialization with custom healthcare parameters."""
        custom_params = {
            'meta_alpha': 0.5,
            'xgb_n_estimators': 80,
            'rf_n_estimators': 120,
            'ridge_alpha': 5.0,
            'cv_folds': 3
        }
        
        ensemble = StackingTATRegressor(random_state=123, **custom_params)
        
        # Validate custom parameter integration
        assert ensemble.ensemble_params['meta_alpha'] == 0.5
        assert ensemble.ensemble_params['xgb_n_estimators'] == 80
        assert ensemble.ensemble_params['rf_n_estimators'] == 120
        assert ensemble.ensemble_params['ridge_alpha'] == 5.0
        assert ensemble.ensemble_params['cv_folds'] == 3
        
        # Validate meta-learner reflects custom parameters
        assert ensemble.meta_learner.alpha == 0.5
        
        # Validate stacking regressor reflects custom parameters
        assert ensemble.model.cv == 3
        
    def test_base_model_construction(self):
        """Test base model construction with healthcare optimization."""
        ensemble = StackingTATRegressor()
        
        # Validate Ridge base model
        ridge_name, ridge_model = ensemble.base_models[0]
        assert ridge_name == 'ridge'
        assert isinstance(ridge_model, Ridge)
        assert ridge_model.alpha == 10.0  # Default ridge_alpha
        
        # Validate Random Forest base model
        rf_name, rf_model = ensemble.base_models[1]
        assert rf_name == 'rf'
        assert isinstance(rf_model, RandomForestRegressor)
        assert rf_model.n_estimators == 100  # Default rf_n_estimators
        assert rf_model.max_depth == 6
        
        # Validate XGBoost base model
        xgb_name, xgb_model = ensemble.base_models[2]
        assert xgb_name == 'xgb'
        assert isinstance(xgb_model, XGBRegressor)
        assert xgb_model.n_estimators == 100  # Default xgb_n_estimators
        assert xgb_model.max_depth == 4
        
    def test_compatibility_handler_integration(self):
        """Test XGBoost compatibility handler integration."""
        ensemble = StackingTATRegressor()
        
        # Validate compatibility handler exists
        assert hasattr(ensemble, 'compatibility_handler')
        assert ensemble.compatibility_handler is not None
        
        # Validate capabilities
        assert hasattr(ensemble.compatibility_handler, 'capabilities')
        
    def test_healthcare_metadata_completeness(self):
        """Test comprehensive healthcare metadata initialization."""
        ensemble = StackingTATRegressor()
        
        # Validate clinical metadata completeness
        required_metadata = [
            'algorithm', 'base_models', 'meta_learner', 
            'healthcare_optimization', 'interpretability',
            'clinical_focus', 'ensemble_diversity'
        ]
        
        for key in required_metadata:
            assert key in ensemble.metadata
            assert isinstance(ensemble.metadata[key], (str, list))


class TestEnsembleTraining:
    """Test stacking ensemble training with healthcare data."""
    
    @pytest.fixture
    def ensemble_training_data(self):
        """Generate realistic healthcare TAT training dataset for ensemble."""
        np.random.seed(42)
        n_samples = 800
        
        # Healthcare feature simulation with diverse patterns for ensemble
        data = {
            'medication_complexity': np.random.exponential(2.8, n_samples),
            'queue_length': np.random.poisson(7, n_samples),
            'staff_experience_years': np.random.gamma(4.5, 2, n_samples),
            'time_of_day': np.random.uniform(0, 24, n_samples),
            'day_of_week': np.random.randint(1, 8, n_samples),
            'preparation_urgency': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'equipment_availability': np.random.beta(3.5, 1, n_samples),
            'verification_required': np.random.binomial(1, 0.4, n_samples),
            'batch_complexity': np.random.gamma(2.8, 1.5, n_samples),
            'workflow_efficiency': np.random.exponential(1.6, n_samples)
        }
        
        X = pd.DataFrame(data)
        
        # Complex TAT generation with multiple patterns for ensemble learning
        y = (20 + 
             X['medication_complexity'] * 12 +          # Linear pattern for Ridge
             X['queue_length'] * 3.5 +                 # Linear factor
             -X['staff_experience_years'] * 1.0 +      # Efficiency factor
             X['preparation_urgency'] * 5 +            # Priority impact
             # Non-linear interactions for tree models
             (X['medication_complexity'] * X['batch_complexity']) * 0.4 +
             np.random.exponential(4.5, n_samples))
        
        return X, pd.Series(y, name='tat_minutes')
    
    def test_successful_ensemble_training(self, ensemble_training_data):
        """Test successful stacking ensemble training."""
        X, y = ensemble_training_data
        ensemble = StackingTATRegressor(
            xgb_n_estimators=30,  # Faster for testing
            rf_n_estimators=30
        )
        
        # Train ensemble model
        fitted_ensemble = ensemble.fit(X, y)
        
        # Validate training completion
        assert fitted_ensemble is ensemble
        assert ensemble.is_fitted
        
        # Validate training metadata
        assert ensemble.metadata['training_samples'] == len(X)
        assert ensemble.metadata['feature_count'] == X.shape[1]
        assert ensemble.metadata['training_completed'] is True
        assert ensemble.metadata['clinical_deployment_ready'] is True
        
        # Validate ensemble-specific metadata
        assert ensemble.metadata['base_model_count'] == 3
        assert 'ensemble_params' in ensemble.metadata
        
    def test_target_transformation_during_training(self, ensemble_training_data):
        """Test target transformation handling during ensemble training."""
        X, y = ensemble_training_data
        ensemble = StackingTATRegressor()
        ensemble.target_transform = 'log1p'  # Set after initialization
        
        # Test with transformed targets
        ensemble.fit(X, y)
        
        assert ensemble.is_fitted
        assert ensemble.metadata['target_transform'] == 'log1p'
        
    def test_ensemble_configuration_preservation(self, ensemble_training_data):
        """Test ensemble configuration preservation after training."""
        X, y = ensemble_training_data
        ensemble = StackingTATRegressor(
            meta_alpha=2.0,
            xgb_n_estimators=40,
            rf_n_estimators=60,
            cv_folds=3
        )
        
        ensemble.fit(X, y)
        
        # Validate ensemble configuration preserved
        saved_params = ensemble.metadata['ensemble_params']
        assert saved_params['meta_alpha'] == 2.0
        assert saved_params['xgb_n_estimators'] == 40
        assert saved_params['rf_n_estimators'] == 60
        assert saved_params['cv_folds'] == 3
        
    def test_training_with_validation_data_api_consistency(self, ensemble_training_data):
        """Test ensemble training with validation data (API consistency)."""
        X, y = ensemble_training_data
        X_val, y_val = X[:100], y[:100]
        
        ensemble = StackingTATRegressor()
        
        # Validation data maintained for API consistency but not used
        ensemble.fit(X[100:], y[100:], validation_data=(X_val, y_val))
        
        assert ensemble.is_fitted
        assert ensemble.metadata['training_samples'] == len(X) - 100


class TestBaseModelImportance:
    """Test base model feature importance extraction for clinical insights."""
    
    @pytest.fixture
    def fitted_ensemble_model(self):
        """Create fitted ensemble model for importance testing."""
        np.random.seed(42)
        n_samples = 600
        
        # Healthcare features with known importance patterns
        X = pd.DataFrame({
            'critical_ensemble_factor': np.random.exponential(2.5, n_samples),     # High importance
            'moderate_ensemble_factor': np.random.gamma(2.2, 1, n_samples),       # Moderate importance  
            'minor_ensemble_factor': np.random.uniform(0, 1, n_samples),          # Low importance
            'interaction_factor_a': np.random.exponential(1.8, n_samples),        # For interactions
            'interaction_factor_b': np.random.gamma(1.8, 1.2, n_samples),        # For interactions
            'noise_factor': np.random.normal(0, 0.1, n_samples)                  # Minimal importance
        })
        
        # Target with multiple algorithmic patterns
        y = (15 + 
             X['critical_ensemble_factor'] * 16 +      # Strong linear and non-linear
             X['moderate_ensemble_factor'] * 7 +       # Moderate relationship
             X['minor_ensemble_factor'] * 2 +          # Weak relationship
             # Interactions for tree models to capture
             (X['interaction_factor_a'] * X['interaction_factor_b']) * 0.6 +
             np.random.exponential(3.8, n_samples))
        
        ensemble = StackingTATRegressor(
            xgb_n_estimators=50, 
            rf_n_estimators=50,
            random_state=42
        )
        ensemble.fit(X, pd.Series(y))
        return ensemble
    
    def test_base_model_importance_extraction(self, fitted_ensemble_model):
        """Test comprehensive base model importance extraction."""
        importance = fitted_ensemble_model.get_base_model_importance()
        
        # Validate importance results structure
        assert isinstance(importance, dict)
        assert len(importance) >= 2  # At least tree-based models have importance
        
        # Validate Random Forest importance
        if 'rf' in importance:
            rf_importance = importance['rf']
            assert rf_importance['type'] == 'feature_importance'
            assert rf_importance['algorithm'] == 'tree_based'
            assert len(rf_importance['values']) == 6  # Six features
            assert len(rf_importance['feature_names']) == 6
            
        # Validate XGBoost importance
        if 'xgb' in importance:
            xgb_importance = importance['xgb']
            assert xgb_importance['type'] == 'feature_importance'
            assert xgb_importance['algorithm'] == 'tree_based'
            assert len(xgb_importance['values']) == 6  # Six features
            
        # Validate Ridge coefficients
        if 'ridge' in importance:
            ridge_importance = importance['ridge']
            assert ridge_importance['type'] == 'coefficients'
            assert ridge_importance['algorithm'] == 'linear'
            assert len(ridge_importance['values']) == 6  # Six features
            
    def test_tree_based_importance_validation(self, fitted_ensemble_model):
        """Test tree-based model importance validation."""
        importance = fitted_ensemble_model.get_base_model_importance()
        
        # Random Forest importance validation
        if 'rf' in importance:
            rf_values = importance['rf']['values']
            assert all(val >= 0 for val in rf_values)  # Non-negative importance
            assert np.sum(rf_values) > 0  # Some features should have importance
            
        # XGBoost importance validation
        if 'xgb' in importance:
            xgb_values = importance['xgb']['values']
            assert all(val >= 0 for val in xgb_values)  # Non-negative importance
            assert np.sum(xgb_values) > 0  # Some features should have importance
            
    def test_clinical_interpretation_metadata(self, fitted_ensemble_model):
        """Test clinical interpretation metadata in base model importance."""
        importance = fitted_ensemble_model.get_base_model_importance()
        
        for model_name, model_importance in importance.items():
            assert 'clinical_interpretation' in model_importance
            assert isinstance(model_importance['clinical_interpretation'], str)
            assert len(model_importance['clinical_interpretation']) > 0
            
    def test_unfitted_ensemble_importance_error(self):
        """Test base model importance extraction error on unfitted ensemble."""
        ensemble = StackingTATRegressor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            ensemble.get_base_model_importance()


class TestMetaLearnerCoefficients:
    """Test meta-learner coefficient analysis for ensemble interpretability."""
    
    @pytest.fixture
    def meta_learner_ensemble(self):
        """Create fitted ensemble for meta-learner coefficient testing."""
        np.random.seed(42)
        n_samples = 500
        
        # Features designed to favor different algorithms
        X = pd.DataFrame({
            'linear_pattern': np.random.normal(5, 2, n_samples),           # Ridge-friendly
            'tree_pattern': np.random.exponential(2, n_samples),          # Tree-friendly
            'interaction_pattern_1': np.random.gamma(2, 1.5, n_samples),  # XGBoost-friendly
            'interaction_pattern_2': np.random.uniform(0, 10, n_samples), # General
            'noise_feature': np.random.normal(0, 0.5, n_samples)          # Noise
        })
        
        # Target with patterns that benefit different algorithms
        y = (12 + 
             X['linear_pattern'] * 3 +                                     # Linear pattern
             np.where(X['tree_pattern'] > 2, X['tree_pattern'] * 2, 5) +   # Tree pattern
             (X['interaction_pattern_1'] * X['interaction_pattern_2']) * 0.1 + # Interaction
             np.random.exponential(2.5, n_samples))
        
        ensemble = StackingTATRegressor(
            xgb_n_estimators=40,
            rf_n_estimators=40, 
            random_state=42
        )
        ensemble.fit(X, pd.Series(y))
        return ensemble
    
    def test_meta_learner_coefficient_extraction(self, meta_learner_ensemble):
        """Test comprehensive meta-learner coefficient extraction."""
        coefficients = meta_learner_ensemble.get_meta_learner_coefficients()
        
        # Validate coefficient DataFrame structure
        assert isinstance(coefficients, pd.DataFrame)
        assert len(coefficients) == 3  # Three base models
        
        # Validate required columns
        required_columns = [
            'base_model', 'meta_coefficient', 'abs_coefficient', 
            'contribution_rank', 'clinical_interpretation'
        ]
        for col in required_columns:
            assert col in coefficients.columns
            
        # Validate base model names
        base_models = coefficients['base_model'].tolist()
        assert 'ridge' in base_models
        assert 'rf' in base_models
        assert 'xgb' in base_models
        
    def test_coefficient_ranking_validation(self, meta_learner_ensemble):
        """Test meta-learner coefficient ranking and magnitude analysis."""
        coefficients = meta_learner_ensemble.get_meta_learner_coefficients()
        
        # Validate ranking
        assert coefficients['contribution_rank'].iloc[0] == 1  # Top contributor
        
        # Validate sorted by absolute coefficient (descending)
        assert coefficients['abs_coefficient'].is_monotonic_decreasing
        
        # Validate contribution ranks are sequential
        expected_ranks = sorted(coefficients['contribution_rank'].tolist())
        assert expected_ranks == [1, 2, 3]
        
    def test_clinical_interpretation_completeness(self, meta_learner_ensemble):
        """Test clinical interpretation completeness for each base model."""
        coefficients = meta_learner_ensemble.get_meta_learner_coefficients()
        
        # Validate clinical interpretations exist and are meaningful
        for _, row in coefficients.iterrows():
            interpretation = row['clinical_interpretation']
            assert isinstance(interpretation, str)
            assert len(interpretation) > 10  # Meaningful interpretation
            
        # Check for algorithm-specific interpretations
        interpretations = coefficients['clinical_interpretation'].tolist()
        interpretation_text = ' '.join(interpretations)
        assert 'linear' in interpretation_text.lower()
        assert 'pattern' in interpretation_text.lower()
        
    def test_coefficient_magnitude_analysis(self, meta_learner_ensemble):
        """Test coefficient magnitude and contribution analysis."""
        coefficients = meta_learner_ensemble.get_meta_learner_coefficients()
        
        # Validate coefficient values are reasonable
        assert all(np.isfinite(coefficients['meta_coefficient']))
        assert all(coefficients['abs_coefficient'] >= 0)
        
        # Validate absolute coefficients match meta coefficients
        expected_abs = np.abs(coefficients['meta_coefficient'])
        np.testing.assert_array_almost_equal(
            coefficients['abs_coefficient'].values, expected_abs, decimal=10
        )
        
    def test_unfitted_ensemble_coefficient_error(self):
        """Test meta-learner coefficient extraction error on unfitted ensemble."""
        ensemble = StackingTATRegressor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            ensemble.get_meta_learner_coefficients()


class TestHyperparameterSpace:
    """Test ensemble hyperparameter space definition for healthcare optimization."""
    
    def test_hyperparameter_space_structure(self):
        """Test ensemble hyperparameter space structure and ranges."""
        ensemble = StackingTATRegressor()
        param_space = ensemble.get_hyperparameter_space()
        
        # Validate parameter space structure
        assert isinstance(param_space, dict)
        
        # Validate required ensemble hyperparameters
        required_params = [
            'meta_alpha', 'xgb_n_estimators', 'rf_n_estimators', 'ridge_alpha'
        ]
        for param in required_params:
            assert param in param_space
            
    def test_meta_learner_parameter_ranges(self):
        """Test meta-learner hyperparameter ranges for healthcare."""
        ensemble = StackingTATRegressor()
        param_space = ensemble.get_hyperparameter_space()
        
        # Validate meta_alpha range for ensemble stability
        meta_alpha = param_space['meta_alpha']
        assert meta_alpha[0] == 'float'
        assert meta_alpha[1] == 0.1   # Minimum regularization
        assert meta_alpha[2] == 100.0  # Maximum regularization
        assert meta_alpha[3] == 'log'  # Logarithmic sampling
        
        # Validate ridge_alpha range for base model
        ridge_alpha = param_space['ridge_alpha']
        assert ridge_alpha[0] == 'float'
        assert ridge_alpha[1] == 1.0    # Minimum regularization
        assert ridge_alpha[2] == 100.0  # Maximum regularization
        assert ridge_alpha[3] == 'log'  # Logarithmic sampling
        
    def test_base_model_parameter_ranges(self):
        """Test base model hyperparameter ranges for ensemble diversity."""
        ensemble = StackingTATRegressor()
        param_space = ensemble.get_hyperparameter_space()
        
        # Validate XGBoost n_estimators range
        xgb_n_est = param_space['xgb_n_estimators']
        assert xgb_n_est[0] == 'int'
        assert xgb_n_est[1] == 50   # Minimum complexity
        assert xgb_n_est[2] == 150  # Maximum complexity
        
        # Validate Random Forest n_estimators range
        rf_n_est = param_space['rf_n_estimators']
        assert rf_n_est[0] == 'int'
        assert rf_n_est[1] == 50   # Minimum diversity
        assert rf_n_est[2] == 150  # Maximum diversity


class TestEnsemblePrediction:
    """Test stacking ensemble prediction capabilities."""
    
    @pytest.fixture
    def prediction_ensemble_model(self):
        """Create fitted ensemble model for prediction testing."""
        np.random.seed(42)
        n_samples = 600
        
        X = pd.DataFrame({
            'factor_1': np.random.exponential(2.2, n_samples),
            'factor_2': np.random.gamma(2.5, 1.5, n_samples), 
            'factor_3': np.random.uniform(0, 15, n_samples),
            'factor_4': np.random.normal(8, 3, n_samples),
            'factor_5': np.random.exponential(1.5, n_samples)
        })
        
        # Multi-pattern target for ensemble learning
        y = (14 + X['factor_1'] * 5 + X['factor_2'] * 3 + 
             (X['factor_1'] * X['factor_2']) * 0.2 +  # Interaction
             np.random.exponential(3.2, n_samples))
        
        ensemble = StackingTATRegressor(
            xgb_n_estimators=40, 
            rf_n_estimators=40,
            random_state=42
        )
        ensemble.fit(X, pd.Series(y))
        return ensemble, X
    
    def test_ensemble_prediction(self, prediction_ensemble_model):
        """Test stacking ensemble prediction functionality."""
        ensemble, X = prediction_ensemble_model
        
        # Test prediction
        predictions = ensemble.predict(X[:100])
        
        # Validate prediction output
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 100
        assert all(pred > 0 for pred in predictions)  # Positive TAT values
        
    def test_prediction_with_target_transformation(self, prediction_ensemble_model):
        """Test ensemble prediction with target transformation."""
        ensemble, X = prediction_ensemble_model
        
        # Enable target transformation
        ensemble.target_transform = 'log1p'
        
        predictions = ensemble.predict(X[:50])
        
        # Validate transformed predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 50
        assert all(pred > 0 for pred in predictions)
        
    def test_unfitted_ensemble_prediction_error(self):
        """Test prediction error on unfitted ensemble model."""
        ensemble = StackingTATRegressor()
        X = pd.DataFrame({'feature': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="must be fitted"):
            ensemble.predict(X)


class TestEdgeCasesAndErrorHandling:
    """Test ensemble edge cases and error handling for robust deployment."""
    
    def test_single_feature_ensemble_training(self):
        """Test ensemble training with single feature."""
        np.random.seed(42)
        X = pd.DataFrame({'single_feature': np.random.exponential(3, 200)})
        y = pd.Series(10 + X['single_feature'] * 4 + np.random.normal(0, 2, 200))
        
        ensemble = StackingTATRegressor(
            xgb_n_estimators=20,
            rf_n_estimators=20
        )
        ensemble.fit(X, y)
        
        assert ensemble.is_fitted
        assert ensemble.metadata['feature_count'] == 1
        
    def test_small_dataset_ensemble_handling(self):
        """Test ensemble handling of small healthcare datasets."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,
            'feature_2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] * 2
        })
        y = pd.Series([15, 22, 29, 36, 43, 50, 57, 64, 71, 78] * 2)
        
        ensemble = StackingTATRegressor(
            xgb_n_estimators=10,
            rf_n_estimators=10,
            cv_folds=3  # Reduced for small dataset
        )
        ensemble.fit(X, y)
        
        assert ensemble.is_fitted
        assert ensemble.metadata['training_samples'] == 20
        
    def test_extreme_tat_values_ensemble_handling(self):
        """Test ensemble handling of extreme TAT values."""
        np.random.seed(42)
        X = pd.DataFrame({
            'normal_factor': np.random.normal(8, 3, 100),
            'extreme_factor': np.random.exponential(15, 100)
        })
        
        # Include extreme TAT values
        y_extreme = pd.Series(np.concatenate([
            np.random.exponential(20, 80),  # Normal range
            [500, 800, 1100, 1400] * 4,    # Extreme values
            [1.2, 1.8, 2.5, 3.0]           # Very fast
        ]))
        
        ensemble = StackingTATRegressor(
            xgb_n_estimators=15,
            rf_n_estimators=15
        )
        ensemble.fit(X, y_extreme)
        
        assert ensemble.is_fitted
        predictions = ensemble.predict(X[:10])
        assert all(pred >= 0 for pred in predictions)


class TestProductionReadiness:
    """Test ensemble production deployment readiness and healthcare compliance."""
    
    def test_ensemble_serialization_compatibility(self):
        """Test ensemble model serialization for production deployment."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.exponential(2.5, 300),
            'feature_2': np.random.gamma(4, 1, 300),
            'feature_3': np.random.uniform(0, 20, 300)
        })
        y = pd.Series(16 + X['feature_1'] * 7 + X['feature_2'] * 4 + 
                     np.random.normal(0, 3, 300))
        
        ensemble = StackingTATRegressor(
            xgb_n_estimators=25,
            rf_n_estimators=25
        )
        ensemble.fit(X, y)
        
        # Test model save/load functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'ensemble_model'
            ensemble.save_model(model_path)
            
            # Validate model file created
            assert model_path.exists()
            
            # Test model loading
            loaded_ensemble = StackingTATRegressor.load_model(model_path)
            
            # Validate loaded model functionality
            original_pred = ensemble.predict(X[:10])
            loaded_pred = loaded_ensemble.predict(X[:10])
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)
            
    def test_healthcare_metadata_preservation(self):
        """Test healthcare metadata preservation across ensemble lifecycle."""
        np.random.seed(42)
        X = pd.DataFrame({'healthcare_feature': np.random.exponential(2.8, 200)})
        y = pd.Series(12 + X['healthcare_feature'] * 6 + np.random.normal(0, 2, 200))
        
        ensemble = StackingTATRegressor(
            meta_alpha=1.5,
            xgb_n_estimators=30,
            rf_n_estimators=40,
            ridge_alpha=8.0,
            random_state=123
        )
        ensemble.fit(X, y)
        
        # Validate comprehensive metadata preservation
        metadata = ensemble.metadata
        
        # Ensemble configuration preserved
        assert metadata['algorithm'] == 'Stacking Ensemble Regression'
        assert metadata['base_model_count'] == 3
        assert len(metadata['base_models']) == 3
        
        # Training details preserved
        assert metadata['training_samples'] == 200
        assert metadata['feature_count'] == 1
        assert metadata['training_completed'] is True
        
        # Ensemble-specific context preserved
        assert 'ensemble robustness' in metadata['healthcare_optimization']
        assert metadata['clinical_deployment_ready'] is True
        assert 'ensemble_params' in metadata
        
        # Validate ensemble parameters preserved
        saved_params = metadata['ensemble_params']
        assert saved_params['meta_alpha'] == 1.5
        assert saved_params['xgb_n_estimators'] == 30
        assert saved_params['rf_n_estimators'] == 40
        
    def test_ensemble_consistency_validation(self):
        """Test ensemble prediction consistency."""
        np.random.seed(42)
        X = pd.DataFrame({
            'consistent_feature': np.random.gamma(3, 2, 400)
        })
        y = pd.Series(18 + X['consistent_feature'] * 8 + np.random.exponential(3.5, 400))
        
        # Train multiple ensembles with same configuration
        ensemble1 = StackingTATRegressor(
            xgb_n_estimators=35, 
            rf_n_estimators=35,
            random_state=42
        )
        ensemble2 = StackingTATRegressor(
            xgb_n_estimators=35,
            rf_n_estimators=35, 
            random_state=42
        )
        
        ensemble1.fit(X, y)
        ensemble2.fit(X, y)
        
        # Test prediction consistency
        test_X = X[:20]
        pred1 = ensemble1.predict(test_X)
        pred2 = ensemble2.predict(test_X)
        
        # With same random_state, predictions should be very close
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=6)
        
    def test_ensemble_clinical_interpretability_validation(self):
        """Test ensemble clinical interpretability and transparency."""
        np.random.seed(42)
        X = pd.DataFrame({
            'critical_ensemble_factor': np.random.exponential(3.5, 350),
            'secondary_ensemble_factor': np.random.gamma(3, 2, 350),
            'operational_ensemble_factor': np.random.uniform(1, 15, 350),
            'interaction_ensemble_factor': np.random.exponential(2, 350)
        })
        
        # Clear patterns for ensemble learning
        y = (22 + 
             X['critical_ensemble_factor'] * 12 +           # Strong impact
             X['secondary_ensemble_factor'] * 5 +           # Moderate impact  
             X['operational_ensemble_factor'] * 2 +         # Minor impact
             # Interaction for ensemble diversity
             (X['critical_ensemble_factor'] * X['interaction_ensemble_factor']) * 0.3 +
             np.random.exponential(4, 350))
        
        ensemble = StackingTATRegressor(
            xgb_n_estimators=60,
            rf_n_estimators=60, 
            random_state=42
        )
        ensemble.fit(X, pd.Series(y))
        
        # Validate clinical interpretability through base model importance
        base_importance = ensemble.get_base_model_importance()
        
        # Should have importance from interpretable models
        assert len(base_importance) >= 2
        
        # Generate meta-learner insights
        meta_coefficients = ensemble.get_meta_learner_coefficients()
        
        # Validate ensemble interpretability completeness
        assert len(meta_coefficients) == 3  # Three base models
        assert 'contribution_rank' in meta_coefficients.columns
        
    def test_ensemble_healthcare_compliance_validation(self):
        """Test ensemble healthcare regulatory compliance features."""
        ensemble = StackingTATRegressor(random_state=42)
        
        # Validate ensemble compliance attributes
        assert 'interpretability' in ensemble.metadata
        interpretability_info = ensemble.metadata['interpretability']
        assert 'Limited' in interpretability_info  # Ensemble black box limitation
        assert 'base model insights' in interpretability_info
        
        # Validate ensemble diversity documentation
        assert 'ensemble_diversity' in ensemble.metadata
        diversity_info = ensemble.metadata['ensemble_diversity']
        assert 'Linear' in diversity_info
        assert 'Tree-based' in diversity_info
        assert 'Gradient boosting' in diversity_info
        
        # Validate clinical focus
        assert 'clinical_focus' in ensemble.metadata
        clinical_info = ensemble.metadata['clinical_focus']
        assert '60-minute' in clinical_info
        assert 'TAT' in clinical_info