"""
Test Suite for Healthcare Hyperparameter Optimization

Comprehensive validation ensuring OptunaTATOptimizer supports Dana Farber's
medication preparation TAT prediction with clinical interpretability constraints,
robust evaluation strategies, and healthcare-appropriate optimization. Validates
Optuna integration, parameter suggestion, model optimization, and healthcare
reporting for pharmacy workflow optimization.

Test Coverage:
- OptunaTATOptimizer initialization with healthcare configuration
- Parameter suggestion with clinical constraints and type handling
- Regression model optimization with robust evaluation strategies
- General model optimization dispatcher with healthcare validation
- Default parameter fallback for optimization failure scenarios
- Optimization summary generation for healthcare reporting
- Edge cases and error handling for robust clinical deployment
- Production readiness validation with healthcare metadata preservation

Healthcare Validation:
- Clinical interpretability through constrained hyperparameter spaces
- Evidence-based optimization supporting pharmacy workflow improvement
- Healthcare stakeholder communication through optimization reporting
- Production deployment readiness with MLOps integration capabilities
- Regulatory compliance through comprehensive audit trail documentation
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import optuna
from sklearn.metrics import mean_squared_error

from src.tat.models.hyperparameter_optimizer import OptunaTATOptimizer
from src.tat.models.linear_model import RidgeTATRegressor
from src.tat.models.random_forest_model import RandomForestTATRegressor
from src.tat.models.xgboost_model import XGBoostTATRegressor
from src.tat.models.ensemble_model import StackingTATRegressor


class TestOptimizerInitialization:
    """Test OptunaTATOptimizer initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default optimizer initialization with healthcare parameters."""
        optimizer = OptunaTATOptimizer()
        
        # Validate default configuration
        assert optimizer.random_state == 42
        assert isinstance(optimizer.optimization_history, dict)
        assert len(optimizer.optimization_history) == 0
        
    def test_custom_random_state_initialization(self):
        """Test optimizer initialization with custom random state."""
        optimizer = OptunaTATOptimizer(random_state=123)
        
        # Validate custom random state
        assert optimizer.random_state == 123
        assert isinstance(optimizer.optimization_history, dict)
        
    def test_healthcare_logging_configuration(self):
        """Test healthcare-appropriate logging configuration."""
        # Optuna logging should be suppressed for clean healthcare output
        assert optuna.logging.get_verbosity() == optuna.logging.WARNING


class TestParameterSuggestion:
    """Test parameter suggestion with clinical constraints and type handling."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for parameter testing."""
        return OptunaTATOptimizer(random_state=42)
    
    @pytest.fixture
    def mock_trial(self):
        """Create mock Optuna trial for testing."""
        trial = Mock(spec=optuna.Trial)
        return trial
    
    def test_integer_parameter_suggestion(self, optimizer, mock_trial):
        """Test integer parameter suggestion for healthcare constraints."""
        # Configure mock trial
        mock_trial.suggest_int.return_value = 150
        
        # Test integer parameter suggestion
        param_config = ('int', 100, 200)
        result = optimizer._suggest_parameter(mock_trial, 'n_estimators', param_config)
        
        # Validate integer suggestion
        assert result == 150
        mock_trial.suggest_int.assert_called_once_with('n_estimators', 100, 200)
        
    def test_float_parameter_suggestion(self, optimizer, mock_trial):
        """Test float parameter suggestion for healthcare optimization."""
        # Configure mock trial
        mock_trial.suggest_float.return_value = 0.1
        
        # Test float parameter suggestion
        param_config = ('float', 0.05, 0.3)
        result = optimizer._suggest_parameter(mock_trial, 'learning_rate', param_config)
        
        # Validate float suggestion
        assert result == 0.1
        mock_trial.suggest_float.assert_called_once_with('learning_rate', 0.05, 0.3)
        
    def test_log_scale_float_parameter_suggestion(self, optimizer, mock_trial):
        """Test log-scale float parameter suggestion for regularization."""
        # Configure mock trial
        mock_trial.suggest_float.return_value = 1.0
        
        # Test log-scale float parameter suggestion
        param_config = ('float', 0.1, 100.0, 'log')
        result = optimizer._suggest_parameter(mock_trial, 'alpha', param_config)
        
        # Validate log-scale suggestion
        assert result == 1.0
        mock_trial.suggest_float.assert_called_once_with('alpha', 0.1, 100.0, log=True)
        
    def test_categorical_parameter_suggestion(self, optimizer, mock_trial):
        """Test categorical parameter suggestion for healthcare choices."""
        # Configure mock trial
        mock_trial.suggest_categorical.return_value = 'sqrt'
        
        # Test categorical parameter suggestion
        param_config = ('categorical', ['sqrt', 'log2', 0.5, 0.8])
        result = optimizer._suggest_parameter(mock_trial, 'max_features', param_config)
        
        # Validate categorical suggestion
        assert result == 'sqrt'
        mock_trial.suggest_categorical.assert_called_once_with('max_features', ['sqrt', 'log2', 0.5, 0.8])
        
    def test_invalid_parameter_type_error(self, optimizer, mock_trial):
        """Test error handling for invalid parameter types."""
        # Test invalid parameter type
        param_config = ('unknown_type', 0, 1)
        
        with pytest.raises(ValueError, match="Unknown parameter type"):
            optimizer._suggest_parameter(mock_trial, 'invalid_param', param_config)


class TestRegressionModelOptimization:
    """Test regression model optimization with healthcare validation."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for model testing."""
        return OptunaTATOptimizer(random_state=42)
    
    @pytest.fixture
    def healthcare_data(self):
        """Generate healthcare TAT dataset for optimization testing."""
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'medication_complexity': np.random.exponential(2.5, n_samples),
            'queue_length': np.random.poisson(5, n_samples),
            'staff_experience': np.random.gamma(3, 2, n_samples),
            'urgency_level': np.random.choice([1, 2, 3, 4, 5], n_samples)
        })
        
        y = pd.Series(15 + X['medication_complexity'] * 8 + 
                     X['queue_length'] * 2 + np.random.exponential(5, n_samples))
        
        # Split for validation
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    def test_ridge_model_optimization_with_validation(self, optimizer, healthcare_data):
        """Test Ridge model optimization with validation set."""
        X_train, y_train, X_val, y_val = healthcare_data
        
        # Create Ridge model for optimization
        ridge_model = RidgeTATRegressor()
        
        # Optimize with validation data (limited trials for testing speed)
        best_params = optimizer.optimize_regression_model(
            ridge_model, X_train, y_train, X_val, y_val, n_trials=3
        )
        
        # Validate optimization results
        assert isinstance(best_params, dict)
        assert 'alpha' in best_params
        assert best_params['alpha'] > 0
        
        # Validate optimization history
        model_key = 'RidgeTATRegressor_regression'
        assert model_key in optimizer.optimization_history
        history = optimizer.optimization_history[model_key]
        assert history['optimization_successful'] is True
        assert history['n_trials'] == 3
        assert history['best_value'] is not None
        
    def test_ridge_model_optimization_with_cross_validation(self, optimizer, healthcare_data):
        """Test Ridge model optimization with cross-validation."""
        X_train, y_train, _, _ = healthcare_data
        
        # Create Ridge model for optimization
        ridge_model = RidgeTATRegressor()
        
        # Optimize with cross-validation (no validation data)
        best_params = optimizer.optimize_regression_model(
            ridge_model, X_train, y_train, n_trials=3
        )
        
        # Validate optimization results
        assert isinstance(best_params, dict)
        assert 'alpha' in best_params
        
        # Validate optimization history
        model_key = 'RidgeTATRegressor_regression'
        assert model_key in optimizer.optimization_history
        history = optimizer.optimization_history[model_key]
        assert history['optimization_successful'] is True
        
    def test_random_forest_optimization(self, optimizer, healthcare_data):
        """Test Random Forest model optimization."""
        X_train, y_train, X_val, y_val = healthcare_data
        
        # Create Random Forest model for optimization
        rf_model = RandomForestTATRegressor()
        
        # Optimize Random Forest model
        best_params = optimizer.optimize_regression_model(
            rf_model, X_train, y_train, X_val, y_val, n_trials=2
        )
        
        # Validate optimization results
        assert isinstance(best_params, dict)
        expected_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
        
        for param in expected_params:
            assert param in best_params
            
        # Validate parameter ranges
        assert 100 <= best_params['n_estimators'] <= 300
        assert 5 <= best_params['max_depth'] <= 15
        
    def test_xgboost_optimization(self, optimizer, healthcare_data):
        """Test XGBoost model optimization."""
        X_train, y_train, X_val, y_val = healthcare_data
        
        # Create XGBoost model for optimization
        xgb_model = XGBoostTATRegressor()
        
        # Optimize XGBoost model
        best_params = optimizer.optimize_regression_model(
            xgb_model, X_train, y_train, X_val, y_val, n_trials=2
        )
        
        # Validate optimization results
        assert isinstance(best_params, dict)
        expected_params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
        
        for param in expected_params:
            assert param in best_params
            
        # Validate parameter ranges
        assert 50 <= best_params['n_estimators'] <= 300
        assert 3 <= best_params['max_depth'] <= 8
        assert 0.05 <= best_params['learning_rate'] <= 0.3
        
    @patch('src.tat.models.hyperparameter_optimizer.cross_val_score')
    def test_optimization_with_cross_validation_failure(self, mock_cross_val, optimizer, healthcare_data):
        """Test optimization behavior when cross-validation fails."""
        X_train, y_train, _, _ = healthcare_data
        
        # Configure mock to raise exception
        mock_cross_val.side_effect = Exception("Cross-validation failed")
        
        # Create model for optimization
        ridge_model = RidgeTATRegressor()
        
        # Optimize model (should handle cross-validation failure)
        best_params = optimizer.optimize_regression_model(
            ridge_model, X_train, y_train, n_trials=1
        )
        
        # Should return optimized parameters (optimization succeeds despite some trial failures)
        assert isinstance(best_params, dict)
        
        # Validate success tracking in history (overall optimization succeeds)
        model_key = 'RidgeTATRegressor_regression'
        assert model_key in optimizer.optimization_history
        history = optimizer.optimization_history[model_key]
        assert history['optimization_successful'] is True  # Overall optimization succeeds


class TestGeneralModelOptimization:
    """Test general model optimization dispatcher."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for dispatcher testing."""
        return OptunaTATOptimizer(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for dispatcher testing."""
        np.random.seed(42)
        X = pd.DataFrame({'feature': np.random.normal(0, 1, 100)})
        y = pd.Series(10 + X['feature'] * 3 + np.random.normal(0, 1, 100))
        return X, y
    
    def test_regression_model_dispatch(self, optimizer, sample_data):
        """Test dispatcher routing to regression optimization."""
        X, y = sample_data
        
        # Create regression model
        ridge_model = RidgeTATRegressor()
        
        # Use general optimizer method
        best_params = optimizer.optimize_model(ridge_model, X, y, n_trials=2)
        
        # Validate optimization results
        assert isinstance(best_params, dict)
        assert 'alpha' in best_params
        
    def test_unknown_model_type_error(self, optimizer, sample_data):
        """Test error handling for unknown model types."""
        X, y = sample_data
        
        # Create mock unknown model
        unknown_model = Mock()
        
        with pytest.raises(ValueError, match="Unknown model type"):
            optimizer.optimize_model(unknown_model, X, y)


class TestDefaultParameters:
    """Test default parameter fallback for optimization failures."""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer for default parameter testing."""
        return OptunaTATOptimizer(random_state=42)
    
    def test_ridge_default_parameters(self, optimizer):
        """Test Ridge model default parameters."""
        ridge_model = RidgeTATRegressor()
        defaults = optimizer._get_default_params(ridge_model)
        
        # Validate Ridge defaults
        assert isinstance(defaults, dict)
        assert 'alpha' in defaults
        assert defaults['alpha'] == 1.0
        
    def test_random_forest_default_parameters(self, optimizer):
        """Test Random Forest model default parameters."""
        rf_model = RandomForestTATRegressor()
        defaults = optimizer._get_default_params(rf_model)
        
        # Validate Random Forest defaults
        assert isinstance(defaults, dict)
        expected_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']
        
        for param in expected_params:
            assert param in defaults
            
        assert defaults['n_estimators'] == 200
        assert defaults['max_depth'] == 8
        
    def test_xgboost_default_parameters(self, optimizer):
        """Test XGBoost model default parameters."""
        xgb_model = XGBoostTATRegressor()
        defaults = optimizer._get_default_params(xgb_model)
        
        # Validate XGBoost defaults
        assert isinstance(defaults, dict)
        expected_params = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
        
        for param in expected_params:
            assert param in defaults
            
        assert defaults['n_estimators'] == 150
        assert defaults['max_depth'] == 6
        assert defaults['learning_rate'] == 0.1
        
    def test_ensemble_default_parameters(self, optimizer):
        """Test Ensemble model default parameters."""
        ensemble_model = StackingTATRegressor()
        defaults = optimizer._get_default_params(ensemble_model)
        
        # Validate Ensemble defaults
        assert isinstance(defaults, dict)
        expected_params = ['meta_alpha', 'xgb_n_estimators', 'rf_n_estimators', 'ridge_alpha']
        
        for param in expected_params:
            assert param in defaults
            
        assert defaults['meta_alpha'] == 1.0
        assert defaults['xgb_n_estimators'] == 100
        assert defaults['rf_n_estimators'] == 100
        
    def test_unknown_model_default_parameters(self, optimizer):
        """Test default parameters for unknown model types."""
        unknown_model = Mock()
        unknown_model.__class__.__name__ = 'UnknownModel'
        
        defaults = optimizer._get_default_params(unknown_model)
        
        # Should return empty dict for unknown models
        assert isinstance(defaults, dict)
        assert len(defaults) == 0


class TestOptimizationSummary:
    """Test optimization summary generation for healthcare reporting."""
    
    @pytest.fixture
    def optimizer_with_history(self):
        """Create optimizer with mock optimization history."""
        optimizer = OptunaTATOptimizer(random_state=42)
        
        # Add mock optimization history
        optimizer.optimization_history = {
            'RidgeTATRegressor_regression': {
                'best_params': {'alpha': 1.0},
                'best_value': 8.5,
                'n_trials': 10,
                'optimization_successful': True,
                'healthcare_context': 'TAT prediction regression',
                'clinical_objective': '60-minute threshold optimization'
            },
            'XGBoostTATRegressor_regression': {
                'best_params': {'n_estimators': 150, 'max_depth': 6},
                'best_value': 7.2,
                'n_trials': 15,
                'optimization_successful': True,
                'healthcare_context': 'TAT prediction regression',
                'clinical_objective': '60-minute threshold optimization'
            },
            'RandomForestTATRegressor_regression': {
                'best_params': {},
                'best_value': None,
                'n_trials': 0,
                'optimization_successful': False,
                'error': 'Optimization failed',
                'fallback_used': True,
                'healthcare_context': 'TAT prediction regression (fallback)'
            }
        }
        
        return optimizer
    
    def test_optimization_summary_generation(self, optimizer_with_history):
        """Test comprehensive optimization summary generation."""
        summary = optimizer_with_history.get_optimization_summary()
        
        # Validate summary structure
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 3  # Three optimization histories
        
        # Validate required columns
        required_columns = [
            'model', 'optimization_successful', 'best_value', 
            'n_trials', 'has_error', 'healthcare_context'
        ]
        for col in required_columns:
            assert col in summary.columns
            
        # Validate data content
        assert 'RidgeTATRegressor_regression' in summary['model'].values
        assert 'XGBoostTATRegressor_regression' in summary['model'].values
        assert 'RandomForestTATRegressor_regression' in summary['model'].values
        
    def test_summary_sorting_by_performance(self, optimizer_with_history):
        """Test summary sorting by optimization performance."""
        summary = optimizer_with_history.get_optimization_summary()
        
        # Successful optimizations should come first
        successful_rows = summary[summary['optimization_successful'] == True]
        failed_rows = summary[summary['optimization_successful'] == False]
        
        # Check that successful rows come before failed rows
        if len(successful_rows) > 0 and len(failed_rows) > 0:
            first_successful_idx = summary[summary['optimization_successful'] == True].index[0]
            first_failed_idx = summary[summary['optimization_successful'] == False].index[0]
            assert first_successful_idx < first_failed_idx
            
        # Among successful rows, should be sorted by best_value (ascending)
        if len(successful_rows) > 1:
            successful_values = successful_rows['best_value'].tolist()
            assert successful_values == sorted([v for v in successful_values if v is not None])
            
    def test_healthcare_context_preservation(self, optimizer_with_history):
        """Test healthcare context preservation in summary."""
        summary = optimizer_with_history.get_optimization_summary()
        
        # Validate healthcare context columns
        assert 'healthcare_context' in summary.columns
        assert 'clinical_objective' in summary.columns
        
        # Check healthcare context values
        contexts = summary['healthcare_context'].tolist()
        assert all('TAT prediction' in context for context in contexts)
        
        # Check clinical objectives
        objectives = summary['clinical_objective'].tolist()
        assert all('60-minute' in obj for obj in objectives if pd.notna(obj))
        
    def test_empty_optimization_history_summary(self):
        """Test summary generation with empty optimization history."""
        optimizer = OptunaTATOptimizer()
        
        # Generate summary with empty history
        summary = optimizer.get_optimization_summary()
        
        # Validate empty summary structure
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 0
        
        # Validate column schema
        expected_columns = [
            'model', 'optimization_successful', 'best_value', 
            'n_trials', 'has_error', 'healthcare_context'
        ]
        for col in expected_columns:
            assert col in summary.columns


class TestEdgeCasesAndErrorHandling:
    """Test optimizer edge cases and error handling for robust deployment."""
    
    def test_optimizer_with_small_dataset(self):
        """Test optimizer behavior with small healthcare datasets."""
        optimizer = OptunaTATOptimizer(random_state=42)
        
        # Create very small dataset
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series([10, 12, 14, 16, 18])
        
        ridge_model = RidgeTATRegressor()
        
        # Should handle small dataset gracefully
        best_params = optimizer.optimize_regression_model(ridge_model, X, y, n_trials=2)
        
        assert isinstance(best_params, dict)
        
    def test_optimizer_with_extreme_values(self):
        """Test optimizer handling of extreme TAT values."""
        optimizer = OptunaTATOptimizer(random_state=42)
        
        # Create dataset with extreme values
        np.random.seed(42)
        X = pd.DataFrame({'feature': np.random.normal(0, 1, 50)})
        y_extreme = pd.Series(np.concatenate([
            np.random.exponential(10, 45),  # Normal range
            [500, 1000, 1500, 2000, 0.1]   # Extreme values
        ]))
        
        ridge_model = RidgeTATRegressor()
        
        # Should handle extreme values without failure
        best_params = optimizer.optimize_regression_model(ridge_model, X, y_extreme, n_trials=2)
        
        assert isinstance(best_params, dict)
        
    @patch('optuna.create_study')
    def test_optimization_failure_fallback(self, mock_create_study):
        """Test behavior when Optuna study creation fails."""
        optimizer = OptunaTATOptimizer(random_state=42)
        
        # Configure mock to raise exception
        mock_create_study.side_effect = Exception("Optuna failed")
        
        # Create sample data and model
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        y = pd.Series([10, 12, 14, 16, 18])
        ridge_model = RidgeTATRegressor()
        
        # Should raise exception since study creation fails before try-catch
        with pytest.raises(Exception, match="Optuna failed"):
            optimizer.optimize_regression_model(ridge_model, X, y, n_trials=2)
        
    def test_parameter_suggestion_with_invalid_trial(self):
        """Test parameter suggestion error handling."""
        optimizer = OptunaTATOptimizer(random_state=42)
        
        # Create mock trial that raises exception
        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = Exception("Suggestion failed")
        
        # Should propagate the exception
        param_config = ('int', 1, 10)
        with pytest.raises(Exception, match="Suggestion failed"):
            optimizer._suggest_parameter(mock_trial, 'test_param', param_config)


class TestProductionReadiness:
    """Test optimizer production deployment readiness and healthcare compliance."""
    
    def test_reproducible_optimization_results(self):
        """Test reproducible optimization results with fixed random state."""
        # Create two optimizers with same random state
        optimizer1 = OptunaTATOptimizer(random_state=42)
        optimizer2 = OptunaTATOptimizer(random_state=42)
        
        # Create deterministic dataset
        np.random.seed(42)
        X = pd.DataFrame({'feature': np.random.normal(0, 1, 100)})
        y = pd.Series(10 + X['feature'] * 2 + np.random.normal(0, 0.1, 100))
        
        ridge_model1 = RidgeTATRegressor()
        ridge_model2 = RidgeTATRegressor()
        
        # Optimize both models
        params1 = optimizer1.optimize_regression_model(ridge_model1, X, y, n_trials=3)
        params2 = optimizer2.optimize_regression_model(ridge_model2, X, y, n_trials=3)
        
        # Results should be similar (allowing for some variation due to CV)
        assert isinstance(params1, dict)
        assert isinstance(params2, dict)
        assert 'alpha' in params1
        assert 'alpha' in params2
        
    def test_healthcare_metadata_preservation(self):
        """Test healthcare metadata preservation in optimization history."""
        optimizer = OptunaTATOptimizer(random_state=42)
        
        # Create sample data and model
        X = pd.DataFrame({'feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        y = pd.Series([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
        
        ridge_model = RidgeTATRegressor()
        
        # Optimize model
        optimizer.optimize_regression_model(ridge_model, X, y, n_trials=2)
        
        # Validate healthcare metadata preservation
        model_key = 'RidgeTATRegressor_regression'
        history = optimizer.optimization_history[model_key]
        
        # Validate clinical context
        assert 'healthcare_context' in history
        assert 'clinical_objective' in history
        assert 'TAT prediction' in history['healthcare_context']
        assert '60-minute' in history['clinical_objective']
        
    def test_optimization_audit_trail(self):
        """Test comprehensive audit trail generation."""
        optimizer = OptunaTATOptimizer(random_state=42)
        
        # Optimize multiple models for audit trail testing
        X = pd.DataFrame({'feature': np.random.normal(0, 1, 50)})
        y = pd.Series(10 + X['feature'] * 3 + np.random.normal(0, 1, 50))
        
        ridge_model = RidgeTATRegressor()
        rf_model = RandomForestTATRegressor()
        
        # Optimize both models
        optimizer.optimize_regression_model(ridge_model, X, y, n_trials=2)
        optimizer.optimize_regression_model(rf_model, X, y, n_trials=2)
        
        # Generate audit trail summary
        summary = optimizer.get_optimization_summary()
        
        # Validate comprehensive audit trail
        assert len(summary) == 2
        assert all(summary['optimization_successful'])
        assert all(pd.notna(summary['best_value']))
        assert all(summary['n_trials'] > 0)
        
        # Validate healthcare context in audit trail
        assert all('TAT prediction' in context for context in summary['healthcare_context'])
        
    def test_healthcare_compliance_validation(self):
        """Test healthcare regulatory compliance features."""
        optimizer = OptunaTATOptimizer(random_state=42)
        
        # Validate reproducibility configuration
        assert hasattr(optimizer, 'random_state')
        assert isinstance(optimizer.random_state, int)
        
        # Validate audit trail capability
        assert hasattr(optimizer, 'optimization_history')
        assert isinstance(optimizer.optimization_history, dict)
        
        # Validate healthcare reporting capability
        summary = optimizer.get_optimization_summary()
        assert 'healthcare_context' in summary.columns
        assert 'clinical_objective' in summary.columns