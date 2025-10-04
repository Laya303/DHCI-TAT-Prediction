"""
Test suite for base model classes.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib
from unittest.mock import patch, MagicMock

from tat.models.base import BaseTATModel, BaseRegressionTATModel


class TestBaseTATModelInterface:
    """Validate BaseTATModel abstract interface and healthcare compliance."""
    
    def test_abstract_base_class_instantiation(self):
        """Verify BaseTATModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTATModel()
    
    def test_concrete_implementation_initialization(self):
        """Validate concrete model implementation initialization."""
        
        class ConcreteTATModel(BaseTATModel):
            def fit(self, X, y):
                self.is_fitted = True
                return self
            
            def predict(self, X):
                return np.random.rand(len(X))
            
            def get_hyperparameter_space(self):
                return {'param1': [1, 2, 3]}
        
        model = ConcreteTATModel(random_state=123)
        
        # Validate initialization attributes
        assert model.random_state == 123
        assert model.model is None
        assert model.is_fitted is False
        assert model.metadata['model_type'] == 'ConcreteTATModel'
        assert model.metadata['healthcare_context'] == 'TAT Prediction'
        assert model.metadata['clinical_objective'] == '60-minute threshold optimization'
        assert model.metadata['random_state'] == 123
    
    def test_default_random_state_initialization(self):
        """Verify default random state configuration."""
        
        class ConcreteTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        model = ConcreteTATModel()
        assert model.random_state == 42
        assert model.metadata['random_state'] == 42


class TestBaseTATModelAbstractMethods:
    """Validate abstract method enforcement and interface requirements."""
    
    def test_fit_method_required(self):
        """Verify fit method must be implemented in concrete classes."""
        
        class IncompleteTATModel(BaseTATModel):
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        with pytest.raises(TypeError):
            IncompleteTATModel()
    
    def test_predict_method_required(self):
        """Verify predict method must be implemented in concrete classes."""
        
        class IncompleteTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def get_hyperparameter_space(self): return {}
        
        with pytest.raises(TypeError):
            IncompleteTATModel()
    
    def test_hyperparameter_space_method_required(self):
        """Verify get_hyperparameter_space method must be implemented."""
        
        class IncompleteTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
        
        with pytest.raises(TypeError):
            IncompleteTATModel()


class TestHealthcareMetricsEvaluation:
    """Validate healthcare-specific performance evaluation metrics."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample TAT predictions for metric validation."""
        return {
            'y_true': pd.Series([45, 65, 30, 75, 20, 90, 55, 85]),
            'y_pred': np.array([50, 70, 25, 80, 15, 85, 60, 90])
        }
    
    def test_healthcare_metrics_calculation(self, sample_predictions):
        """Verify comprehensive healthcare metrics computation."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        model = TestTATModel()
        y_true = sample_predictions['y_true']
        y_pred = sample_predictions['y_pred']
        
        metrics = model.evaluate_healthcare_metrics(y_true, y_pred)
        
        # Validate metric presence and types
        expected_metrics = [
            'RMSE', 'MAE', 'MedianAE', 'within_10min_pct', 
            'within_30min_pct', 'threshold_60min_accuracy', 'healthcare_score'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
    
    def test_threshold_accuracy_calculation(self):
        """Validate 60-minute threshold accuracy computation."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        model = TestTATModel()
        
        # Perfect threshold classification
        y_true = pd.Series([45, 75, 30, 85])  # 2 below, 2 above 60
        y_pred = np.array([50, 70, 35, 90])   # 2 below, 2 above 60
        
        metrics = model.evaluate_healthcare_metrics(y_true, y_pred)
        assert metrics['threshold_60min_accuracy'] == 100.0
        
        # Imperfect threshold classification
        y_true = pd.Series([45, 75, 30, 85])  # 2 below, 2 above 60
        y_pred = np.array([70, 50, 35, 90])   # 1 below, 3 above 60 (2 misclassified)
        
        metrics = model.evaluate_healthcare_metrics(y_true, y_pred)
        assert metrics['threshold_60min_accuracy'] == 50.0
    
    def test_accuracy_bands_calculation(self):
        """Validate clinical accuracy band percentage calculations."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        model = TestTATModel()
        
        # Perfect predictions within 10 minutes
        y_true = pd.Series([50, 60, 70, 80])
        y_pred = np.array([55, 65, 75, 85])  # All within 5 minutes
        
        metrics = model.evaluate_healthcare_metrics(y_true, y_pred)
        assert metrics['within_10min_pct'] == 100.0
        assert metrics['within_30min_pct'] == 100.0
        
        # Mixed accuracy predictions
        y_true = pd.Series([50, 60, 70, 80])
        y_pred = np.array([65, 90, 75, 85])  # 15, 30, 5, 5 minute errors
        
        metrics = model.evaluate_healthcare_metrics(y_true, y_pred)
        assert metrics['within_10min_pct'] == 50.0  # 2/4 within 10 minutes
        assert metrics['within_30min_pct'] == 100.0  # 4/4 within 30 minutes (15, 30, 5, 5 all <= 30)
    
    def test_healthcare_score_calculation(self):
        """Validate healthcare composite score computation."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        model = TestTATModel()
        
        y_true = pd.Series([50, 70, 40, 80])
        y_pred = np.array([55, 75, 35, 85])
        
        metrics = model.evaluate_healthcare_metrics(y_true, y_pred)
        
        # Healthcare score should be average of within_30min_pct and threshold_60min_accuracy
        expected_score = (metrics['within_30min_pct'] + metrics['threshold_60min_accuracy']) / 2
        assert abs(metrics['healthcare_score'] - expected_score) < 0.001
    
    def test_mismatched_array_lengths_error(self):
        """Validate error handling for mismatched prediction arrays."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        model = TestTATModel()
        
        y_true = pd.Series([50, 60, 70])
        y_pred = np.array([55, 65])  # Different length
        
        with pytest.raises(ValueError, match="must have same length"):
            model.evaluate_healthcare_metrics(y_true, y_pred)


class TestModelLifecycleManagement:
    """Validate model save/load functionality and metadata preservation."""
    
    @pytest.fixture
    def trained_model(self):
        """Create trained model for lifecycle testing."""
        
        class TestTATModel(BaseTATModel):
            def __init__(self, random_state=42):
                super().__init__(random_state)
                self.model = "mock_model_state"  # Simple string instead of MagicMock
            
            def fit(self, X, y):
                self.is_fitted = True
                return self
            
            def predict(self, X):
                return np.array([60] * len(X))
            
            def get_hyperparameter_space(self):
                return {'param1': [1, 2, 3]}
        
        model = TestTATModel(random_state=123)
        model.fit(pd.DataFrame({'feature1': [1, 2, 3]}), pd.Series([60, 70, 50]))
        return model
    
    def test_save_model_functionality(self, trained_model):
        """Validate model saving with healthcare metadata."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.joblib"
            
            # Save trained model
            trained_model.save_model(save_path)
            
            # Verify file exists
            assert save_path.exists()
            
            # Verify saved content
            saved_data = joblib.load(save_path)
            
            assert 'model' in saved_data
            assert 'metadata' in saved_data
            assert 'is_fitted' in saved_data
            assert saved_data['is_fitted'] is True
            assert saved_data['metadata']['random_state'] == 123
    
    def test_save_unfitted_model_error(self):
        """Verify error handling for saving unfitted models."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        model = TestTATModel()  # Not fitted
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.joblib"
            
            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                model.save_model(save_path)
    
    def test_save_model_directory_creation(self, trained_model):
        """Validate automatic directory creation during model saving."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "new_directory" / "test_model.joblib"
            
            # Directory should not exist initially
            assert not save_path.parent.exists()
            
            # Save should create directory
            trained_model.save_model(save_path)
            
            assert save_path.parent.exists()
            assert save_path.exists()
    
    def test_load_model_functionality(self):
        """Validate model loading with metadata reconstruction."""
        
        # Create a concrete test class for loading
        class ConcreteTestModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "test_model.joblib"
            
            # Create and save a real model
            original_model = ConcreteTestModel(random_state=123)
            original_model.model = "simple_model_state"
            original_model.is_fitted = True
            original_model.save_model(save_path)
            
            # Load model using the concrete class
            loaded_model = ConcreteTestModel.load_model(save_path)
            
            # Verify model reconstruction
            assert loaded_model.is_fitted is True
            assert loaded_model.random_state == 123
            assert loaded_model.model == "simple_model_state"
            assert loaded_model.metadata['model_type'] == 'ConcreteTestModel'
    
    def test_load_nonexistent_model_error(self):
        """Validate error handling for loading nonexistent models."""
        
        nonexistent_path = Path("/nonexistent/model.joblib")
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            BaseTATModel.load_model(nonexistent_path)


class TestBaseRegressionTATModel:
    """Validate regression-specific base class functionality."""
    
    def test_regression_model_initialization(self):
        """Verify regression model initialization with target transformation."""
        
        class TestRegressionModel(BaseRegressionTATModel):
            def fit(self, X, y): return self
            def get_hyperparameter_space(self): return {}
        
        model = TestRegressionModel(random_state=456)
        
        # Validate inherited base initialization
        assert model.random_state == 456
        assert model.metadata['model_type'] == 'TestRegressionModel'
        
        # Validate regression-specific initialization
        assert model.target_transform == 'log1p'
    
    def test_target_transformation_log1p(self):
        """Validate log1p target transformation for skewed data."""
        
        class TestRegressionModel(BaseRegressionTATModel):
            def fit(self, X, y): return self
            def get_hyperparameter_space(self): return {}
        
        model = TestRegressionModel()
        
        # Test log1p transformation
        y_original = pd.Series([0, 1, 10, 100])
        y_transformed = model._transform_target(y_original)
        
        # Verify log1p transformation
        expected = np.log1p(y_original)
        pd.testing.assert_series_equal(y_transformed, expected)
    
    def test_target_no_transformation(self):
        """Validate behavior when no transformation is specified."""
        
        class TestRegressionModel(BaseRegressionTATModel):
            def fit(self, X, y): return self
            def get_hyperparameter_space(self): return {}
        
        model = TestRegressionModel()
        model.target_transform = None  # No transformation
        
        y_original = pd.Series([10, 20, 30])
        y_transformed = model._transform_target(y_original)
        
        # Should return original values
        pd.testing.assert_series_equal(y_transformed, y_original)
    
    def test_inverse_target_transformation(self):
        """Validate inverse transformation returning to original scale."""
        
        class TestRegressionModel(BaseRegressionTATModel):
            def fit(self, X, y): return self
            def get_hyperparameter_space(self): return {}
        
        model = TestRegressionModel()
        
        # Test inverse log1p transformation (expm1)
        y_transformed = np.array([0, 0.693, 2.303, 4.605])  # log1p([0,1,10,100])
        y_inverse = model._inverse_transform_target(y_transformed)
        
        # Verify expm1 inverse transformation
        expected = np.expm1(y_transformed)
        np.testing.assert_array_almost_equal(y_inverse, expected, decimal=3)
    
    def test_inverse_no_transformation(self):
        """Validate inverse transformation when no transform is applied."""
        
        class TestRegressionModel(BaseRegressionTATModel):
            def fit(self, X, y): return self
            def get_hyperparameter_space(self): return {}
        
        model = TestRegressionModel()
        model.target_transform = None  # No transformation
        
        y_transformed = np.array([10, 20, 30])
        y_inverse = model._inverse_transform_target(y_transformed)
        
        # Should return original values
        np.testing.assert_array_equal(y_inverse, y_transformed)
    
    def test_regression_predict_with_transformation(self):
        """Validate prediction with target transformation handling."""
        
        class TestRegressionModel(BaseRegressionTATModel):
            def __init__(self, random_state=42):
                super().__init__(random_state)
                self.model = MagicMock()
                
            def fit(self, X, y):
                self.is_fitted = True
                return self
                
            def get_hyperparameter_space(self):
                return {}
        
        model = TestRegressionModel()
        model.fit(pd.DataFrame({'feature1': [1, 2]}), pd.Series([60, 70]))
        
        # Mock transformed predictions
        model.model.predict.return_value = np.array([4.0, 4.3])  # log1p scale
        
        X_test = pd.DataFrame({'feature1': [3, 4]})
        predictions = model.predict(X_test)
        
        # Verify model called and inverse transformation applied
        model.model.predict.assert_called_once_with(X_test)
        
        # Predictions should be in original scale (expm1 of mock values)
        expected_predictions = np.expm1(np.array([4.0, 4.3]))
        np.testing.assert_array_almost_equal(predictions, expected_predictions, decimal=3)
    
    def test_regression_predict_unfitted_error(self):
        """Validate error handling for predicting with unfitted model."""
        
        class TestRegressionModel(BaseRegressionTATModel):
            def fit(self, X, y): return self
            def get_hyperparameter_space(self): return {}
        
        model = TestRegressionModel()  # Not fitted
        
        X_test = pd.DataFrame({'feature1': [1, 2]})
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict(X_test)


class TestEdgeCasesAndErrorHandling:
    """Validate edge cases and error handling in base model classes."""
    
    def test_empty_dataframe_metrics_evaluation(self):
        """Validate behavior with empty prediction arrays."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([])
            def get_hyperparameter_space(self): return {}
        
        model = TestTATModel()
        
        y_true = pd.Series([])
        y_pred = np.array([])
        
        # Empty arrays should raise appropriate error from sklearn
        with pytest.raises(ValueError, match="Found array with 0 sample"):
            model.evaluate_healthcare_metrics(y_true, y_pred)
    
    def test_extreme_tat_values_handling(self):
        """Validate handling of extreme TAT values in metrics."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        model = TestTATModel()
        
        # Test with extreme values
        y_true = pd.Series([0, 1440, 5000])  # 0 minutes, 24 hours, very long
        y_pred = np.array([10, 1400, 4800])
        
        # Should compute metrics without errors
        metrics = model.evaluate_healthcare_metrics(y_true, y_pred)
        
        assert isinstance(metrics['RMSE'], (int, float))
        assert not np.isnan(metrics['RMSE'])
        assert metrics['RMSE'] >= 0
    
    def test_target_transformation_edge_cases(self):
        """Validate target transformation with edge case values."""
        
        class TestRegressionModel(BaseRegressionTATModel):
            def fit(self, X, y): return self
            def get_hyperparameter_space(self): return {}
        
        model = TestRegressionModel()
        
        # Test with zeros and small values
        y_with_zeros = pd.Series([0, 0.1, 1, 10])
        y_transformed = model._transform_target(y_with_zeros)
        
        # Log1p should handle zeros correctly
        expected = np.log1p(y_with_zeros)
        pd.testing.assert_series_equal(y_transformed, expected)
        
        # Verify inverse transformation
        y_inverse = model._inverse_transform_target(y_transformed.values)
        np.testing.assert_array_almost_equal(y_inverse, y_with_zeros.values, decimal=6)


class TestProductionReadinessValidation:
    """Validate production deployment readiness and MLOps integration."""
    
    def test_metadata_preservation_across_lifecycle(self):
        """Validate metadata preservation through save/load cycle."""
        
        class TestTATModel(BaseTATModel):
            def __init__(self, random_state=42):
                super().__init__(random_state)
                self.model = "mock_model_state"
                
            def fit(self, X, y):
                self.is_fitted = True
                # Add custom metadata during training
                self.metadata['training_features'] = list(X.columns)
                self.metadata['training_samples'] = len(X)
                return self
                
            def predict(self, X):
                return np.array([60] * len(X))
                
            def get_hyperparameter_space(self):
                return {'param1': [1, 2, 3]}
        
        # Train model with metadata
        model = TestTATModel(random_state=789)
        X_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_train = pd.Series([60, 70, 50])
        
        model.fit(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "model_with_metadata.joblib"
            
            # Save and verify custom metadata
            model.save_model(save_path)
            
            # Verify metadata in saved file
            saved_data = joblib.load(save_path)
            metadata = saved_data['metadata']
            
            assert metadata['training_features'] == ['feature1', 'feature2']
            assert metadata['training_samples'] == 3
            assert metadata['random_state'] == 789
            assert metadata['model_type'] == 'TestTATModel'
    
    def test_consistent_random_state_behavior(self):
        """Validate consistent behavior across multiple instantiations."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.random.rand(len(X))
            def get_hyperparameter_space(self): return {}
        
        # Multiple models with same random state should have consistent metadata
        model1 = TestTATModel(random_state=999)
        model2 = TestTATModel(random_state=999)
        
        assert model1.random_state == model2.random_state
        assert model1.metadata['random_state'] == model2.metadata['random_state']
        assert model1.metadata['model_type'] == model2.metadata['model_type']
    
    def test_healthcare_context_validation(self):
        """Validate healthcare context preservation in model metadata."""
        
        class TestTATModel(BaseTATModel):
            def fit(self, X, y): return self
            def predict(self, X): return np.array([60])
            def get_hyperparameter_space(self): return {}
        
        model = TestTATModel()
        
        # Verify healthcare-specific metadata
        assert model.metadata['healthcare_context'] == 'TAT Prediction'
        assert model.metadata['clinical_objective'] == '60-minute threshold optimization'
        
        # Metadata should be suitable for healthcare audit trails
        required_fields = ['model_type', 'healthcare_context', 'clinical_objective', 'random_state']
        for field in required_fields:
            assert field in model.metadata
            assert model.metadata[field] is not None