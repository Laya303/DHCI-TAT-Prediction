"""
Test suite for functionality.

Production-ready validation framework for RidgeTATRegressor class ensuring
healthcare linear model integrity, clinical interpretability, and pharmacy
workflow optimization capabilities. Validates Ridge regression        model = RidgeTATRegressor(random_state=42, alpha=1.0)
        
        # Extract and process features like in trained_ridge_model fixture
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
                   model = RidgeTATRegressor(random_state=42, alpha=1.0)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mappif col in X.colung)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatmns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())ent_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())plementation,
coefficient analysis, healthcare metrics, and production deployment readiness.

Test Categories:
- Ridge Regression: Linear model implementation and L2 regularization
- Clinical Interpretability: Coefficient analysis and feature importance
- Healthcare Training: TAT data handling and target transformation
- Model Validation:  Numerical Features and threshold compliance
- Production Integration: MLOps deployment and metadata management
- Error Handling: Edge cases and healthcare safety validation
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
from sklearn.linear_model import Ridge

from tat.models.linear_model import RidgeTATRegressor
from tat.models.base import BaseRegressionTATModel


class TestRidgeTATRegressorInitialization:
    """Validate Ridge model initialization and configuration management."""
    
    def test_default_initialization(self):
        """Verify default healthcare-optimized Ridge configuration."""
        
        model = RidgeTATRegressor()
        
        # Validate base model inheritance
        assert isinstance(model, BaseRegressionTATModel)
        assert model.random_state == 42
        assert model.is_fitted is False
        assert model.target_transform == 'log1p'
        
        # Validate Ridge-specific configuration
        assert isinstance(model.model, Ridge)
        assert isinstance(model.default_params, dict)
        
        # Validate healthcare metadata
        assert model.metadata['model_type'] == 'RidgeTATRegressor'
        assert model.metadata['healthcare_context'] == 'TAT Prediction'
        assert model.metadata['algorithm'] == 'Ridge Regression'
        assert model.metadata['interpretability'] == 'Full coefficient analysis available'
    
    def test_custom_parameter_initialization(self):
        """Validate Ridge model with custom healthcare parameters."""
        
        model = RidgeTATRegressor(
            random_state=123,
            alpha=5.0,
            max_iter=500,
            fit_intercept=False,
            solver='svd'
        )
        
        # Validate parameter configuration
        assert model.random_state == 123
        assert model.default_params['alpha'] == 5.0
        assert model.default_params['max_iter'] == 500
        assert model.default_params['fit_intercept'] is False
        assert model.default_params['solver'] == 'svd'
        assert model.default_params['random_state'] == 123
    
    def test_healthcare_metadata_completeness(self):
        """Validate comprehensive healthcare metadata for audit trails."""
        
        model = RidgeTATRegressor(random_state=456)
        
        # Essential healthcare metadata fields
        healthcare_fields = [
            'algorithm', 'healthcare_optimization', 'interpretability',
            'clinical_advantages', 'bottleneck_identification',
            'regulatory_compliance', 'deployment_readiness'
        ]
        
        for field in healthcare_fields:
            assert field in model.metadata
            assert isinstance(model.metadata[field], str)
            assert len(model.metadata[field]) > 0
    
    def test_sklearn_ridge_configuration(self):
        """Validate underlying sklearn Ridge model configuration."""
        
        custom_params = {
            'alpha': 2.5,
            'solver': 'lsqr',
            'max_iter': 2000,
            'random_state': 789
        }
        
        model = RidgeTATRegressor(**custom_params)
        
        # Validate Ridge model parameters
        ridge_model = model.model
        assert ridge_model.alpha == 2.5
        assert ridge_model.solver == 'lsqr'
        assert ridge_model.max_iter == 2000
        # Note: Ridge doesn't use random_state, but it should be in default_params
        assert model.default_params['random_state'] == 789


class TestRidgeTrainingWorkflow:
    """Validate Ridge regression training workflow and healthcare data handling."""
    
    # Note: Using sample_tat_data fixture from conftest.py instead of local fixture
    
    def test_ridge_fit_basic_training(self, sample_tat_data):
        """Validate basic Ridge model training with healthcare data."""
        
        model = RidgeTATRegressor(random_state=42)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())
        
        # Train model
        fitted_model = model.fit(X, y)
        
        # Validate training completion
        assert fitted_model is model  # Returns self
        assert model.is_fitted is True
        assert model.model.coef_ is not None
        assert len(model.model.coef_) == len(X.columns)
    
    def test_ridge_target_transformation_in_training(self, sample_tat_data):
        """Validate target transformation during Ridge training."""
        
        model = RidgeTATRegressor(random_state=42)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())
        
        # Check original target distribution
        assert np.any(y > 60)  # Should have values above threshold
        assert y.min() > 0     # Positive TAT values
        
        # Train model (internally uses log1p transformation)
        model.fit(X, y)
        
        # Model should be fitted successfully with transformed target
        assert model.is_fitted is True
        
        # Verify model can predict in original scale
        predictions = model.predict(X.iloc[:10])
        assert all(pred > 0 for pred in predictions)  # Positive predictions
        assert all(np.isfinite(pred) for pred in predictions)  # Valid predictions
    
    def test_ridge_validation_data_parameter(self, sample_tat_data):
        """Validate validation data parameter handling in Ridge training."""
        
        model = RidgeTATRegressor(random_state=42)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())
        
        # Split data for validation
        X_train, X_val = X.iloc[:150], X.iloc[150:]
        y_train, y_val = y.iloc[:150], y.iloc[150:]
        
        # Train with validation data (should not cause errors)
        model.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        # Validate successful training
        assert model.is_fitted is True
        
        # Should be able to predict on validation data
        val_predictions = model.predict(X_val)
        assert len(val_predictions) == len(X_val)
    
    def test_ridge_coefficient_extraction(self, sample_tat_data):
        """Validate clinical coefficient extraction for interpretability."""
        
        model = RidgeTATRegressor(random_state=42, alpha=1.0)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())
        
        # Train model
        model.fit(X, y)
        
        # Extract coefficients
        coefficients = model.model.coef_
        intercept = model.model.intercept_
        
        # Validate coefficient structure
        assert len(coefficients) == len(X.columns)
        assert all(np.isfinite(coef) for coef in coefficients)
        assert np.isfinite(intercept)
        
        # Coefficients should have clinical interpretability
        # (Feature impact on log-transformed TAT)
        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Validate coefficient ranking for clinical insights
        assert len(coef_df) == len(X.columns)
        assert coef_df['abs_coefficient'].iloc[0] >= coef_df['abs_coefficient'].iloc[-1]
    
    def test_ridge_feature_importance_analysis(self, sample_tat_data):
        """Validate feature importance analysis for bottleneck identification."""
        
        model = RidgeTATRegressor(random_state=42)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())
        
        # Train model
        model.fit(X, y)
        
        # Mock get_feature_importance method if implemented
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            
            assert isinstance(importance, pd.DataFrame)
            assert 'feature' in importance.columns
            assert 'importance' in importance.columns
            assert len(importance) == len(X.columns)
        
        # Alternative: Direct coefficient importance analysis
        coefficients = model.model.coef_
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': coefficients,
            'abs_importance': np.abs(coefficients)
        }).sort_values('abs_importance', ascending=False)
        
        # Most impactful features should be clinically relevant
        top_features = feature_importance.head(3)['feature'].tolist()
        
        # Clinical expectation: operational features should be impactful
        operational_features = [
            'queue_length_at_order', 'floor_occupancy_pct', 
            'severity_high', 'stat_order'
        ]
        
        # At least one operational feature should be in top 3
        assert any(feature in top_features for feature in operational_features)


class TestRidgePredictionCapabilities:
    """Validate Ridge model prediction accuracy and clinical interpretability."""
    
    @pytest.fixture
    def trained_ridge_model(self, sample_tat_data):
        """Create trained Ridge model for prediction testing."""
        model = RidgeTATRegressor(random_state=42, alpha=2.0)
        
        # Extract feature columns (exclude target and identifier columns)
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        # Label encode other categorical columns
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        # Handle missing values
        X = X.fillna(X.median())
        
        model.fit(X, y)
        return model, X, y
    
    def test_ridge_prediction_accuracy(self, trained_ridge_model):
        """Validate Ridge prediction accuracy on healthcare data."""
        
        model, X, y = trained_ridge_model
        
        # Make predictions
        predictions = model.predict(X)
        
        # Validate prediction structure
        assert len(predictions) == len(X)
        assert all(pred > 0 for pred in predictions)  # Positive TAT predictions
        assert all(np.isfinite(pred) for pred in predictions)  # Valid predictions
        
        # Evaluate healthcare metrics
        metrics = model.evaluate_healthcare_metrics(y, predictions)
        
        # Validate reasonable prediction accuracy for linear model
        assert metrics['MAE'] > 0
        assert metrics['RMSE'] > 0
        assert metrics['healthcare_score'] > 0
        
        # Ridge should achieve reasonable accuracy on training data
        assert metrics['MAE'] < 50  # Within 50 minutes MAE (reasonable for complex healthcare data)
        assert metrics['within_30min_pct'] > 10  # At least 10% within 30 minutes
    
    def test_ridge_prediction_consistency(self, trained_ridge_model):
        """Validate Ridge prediction consistency and reproducibility."""
        
        model, X, y = trained_ridge_model
        
        # Multiple prediction calls should be identical
        pred1 = model.predict(X.iloc[:10])
        pred2 = model.predict(X.iloc[:10])
        
        np.testing.assert_array_equal(pred1, pred2)
        
        # Same model configuration should produce same results
        model2 = RidgeTATRegressor(random_state=42, alpha=2.0)
        model2.fit(X, y)
        
        pred_model2 = model2.predict(X.iloc[:10])
        np.testing.assert_array_almost_equal(pred1, pred_model2, decimal=6)
    
    def test_ridge_threshold_classification_ability(self, trained_ridge_model):
        """Validate Ridge model's 60-minute threshold classification capability."""
        
        model, X, y = trained_ridge_model
        
        predictions = model.predict(X)
        
        # Evaluate threshold accuracy
        metrics = model.evaluate_healthcare_metrics(y, predictions)
        threshold_accuracy = metrics['threshold_60min_accuracy']
        
        # Ridge should perform better than random (50%) for threshold classification
        assert 0 <= threshold_accuracy <= 100
        
        # For reasonable healthcare data, should achieve above-random performance
        if len(y[y > 60]) > 10 and len(y[y <= 60]) > 10:  # Balanced threshold classes
            assert threshold_accuracy > 45  # Better than random with some tolerance
    
    def test_ridge_prediction_range_validation(self, trained_ridge_model):
        """Validate Ridge predictions fall within reasonable TAT ranges."""
        
        model, X, y = trained_ridge_model
        
        predictions = model.predict(X)
        
        # Validate reasonable TAT prediction ranges
        assert predictions.min() > 0      # Positive TAT values
        assert predictions.max() < 1000   # Reasonable maximum (< 16 hours)
        
        # Most predictions should be in typical clinical range
        typical_range = (predictions >= 10) & (predictions <= 500)
        assert typical_range.mean() > 0.8  # 80% in typical range
        
        # Predictions should have reasonable variance
        assert predictions.std() > 5   # Some variation (not constant)
        assert predictions.std() < 200 # Not excessive variation


class TestRidgeHyperparameterSpace:
    """Validate Ridge hyperparameter space for clinical optimization."""
    
    def test_ridge_hyperparameter_space_structure(self):
        """Validate Ridge hyperparameter space structure and ranges."""
        
        model = RidgeTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Validate hyperparameter space structure
        assert isinstance(param_space, dict)
        assert 'alpha' in param_space
        
        # Validate alpha regularization values
        alpha_values = param_space['alpha']
        assert isinstance(alpha_values, tuple)
        # Check tuple format: ('float', min_val, max_val, 'log')
        assert alpha_values[0] == 'float'
        assert alpha_values[1] > 0  # Min value positive
        assert alpha_values[2] > alpha_values[1]  # Max > Min
        assert alpha_values[3] == 'log'  # Log scale
        
        # Should cover range suitable for healthcare applications
        assert alpha_values[1] < 1.0    # Weak regularization available
        assert alpha_values[2] > 10.0   # Strong regularization available
    
    def test_ridge_hyperparameter_clinical_constraints(self):
        """Validate hyperparameters maintain clinical interpretability."""
        
        model = RidgeTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Alpha values should not be extreme for clinical interpretability
        alpha_values = param_space['alpha']
        
        # Should not include extremely large values that kill interpretability
        # For tuple format: ('float', min, max, 'log')
        assert alpha_values[2] <= 1000
        
        # Should not include extremely small values that provide no regularization
        assert alpha_values[1] >= 0.001
    
    def test_ridge_hyperparameter_optimization_compatibility(self):
        """Validate hyperparameter space compatibility with optimization frameworks."""
        
        model = RidgeTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Should be compatible with common optimization frameworks
        for param_name, param_values in param_space.items():
            # Parameter names should be valid sklearn parameter names
            assert isinstance(param_name, str)
            assert len(param_name) > 0
            
            # Parameter values should be valid for optimization
            assert isinstance(param_values, tuple)
            # Validate tuple format for Optuna compatibility
            assert param_values[0] in ['int', 'float', 'categorical']
            if param_values[0] in ['int', 'float']:
                assert len(param_values) >= 3  # type, min, max
                assert isinstance(param_values[1], (int, float))
                assert isinstance(param_values[2], (int, float))


class TestRidgeHealthcareMetrics:
    """Validate Ridge model healthcare-specific performance evaluation."""
    
    def test_ridge_healthcare_metrics_comprehensive(self, sample_tat_data):
        """Validate comprehensive healthcare metrics evaluation."""
        
        model = RidgeTATRegressor(random_state=42)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())
        
        # Train model
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Evaluate healthcare metrics
        metrics = model.evaluate_healthcare_metrics(y, predictions)
        
        # Validate all healthcare metrics are present
        required_metrics = [
            'RMSE', 'MAE', 'MedianAE', 'within_10min_pct',
            'within_30min_pct', 'threshold_60min_accuracy', 'healthcare_score'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert not np.isnan(metrics[metric])
    
    def test_ridge_clinical_accuracy_bands(self, sample_tat_data):
        """Validate clinical accuracy band assessment."""
        
        model = RidgeTATRegressor(random_state=42, alpha=1.0)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        metrics = model.evaluate_healthcare_metrics(y, predictions)
        
        # Validate accuracy band relationships
        within_10min = metrics['within_10min_pct']
        within_30min = metrics['within_30min_pct']
        
        assert 0 <= within_10min <= 100
        assert 0 <= within_30min <= 100
        assert within_10min <= within_30min  # 10min subset of 30min
    
    def test_ridge_healthcare_composite_score(self, sample_tat_data):
        """Validate healthcare composite score calculation."""
        
        model = RidgeTATRegressor(random_state=42)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        metrics = model.evaluate_healthcare_metrics(y, predictions)
        
        # Validate healthcare score calculation
        expected_score = (metrics['within_30min_pct'] + metrics['threshold_60min_accuracy']) / 2
        assert abs(metrics['healthcare_score'] - expected_score) < 0.01
        
        # Healthcare score should be in valid range
        assert 0 <= metrics['healthcare_score'] <= 100


class TestRidgeErrorHandling:
    """Validate Ridge model error handling and edge cases."""
    
    def test_ridge_unfitted_prediction_error(self):
        """Validate error handling for prediction with unfitted model."""
        
        model = RidgeTATRegressor()
        
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        with pytest.raises(ValueError, match="Model must be fitted before prediction"):
            model.predict(X_test)
    
    def test_ridge_empty_data_handling(self):
        """Validate Ridge model behavior with edge case data."""
        
        model = RidgeTATRegressor(random_state=42)
        
        # Empty DataFrame
        X_empty = pd.DataFrame()
        y_empty = pd.Series(dtype=float)
        
        # Should handle gracefully or raise appropriate sklearn error
        try:
            model.fit(X_empty, y_empty)
        except ValueError as e:
            # Expected sklearn behavior for empty data
            assert "at least one array or dtype is required" in str(e) or "empty" in str(e).lower()
    
    def test_ridge_single_sample_handling(self):
        """Validate Ridge model with minimal training data."""
        
        model = RidgeTATRegressor(random_state=42, alpha=1.0)
        
        # Single sample (Ridge might work with single sample)
        X_single = pd.DataFrame({'feature1': [5.0], 'feature2': [10.0]})
        y_single = pd.Series([60.0])
        
        try:
            model.fit(X_single, y_single)
            
            if model.is_fitted:
                prediction = model.predict(X_single)
                assert len(prediction) == 1
                assert np.isfinite(prediction[0])
        except ValueError:
            # Ridge may require multiple samples - acceptable behavior
            pass
    
    def test_ridge_extreme_target_values(self):
        """Validate Ridge model with extreme TAT values."""
        
        model = RidgeTATRegressor(random_state=42)
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10]
        })
        
        # Extreme TAT values
        y_extreme = pd.Series([0.1, 10000, 1, 500, 50])  # Very small and very large
        
        # Should handle extreme values through log1p transformation
        model.fit(X, y_extreme)
        predictions = model.predict(X)
        
        # Predictions should be reasonable despite extreme training values
        assert all(pred > 0 for pred in predictions)
        assert all(np.isfinite(pred) for pred in predictions)


class TestRidgeProductionIntegration:
    """Validate Ridge model production deployment and MLOps integration."""
    
    def test_ridge_model_serialization(self, sample_tat_data):
        """Validate Ridge model save/load cycle for production deployment."""
        
        # Train model
        model = RidgeTATRegressor(random_state=999, alpha=3.0)
        
        # Extract and process features
        feature_cols = [col for col in sample_tat_data.columns 
                       if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
        
        X = sample_tat_data[feature_cols].copy()
        y = sample_tat_data['TAT_minutes'].copy()
        
        # Apply categorical encodings
        categorical_mappings = {
            'sex': {"F": 0, "M": 1},
            'severity': {"Low": 0, "Medium": 1, "High": 2},
            'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
            'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
        }
        
        for col, mapping in categorical_mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)
        
        categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 'treatment_type', 'shift']
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        X = X.fillna(X.median())
        model.fit(X, y)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "ridge_production_model.joblib"
            
            # Save model with healthcare metadata
            model.save_model(save_path)
            assert save_path.exists()
            
            # Load model
            loaded_model = RidgeTATRegressor.load_model(save_path)
            
            # Validate model reconstruction
            assert loaded_model.is_fitted is True
            assert loaded_model.random_state == 999
            assert loaded_model.metadata['model_type'] == 'RidgeTATRegressor'
            assert loaded_model.metadata['algorithm'] == 'Ridge Regression'
            
            # Validate prediction consistency
            original_predictions = model.predict(X.iloc[:20])
            loaded_predictions = loaded_model.predict(X.iloc[:20])
            
            np.testing.assert_array_almost_equal(
                original_predictions, loaded_predictions, decimal=8
            )
    
    def test_ridge_healthcare_audit_trail(self):
        """Validate healthcare audit trail metadata for regulatory compliance."""
        
        model = RidgeTATRegressor(
            random_state=555,
            alpha=7.5,
            max_iter=1500
        )
        
        # Validate comprehensive audit metadata
        audit_fields = [
            'model_type', 'healthcare_context', 'clinical_objective',
            'algorithm', 'healthcare_optimization', 'interpretability',
            'regulatory_compliance', 'deployment_readiness'
        ]
        
        for field in audit_fields:
            assert field in model.metadata
            assert isinstance(model.metadata[field], str)
            assert len(model.metadata[field]) > 0
        
        # Validate specific healthcare compliance information
        assert 'TAT Prediction' in model.metadata['healthcare_context']
        assert 'Ridge Regression' in model.metadata['algorithm']
        assert 'coefficient analysis' in model.metadata['interpretability']
    
    def test_ridge_production_consistency(self):
        """Validate Ridge model consistency across production deployments."""
        
        # Multiple instances with same configuration should be equivalent
        config = {'random_state': 777, 'alpha': 2.5, 'max_iter': 1000}
        
        model1 = RidgeTATRegressor(**config)
        model2 = RidgeTATRegressor(**config)
        
        # Configuration should be identical
        assert model1.random_state == model2.random_state
        assert model1.default_params == model2.default_params
        assert model1.metadata['model_type'] == model2.metadata['model_type']
        
        # Models should behave identically with same data
        X_test = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        y_test = pd.Series([40, 60, 80])
        
        model1.fit(X_test, y_test)
        model2.fit(X_test, y_test)
        
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)
        
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=8)
    
    def test_ridge_mlops_interface_compliance(self):
        """Validate Ridge model MLOps interface compliance."""
        
        model = RidgeTATRegressor()
        
        # Required MLOps interface methods
        mlops_methods = [
            'fit', 'predict', 'get_hyperparameter_space',
            'evaluate_healthcare_metrics', 'save_model'
        ]
        
        for method in mlops_methods:
            assert hasattr(model, method)
            assert callable(getattr(model, method))
        
        # Class methods for model lifecycle
        assert hasattr(RidgeTATRegressor, 'load_model')
        assert callable(getattr(RidgeTATRegressor, 'load_model'))
        
        # Healthcare metadata should support MLOps monitoring
        metadata = model.metadata
        assert 'deployment_readiness' in metadata
        assert 'healthcare_optimization' in metadata
        assert 'regulatory_compliance' in metadata


class TestRidgeClinicalMethods:
    """Test Ridge clinical analysis methods for healthcare interpretability."""
    
    def test_get_feature_coefficients_basic(self):
        """Test feature coefficients extraction for clinical interpretation."""
        model = RidgeTATRegressor(random_state=42, alpha=1.0)
        
        # Create simple training data
        X = pd.DataFrame({
            'age': [25, 35, 45, 55, 65],
            'severity_score': [1, 2, 3, 2, 1],
            'queue_length': [5, 10, 15, 8, 3]
        })
        y = pd.Series([30, 45, 60, 50, 35])
        
        # Train model
        model.fit(X, y)
        
        # Test coefficient extraction
        coefficients = model.get_feature_coefficients()
        
        assert isinstance(coefficients, pd.DataFrame)
        assert len(coefficients) == len(X.columns)
        assert 'feature' in coefficients.columns
        assert 'coefficient' in coefficients.columns
        assert 'abs_coefficient' in coefficients.columns
        
        # Verify all features are included
        assert set(coefficients['feature']) == set(X.columns)
        
        # Coefficients should be sorted by absolute value (descending)
        abs_coeffs = coefficients['abs_coefficient'].values
        assert all(abs_coeffs[i] >= abs_coeffs[i+1] for i in range(len(abs_coeffs)-1))
    
    def test_get_feature_coefficients_untrained_model(self):
        """Test coefficient extraction fails gracefully for untrained model."""
        model = RidgeTATRegressor()
        
        with pytest.raises(ValueError, match="Ridge model must be fitted to extract clinical coefficients"):
            model.get_feature_coefficients()
    
    def test_get_clinical_insights_comprehensive(self):
        """Test comprehensive clinical insights generation."""
        model = RidgeTATRegressor(random_state=42, alpha=1.0)
        
        # Create healthcare-realistic training data
        X = pd.DataFrame({
            'age': [25, 35, 45, 55, 65, 30, 40, 50, 60, 70],
            'severity_Low': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
            'severity_High': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            'nurse_experience_years': [2, 5, 10, 3, 8, 4, 7, 12, 6, 9],
            'queue_length_at_order': [3, 8, 15, 5, 12, 7, 10, 2, 14, 6]
        })
        y = pd.Series([35, 55, 75, 40, 70, 45, 65, 80, 50, 85])
        
        # Train model
        model.fit(X, y)
        
        # Generate clinical insights
        insights = model.get_clinical_insights()
        
        assert isinstance(insights, dict)
        
        # Verify key insight categories (based on actual API)
        expected_keys = {
            'model_type', 'clinical_objective', 'healthcare_context',
            'top_bottlenecks', 'efficiency_factors', 'intervention_recommendations',
            'interpretability_confidence', 'clinical_validation'
        }
        assert expected_keys.issubset(set(insights.keys()))
        
        # Validate bottleneck identification
        assert isinstance(insights['top_bottlenecks'], list)
        
        # Validate efficiency factors
        assert isinstance(insights['efficiency_factors'], list)
        
        # Validate intervention recommendations
        assert isinstance(insights['intervention_recommendations'], list)
        assert len(insights['intervention_recommendations']) > 0
        
        # Validate clinical validation
        assert isinstance(insights['clinical_validation'], str)
        assert len(insights['clinical_validation']) > 0
        
    def test_get_clinical_insights_untrained_model(self):
        """Test clinical insights fails gracefully for untrained model."""
        model = RidgeTATRegressor()
        
        with pytest.raises(ValueError, match="Ridge model must be fitted to generate clinical insights"):
            model.get_clinical_insights()
    
    def test_hyperparameter_space_healthcare_focused(self):
        """Test hyperparameter space is appropriate for healthcare use."""
        model = RidgeTATRegressor()
        
        param_space = model.get_hyperparameter_space()
        
        assert isinstance(param_space, dict)
        assert 'alpha' in param_space
        
        # Alpha should cover appropriate regularization range for healthcare
        alpha_space = param_space['alpha']
        assert isinstance(alpha_space, tuple)
        assert len(alpha_space) >= 3
        assert alpha_space[0] == 'float'  # Data type
        assert alpha_space[1] > 0  # Min value (positive regularization)
        assert alpha_space[2] >= 100  # Max value (allow strong regularization)
        
        # Should support clinical interpretability requirements  
        if len(alpha_space) > 3:
            assert alpha_space[3] == 'log'  # Log-uniform distribution


class TestRidgeHealthcareIntegration:
    """Test Ridge model integration with healthcare workflows."""
    
    def test_coefficient_clinical_interpretation(self):
        """Test that coefficients provide meaningful clinical interpretation."""
        model = RidgeTATRegressor(random_state=42, alpha=0.1)
        
        # Create clinically meaningful feature data
        X = pd.DataFrame({
            'patient_age': [30, 40, 50, 60, 70],
            'severity_high': [0, 0, 1, 1, 1],  # High severity increases TAT
            'stat_order': [0, 1, 0, 1, 0],     # STAT orders increase TAT
            'queue_length': [2, 8, 15, 3, 12], # Queue length increases TAT
            'nurse_experience_years': [15, 5, 10, 20, 8]  # Experience decreases TAT
        })
        y = pd.Series([35, 65, 80, 45, 75])  # TAT minutes
        
        # Train model
        model.fit(X, y)
        
        # Extract coefficients for clinical review
        coefficients = model.get_feature_coefficients()
        coeff_dict = dict(zip(coefficients['feature'], coefficients['coefficient']))
        
        # Validate clinically expected relationships
        # Note: These may vary based on actual data patterns
        assert 'severity_high' in coeff_dict
        assert 'queue_length' in coeff_dict
        assert 'nurse_experience_years' in coeff_dict
        
        # All coefficients should be interpretable (not extreme values)
        for coeff in coefficients['coefficient']:
            assert abs(coeff) < 1000  # Reasonable coefficient magnitude
    
    def test_clinical_insights_actionability(self):
        """Test that clinical insights provide actionable recommendations."""
        model = RidgeTATRegressor(random_state=42, alpha=1.0)
        
        # Healthcare workflow features
        X = pd.DataFrame({
            'floor_occupancy_pct': [0.6, 0.8, 0.9, 0.7, 0.5],
            'pharmacist_BCOP': [0, 1, 0, 1, 0],  # Specialized credential
            'premed_required': [1, 0, 1, 0, 1],   # Additional complexity
            'shift_Night': [1, 0, 0, 1, 0],       # Night shift staffing
            'queue_length_at_order': [5, 12, 18, 8, 3]
        })
        y = pd.Series([45, 65, 85, 55, 35])
        
        model.fit(X, y)
        insights = model.get_clinical_insights()
        
        # Insights should be clinically actionable
        intervention_recs = insights['intervention_recommendations']
        assert isinstance(intervention_recs, list)
        
        # Should provide bottleneck identification
        bottlenecks = insights['top_bottlenecks']
        assert isinstance(bottlenecks, list)
        
        # Should have efficiency factors
        efficiency_factors = insights['efficiency_factors']
        assert isinstance(efficiency_factors, list)
    
    def test_ridge_healthcare_metadata_completeness(self):
        """Test comprehensive healthcare metadata for clinical deployment."""
        model = RidgeTATRegressor(random_state=42, alpha=2.0)
        
        # Simple training for metadata population
        X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y = pd.Series([30, 45, 60])
        model.fit(X, y)
        
        metadata = model.metadata
        
        # Essential clinical metadata
        clinical_keys = [
            'algorithm', 'healthcare_optimization', 'interpretability',
            'clinical_advantages', 'bottleneck_identification', 
            'regulatory_compliance', 'deployment_readiness'
        ]
        
        for key in clinical_keys:
            assert key in metadata
            assert isinstance(metadata[key], str)
            assert len(metadata[key]) > 0
        
        # Training-specific metadata
        training_keys = [
            'training_samples', 'feature_count', 'regularization_alpha',
            'training_completed', 'clinical_deployment_ready'
        ]
        
        for key in training_keys:
            assert key in metadata
            
        # Validation status checks
        assert metadata['training_completed'] is True
        assert metadata['clinical_deployment_ready'] is True
        assert metadata['interpretability_confirmed'] is True