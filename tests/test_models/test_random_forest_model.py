"""
Test Suite for Healthcare Random Forest TAT Prediction Model

Comprehensive validation ensuring Random Forest ensemble learning supports Dana Farber's
medication preparation TAT prediction with clinical interpretability, robust feature
importance analysis, and healthcare-appropriate hyperparameter optimization. Validates
ensemble architecture, tree-based decision pathways, and clinical insight generation
for pharmacy workflow optimization and bottleneck identification workflows.

Test Coverage:
- RandomForestTATRegressor initialization with healthcare parameters
- Ensemble training with bootstrap aggregation and out-of-bag validation  
- Feature importance extraction and clinical bottleneck identification
- Clinical insights generation with intervention recommendations
- Hyperparameter space definition for healthcare optimization
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
from sklearn.ensemble import RandomForestRegressor

from src.tat.models.random_forest_model import RandomForestTATRegressor


class TestRandomForestInitialization:
    """Test RandomForestTATRegressor model initialization and configuration."""
    
    def test_default_initialization(self):
        """Test default Random Forest model initialization with healthcare parameters."""
        model = RandomForestTATRegressor()
        
        # Validate healthcare-optimized default parameters
        assert model.default_params['n_estimators'] == 200
        assert model.default_params['max_depth'] == 8
        assert model.default_params['min_samples_split'] == 5
        assert model.default_params['min_samples_leaf'] == 2
        assert model.default_params['max_features'] == 'sqrt'
        assert model.default_params['bootstrap'] is True
        assert model.default_params['oob_score'] is True
        assert model.default_params['n_jobs'] == -1
        
        # Validate Random Forest model initialization
        assert isinstance(model.model, RandomForestRegressor)
        assert model.model.n_estimators == 200
        assert model.model.max_depth == 8
        
        # Validate healthcare metadata
        assert model.metadata['algorithm'] == 'Random Forest Regression'
        assert 'ensemble robustness' in model.metadata['healthcare_optimization']
        assert 'Feature importance' in model.metadata['interpretability']
        
    def test_custom_parameters_initialization(self):
        """Test Random Forest initialization with custom healthcare parameters."""
        custom_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 10,
            'max_features': 0.8
        }
        
        model = RandomForestTATRegressor(random_state=123, **custom_params)
        
        # Validate custom parameter integration
        assert model.default_params['n_estimators'] == 100
        assert model.default_params['max_depth'] == 6
        assert model.default_params['min_samples_split'] == 10
        assert model.default_params['max_features'] == 0.8
        assert model.default_params['random_state'] == 123
        
        # Validate model reflects custom parameters
        assert model.model.n_estimators == 100
        assert model.model.max_depth == 6
        
    def test_healthcare_metadata_completeness(self):
        """Test comprehensive healthcare metadata initialization."""
        model = RandomForestTATRegressor()
        
        # Validate clinical metadata completeness
        required_metadata = [
            'algorithm', 'healthcare_optimization', 'interpretability',
            'clinical_advantages', 'bottleneck_identification',
            'regulatory_compliance', 'deployment_readiness'
        ]
        
        for key in required_metadata:
            assert key in model.metadata
            assert isinstance(model.metadata[key], str)
            assert len(model.metadata[key]) > 0


class TestRandomForestTraining:
    """Test Random Forest ensemble training with healthcare data."""
    
    @pytest.fixture
    def training_data(self):
        """Generate realistic healthcare TAT training dataset."""
        np.random.seed(42)
        n_samples = 1000
        
        # Healthcare feature simulation
        data = {
            'medication_complexity': np.random.exponential(2, n_samples),
            'queue_length': np.random.poisson(5, n_samples),
            'staff_experience_years': np.random.gamma(3, 2, n_samples),
            'time_of_day': np.random.uniform(0, 24, n_samples),
            'day_of_week': np.random.randint(1, 8, n_samples),
            'preparation_urgency': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'equipment_availability': np.random.beta(3, 1, n_samples),
            'verification_required': np.random.binomial(1, 0.3, n_samples)
        }
        
        X = pd.DataFrame(data)
        
        # Realistic TAT generation with healthcare constraints
        y = (15 + 
             X['medication_complexity'] * 8 +
             X['queue_length'] * 2 +
             -X['staff_experience_years'] * 0.5 +
             X['preparation_urgency'] * 3 +
             np.random.exponential(5, n_samples))
        
        return X, pd.Series(y, name='tat_minutes')
    
    def test_successful_training(self, training_data):
        """Test successful Random Forest ensemble training."""
        X, y = training_data
        model = RandomForestTATRegressor(n_estimators=50)  # Faster for testing
        
        # Train ensemble model
        fitted_model = model.fit(X, y)
        
        # Validate training completion
        assert fitted_model is model
        assert model.is_fitted
        
        # Validate training metadata
        assert model.metadata['training_samples'] == len(X)
        assert model.metadata['feature_count'] == X.shape[1]
        assert model.metadata['training_completed'] is True
        assert model.metadata['clinical_deployment_ready'] is True
        
        # Validate out-of-bag scoring
        assert model.metadata['oob_score'] is not None
        assert isinstance(model.metadata['oob_score'], float)
        
    def test_target_transformation_during_training(self, training_data):
        """Test target transformation handling during Random Forest training."""
        X, y = training_data
        model = RandomForestTATRegressor()
        model.target_transform = 'log1p'  # Set after initialization
        
        # Test with transformed targets
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.metadata['target_transform'] == 'log1p'
        
    def test_ensemble_configuration_preservation(self, training_data):
        """Test Random Forest ensemble configuration preservation after training."""
        X, y = training_data
        model = RandomForestTATRegressor(n_estimators=75, max_depth=10)
        
        model.fit(X, y)
        
        # Validate ensemble configuration preserved
        assert model.metadata['n_estimators_used'] == 75
        assert model.metadata['max_depth_used'] == 10
        assert model.metadata['feature_importances_available'] is True
        
    def test_training_with_validation_data(self, training_data):
        """Test Random Forest training with validation data (API consistency)."""
        X, y = training_data
        X_val, y_val = X[:100], y[:100]
        
        model = RandomForestTATRegressor()
        
        # Validation data should not affect Random Forest training
        model.fit(X[100:], y[100:], validation_data=(X_val, y_val))
        
        assert model.is_fitted
        assert model.metadata['training_samples'] == len(X) - 100


class TestFeatureImportance:
    """Test Random Forest feature importance extraction for clinical insights."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create fitted Random Forest model for importance testing."""
        np.random.seed(42)
        n_samples = 500
        
        # Healthcare features with known importance patterns
        X = pd.DataFrame({
            'critical_factor': np.random.exponential(2, n_samples),     # High importance
            'moderate_factor': np.random.gamma(2, 1, n_samples),       # Moderate importance  
            'low_factor': np.random.uniform(0, 1, n_samples),          # Low importance
            'noise_factor': np.random.normal(0, 0.1, n_samples)       # Minimal importance
        })
        
        # Target with clear feature relationships
        y = (10 + 
             X['critical_factor'] * 15 +     # Strong relationship
             X['moderate_factor'] * 5 +      # Moderate relationship
             X['low_factor'] * 1 +           # Weak relationship  
             np.random.exponential(2, n_samples))
        
        model = RandomForestTATRegressor(n_estimators=50, random_state=42)
        model.fit(X, pd.Series(y))
        return model
    
    def test_feature_importance_extraction(self, fitted_model):
        """Test comprehensive feature importance extraction."""
        importance = fitted_model.get_feature_importance()
        
        # Validate importance DataFrame structure
        assert isinstance(importance, pd.DataFrame)
        assert len(importance) == 4  # Four features
        
        # Validate required columns
        required_columns = [
            'feature', 'importance', 'importance_pct', 'importance_rank',
            'clinical_significance', 'bottleneck_potential'
        ]
        for col in required_columns:
            assert col in importance.columns
            
        # Validate healthcare context columns
        healthcare_columns = [
            'healthcare_context', 'clinical_objective', 
            'ensemble_confidence', 'interpretation_method'
        ]
        for col in healthcare_columns:
            assert col in importance.columns
            
    def test_feature_importance_ranking(self, fitted_model):
        """Test feature importance ranking and percentage calculation."""
        importance = fitted_model.get_feature_importance()
        
        # Validate importance ranking
        assert importance['importance_rank'].iloc[0] == 1  # Top feature
        assert importance['importance_rank'].is_monotonic_increasing
        
        # Validate percentage calculation
        assert abs(importance['importance_pct'].sum() - 100.0) < 0.001
        assert all(importance['importance_pct'] >= 0)
        
        # Validate sorted by importance
        assert importance['importance'].is_monotonic_decreasing
        
    def test_clinical_significance_categorization(self, fitted_model):
        """Test clinical significance and bottleneck potential categorization."""
        importance = fitted_model.get_feature_importance()
        
        # Validate clinical significance categories
        sig_categories = importance['clinical_significance'].unique()
        assert all(cat in ['High', 'Moderate', 'Low'] for cat in sig_categories)
        
        # Validate bottleneck potential categories
        bottleneck_categories = importance['bottleneck_potential'].unique()
        valid_bottlenecks = ['Critical', 'Significant', 'Moderate', 'Limited']
        assert all(cat in valid_bottlenecks for cat in bottleneck_categories)
        
    def test_unfitted_model_importance_error(self):
        """Test feature importance extraction error on unfitted model."""
        model = RandomForestTATRegressor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.get_feature_importance()


class TestClinicalInsights:
    """Test Random Forest clinical insights generation for healthcare stakeholders."""
    
    @pytest.fixture
    def clinical_model(self):
        """Create Random Forest model with clinical feature patterns."""
        np.random.seed(42)
        n_samples = 800
        
        # Healthcare features with clinical relevance
        X = pd.DataFrame({
            'medication_complexity': np.random.exponential(2, n_samples),
            'pharmacist_experience': np.random.gamma(3, 2, n_samples),
            'queue_depth': np.random.poisson(4, n_samples),
            'verification_steps': np.random.randint(1, 6, n_samples),
            'preparation_urgency': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'equipment_status': np.random.beta(3, 1, n_samples)
        })
        
        # Clinical TAT with realistic bottleneck patterns
        y = (12 + 
             X['medication_complexity'] * 12 +      # Critical bottleneck
             X['queue_depth'] * 4 +                 # Significant factor
             X['verification_steps'] * 3 +          # Moderate factor
             -X['pharmacist_experience'] * 0.8 +    # Efficiency factor
             np.random.exponential(3, n_samples))
        
        model = RandomForestTATRegressor(n_estimators=100, random_state=42)
        model.fit(X, pd.Series(y))
        return model
    
    def test_clinical_insights_generation(self, clinical_model):
        """Test comprehensive clinical insights generation."""
        insights = clinical_model.get_clinical_insights()
        
        # Validate insights structure
        assert isinstance(insights, dict)
        
        # Validate required insight components
        required_components = [
            'model_type', 'clinical_objective', 'healthcare_context',
            'critical_bottlenecks', 'significant_drivers', 
            'intervention_recommendations', 'ensemble_advantages'
        ]
        for component in required_components:
            assert component in insights
            
    def test_critical_bottleneck_identification(self, clinical_model):
        """Test critical bottleneck identification and analysis."""
        insights = clinical_model.get_clinical_insights()
        
        critical_bottlenecks = insights['critical_bottlenecks']
        
        # Validate critical bottleneck structure
        assert isinstance(critical_bottlenecks, list)
        assert len(critical_bottlenecks) <= 5  # Top 5 maximum
        
        if critical_bottlenecks:
            bottleneck = critical_bottlenecks[0]
            
            # Validate bottleneck attributes
            required_attrs = [
                'feature', 'importance_pct', 'clinical_impact',
                'intervention_priority', 'bottleneck_type', 'evidence_strength'
            ]
            for attr in required_attrs:
                assert attr in bottleneck
                
            # Validate intervention priority
            assert bottleneck['intervention_priority'] == 'Immediate'
            assert bottleneck['bottleneck_type'] == 'Critical workflow constraint'
            
    def test_significant_driver_analysis(self, clinical_model):
        """Test significant driver identification and categorization."""
        insights = clinical_model.get_clinical_insights()
        
        significant_drivers = insights['significant_drivers']
        
        # Validate significant drivers structure  
        assert isinstance(significant_drivers, list)
        assert len(significant_drivers) <= 5  # Top 5 maximum
        
        if significant_drivers:
            driver = significant_drivers[0]
            
            # Validate driver attributes
            assert 'feature' in driver
            assert 'importance_pct' in driver
            assert 'clinical_impact' in driver
            
            # Validate intervention priority
            assert driver['intervention_priority'] == 'High'
            assert driver['bottleneck_type'] == 'Secondary workflow factor'
            
    def test_intervention_recommendations(self, clinical_model):
        """Test evidence-based intervention recommendation generation."""
        insights = clinical_model.get_clinical_insights()
        
        recommendations = insights['intervention_recommendations']
        
        # Validate recommendations structure
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 1  # At least ensemble validation
        
        # Validate recommendation content
        for rec in recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0
            
        # Check for ensemble validation recommendation
        ensemble_rec = any('Ensemble validation' in rec for rec in recommendations)
        assert ensemble_rec
        
    def test_ensemble_advantages_documentation(self, clinical_model):
        """Test Random Forest ensemble advantages documentation."""
        insights = clinical_model.get_clinical_insights()
        
        advantages = insights['ensemble_advantages']
        
        # Validate advantages structure
        assert isinstance(advantages, list)
        assert len(advantages) >= 3  # Multiple advantages documented
        
        # Check for key ensemble advantages
        advantage_text = ' '.join(advantages)
        assert 'bootstrap' in advantage_text.lower()
        assert 'interaction' in advantage_text.lower()
        assert 'overfitting' in advantage_text.lower()
        
    def test_clinical_validation_metrics(self, clinical_model):
        """Test clinical validation and confidence metrics."""
        insights = clinical_model.get_clinical_insights()
        
        # Validate validation metrics
        assert 'interpretability_confidence' in insights
        assert 'clinical_validation' in insights
        assert 'total_features_analyzed' in insights
        
        # Check confidence levels
        assert insights['interpretability_confidence'] == 'High - Tree-based ensemble with feature importance'
        assert isinstance(insights['total_features_analyzed'], int)
        assert insights['total_features_analyzed'] > 0
        
    def test_unfitted_model_insights_error(self):
        """Test clinical insights error on unfitted model."""
        model = RandomForestTATRegressor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.get_clinical_insights()


class TestHyperparameterSpace:
    """Test Random Forest hyperparameter space definition for healthcare optimization."""
    
    def test_hyperparameter_space_structure(self):
        """Test hyperparameter space structure and ranges."""
        model = RandomForestTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Validate parameter space structure
        assert isinstance(param_space, dict)
        
        # Validate required hyperparameters
        required_params = [
            'n_estimators', 'max_depth', 'min_samples_split',
            'min_samples_leaf', 'max_features'
        ]
        for param in required_params:
            assert param in param_space
            
    def test_ensemble_parameter_ranges(self):
        """Test Random Forest ensemble parameter ranges for healthcare."""
        model = RandomForestTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Validate n_estimators range
        n_est = param_space['n_estimators']
        assert n_est[0] == 'int'
        assert n_est[1] == 100  # Minimum ensemble size
        assert n_est[2] == 300  # Maximum ensemble size
        
        # Validate max_depth range  
        depth = param_space['max_depth']
        assert depth[0] == 'int'
        assert depth[1] == 5   # Minimum interpretability
        assert depth[2] == 15  # Maximum complexity
        
    def test_sampling_parameter_ranges(self):
        """Test sampling parameter ranges for ensemble diversity."""
        model = RandomForestTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Validate min_samples_split range
        split = param_space['min_samples_split']
        assert split[0] == 'int'
        assert split[1] == 2   # Minimum split
        assert split[2] == 10  # Conservative maximum
        
        # Validate min_samples_leaf range
        leaf = param_space['min_samples_leaf']
        assert leaf[0] == 'int'
        assert leaf[1] == 1  # Minimum leaf size
        assert leaf[2] == 5  # Conservative maximum
        
    def test_feature_sampling_strategies(self):
        """Test feature sampling strategies for ensemble diversity."""
        model = RandomForestTATRegressor()
        param_space = model.get_hyperparameter_space()
        
        # Validate max_features options
        max_feat = param_space['max_features']
        assert max_feat[0] == 'categorical'
        
        expected_strategies = ['sqrt', 'log2', 0.5, 0.8]
        for strategy in expected_strategies:
            assert strategy in max_feat[1]


class TestRandomForestPrediction:
    """Test Random Forest ensemble prediction capabilities."""
    
    @pytest.fixture
    def prediction_model(self):
        """Create fitted Random Forest model for prediction testing."""
        np.random.seed(42)
        n_samples = 600
        
        X = pd.DataFrame({
            'factor_1': np.random.exponential(1.5, n_samples),
            'factor_2': np.random.gamma(2, 1.5, n_samples), 
            'factor_3': np.random.uniform(0, 10, n_samples),
            'factor_4': np.random.normal(5, 2, n_samples)
        })
        
        y = (8 + X['factor_1'] * 3 + X['factor_2'] * 2 + 
             np.random.exponential(2, n_samples))
        
        model = RandomForestTATRegressor(n_estimators=50, random_state=42)
        model.fit(X, pd.Series(y))
        return model, X
    
    def test_ensemble_prediction(self, prediction_model):
        """Test Random Forest ensemble prediction functionality."""
        model, X = prediction_model
        
        # Test prediction
        predictions = model.predict(X[:100])
        
        # Validate prediction output
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 100
        assert all(pred > 0 for pred in predictions)  # Positive TAT values
        
    def test_prediction_with_target_transformation(self, prediction_model):
        """Test ensemble prediction with target transformation."""
        model, X = prediction_model
        
        # Enable target transformation
        model.target_transform = 'log1p'
        
        predictions = model.predict(X[:50])
        
        # Validate transformed predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 50
        assert all(pred > 0 for pred in predictions)
        
    def test_unfitted_model_prediction_error(self):
        """Test prediction error on unfitted ensemble model."""
        model = RandomForestTATRegressor()
        X = pd.DataFrame({'feature': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="must be fitted"):
            model.predict(X)


class TestEdgeCasesAndErrorHandling:
    """Test Random Forest edge cases and error handling for robust deployment."""
    
    def test_single_feature_training(self):
        """Test Random Forest training with single feature."""
        np.random.seed(42)
        X = pd.DataFrame({'single_feature': np.random.exponential(2, 200)})
        y = pd.Series(5 + X['single_feature'] * 3 + np.random.normal(0, 1, 200))
        
        model = RandomForestTATRegressor(n_estimators=20)
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.metadata['feature_count'] == 1
        
    def test_small_dataset_handling(self):
        """Test Random Forest handling of small healthcare datasets."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [2, 4, 6, 8, 10]
        })
        y = pd.Series([10, 15, 20, 25, 30])
        
        model = RandomForestTATRegressor(n_estimators=10)
        model.fit(X, y)
        
        assert model.is_fitted
        assert model.metadata['training_samples'] == 5
        
    def test_extreme_tat_values_handling(self):
        """Test Random Forest handling of extreme TAT values."""
        np.random.seed(42)
        X = pd.DataFrame({
            'normal_factor': np.random.normal(5, 2, 100),
            'extreme_factor': np.random.exponential(10, 100)
        })
        
        # Include extreme TAT values
        y_extreme = pd.Series(np.concatenate([
            np.random.exponential(15, 90),  # Normal range
            [300, 500, 800, 1000] * 2,     # Extreme values
            [0.5, 1.0]                     # Very fast
        ]))
        
        model = RandomForestTATRegressor(n_estimators=20)
        model.fit(X, y_extreme)
        
        assert model.is_fitted
        predictions = model.predict(X[:10])
        assert all(pred >= 0 for pred in predictions)
        
    def test_missing_feature_importance_attributes(self):
        """Test handling of Random Forest without feature importance."""
        model = RandomForestTATRegressor()
        
        # Create a mock that lacks the feature_importances_ attribute entirely
        class MockRF:
            pass
        
        model.model = MockRF()
        model.is_fitted = True
        
        with pytest.raises(AttributeError, match="does not have feature importance"):
            model.get_feature_importance()


class TestProductionReadiness:
    """Test Random Forest production deployment readiness and healthcare compliance."""
    
    def test_model_serialization_compatibility(self):
        """Test Random Forest model serialization for production deployment."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.exponential(2, 300),
            'feature_2': np.random.gamma(3, 1, 300)
        })
        y = pd.Series(10 + X['feature_1'] * 5 + np.random.normal(0, 2, 300))
        
        model = RandomForestTATRegressor(n_estimators=30)
        model.fit(X, y)
        
        # Test model save/load functionality
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / 'rf_model'
            model.save_model(model_path)
            
            # Validate model file created (saved directly without extension)
            assert model_path.exists()
            
            # Test model loading
            loaded_model = RandomForestTATRegressor.load_model(model_path)
            
            # Validate loaded model functionality
            original_pred = model.predict(X[:10])
            loaded_pred = loaded_model.predict(X[:10])
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)
            
    def test_healthcare_metadata_preservation(self):
        """Test healthcare metadata preservation across model lifecycle."""
        np.random.seed(42)
        X = pd.DataFrame({'healthcare_feature': np.random.exponential(2, 200)})
        y = pd.Series(8 + X['healthcare_feature'] * 4 + np.random.normal(0, 1.5, 200))
        
        model = RandomForestTATRegressor(
            n_estimators=25,
            max_depth=6,
            random_state=123
        )
        model.fit(X, y)
        
        # Validate comprehensive metadata preservation
        metadata = model.metadata
        
        # Clinical configuration preserved
        assert metadata['algorithm'] == 'Random Forest Regression'
        assert metadata['n_estimators_used'] == 25
        assert metadata['max_depth_used'] == 6
        
        # Training details preserved
        assert metadata['training_samples'] == 200
        assert metadata['feature_count'] == 1
        assert metadata['training_completed'] is True
        
        # Healthcare context preserved
        assert 'TAT prediction' in metadata['healthcare_optimization']
        assert metadata['clinical_deployment_ready'] is True
        
    def test_ensemble_consistency_validation(self):
        """Test Random Forest ensemble prediction consistency."""
        np.random.seed(42)
        X = pd.DataFrame({
            'consistent_feature': np.random.gamma(2, 2, 400)
        })
        y = pd.Series(12 + X['consistent_feature'] * 6 + np.random.exponential(2, 400))
        
        # Train multiple models with same configuration
        model1 = RandomForestTATRegressor(n_estimators=50, random_state=42)
        model2 = RandomForestTATRegressor(n_estimators=50, random_state=42)
        
        model1.fit(X, y)
        model2.fit(X, y)
        
        # Test prediction consistency
        test_X = X[:20]
        pred1 = model1.predict(test_X)
        pred2 = model2.predict(test_X)
        
        # With same random_state, predictions should be very close (allowing for floating point precision)
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)
        
    def test_clinical_interpretability_validation(self):
        """Test Random Forest clinical interpretability and transparency."""
        np.random.seed(42)
        X = pd.DataFrame({
            'critical_clinical_factor': np.random.exponential(3, 300),
            'secondary_clinical_factor': np.random.gamma(2, 2, 300),
            'operational_factor': np.random.uniform(1, 10, 300)
        })
        
        # Clear clinical relationships
        y = (15 + 
             X['critical_clinical_factor'] * 8 +      # Strong clinical impact
             X['secondary_clinical_factor'] * 3 +     # Moderate impact  
             X['operational_factor'] * 1 +            # Minor impact
             np.random.exponential(3, 300))
        
        model = RandomForestTATRegressor(n_estimators=100, random_state=42)
        model.fit(X, pd.Series(y))
        
        # Validate clinical interpretability through feature importance
        importance = model.get_feature_importance()
        
        # Critical factor should have highest importance
        top_feature = importance.iloc[0]['feature']
        assert 'critical_clinical_factor' == top_feature
        
        # Generate clinical insights
        insights = model.get_clinical_insights()
        
        # Validate clinical insight completeness
        assert len(insights['critical_bottlenecks']) > 0
        assert len(insights['intervention_recommendations']) > 0
        assert insights['interpretability_confidence'].startswith('High')
        
    def test_healthcare_compliance_validation(self):
        """Test Random Forest healthcare regulatory compliance features."""
        model = RandomForestTATRegressor(random_state=42)
        
        # Validate regulatory compliance attributes
        assert 'regulatory_compliance' in model.metadata
        compliance_info = model.metadata['regulatory_compliance']
        assert 'Interpretable' in compliance_info
        assert 'tree-based' in compliance_info
        
        # Validate audit trail capabilities
        assert 'deployment_readiness' in model.metadata
        deployment_info = model.metadata['deployment_readiness']
        assert 'MLOps' in deployment_info
        assert 'monitoring' in deployment_info