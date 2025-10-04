"""
Shared test fixtures and configuration for TAT prediction tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
from unittest.mock import MagicMock, patch
import tempfile
import logging

# Import project configuration for consistent test data generation
from src.tat.config import (
    STEP_COLS, DELAY_PAIRS, LAB_COLS, LAB_NORMALS, LAB_CLIPS,
    CATEGORICAL_MAPPINGS, HEALTHCARE_THRESHOLDS, REQUIRED_COLUMNS,
    SHIFT_BOUNDARIES, CATEGORICAL_PREFIX_MAP
)

# Import model classes for testing
from src.tat.models.xgboost_model import XGBoostTATRegressor
from src.tat.models.random_forest_model import RandomForestTATRegressor
from src.tat.models.linear_model import RidgeTATRegressor

# Configure logging for test execution
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def healthcare_test_config():
    """
    Healthcare-specific test configuration for Dana Farber TAT validation.
    
    Provides comprehensive configuration supporting clinical validation requirements,
    regulatory compliance standards, and production deployment testing criteria
    ensuring robust healthcare analytics validation workflows.
    
    Returns:
        Dict[str, Any]: Healthcare test configuration with clinical validation parameters
    """
    return {
        # Clinical validation thresholds
        'tat_threshold_minutes': HEALTHCARE_THRESHOLDS['tat_target_minutes'],
        'model_accuracy_threshold': HEALTHCARE_THRESHOLDS['model_accuracy_minimum'],
        'bottleneck_threshold_minutes': HEALTHCARE_THRESHOLDS['bottleneck_significance_threshold'],
        
        # Test data generation parameters
        'test_samples_small': 500,        # Small dataset for unit testing
        'test_samples_medium': 2000,      # Medium dataset for integration testing
        'test_samples_large': 10000,      # Large dataset for performance testing
        'missing_data_rate': 0.10,        # 10% missing data for realistic scenarios
        
        # Clinical data validation
        'age_range': (18, 85),            # Adult patient population
        'tat_realistic_range': (10, 180), # Clinically plausible TAT range
        'lab_missing_rate': 0.15,         # Realistic lab data missing rate
        
        # Production testing requirements
        'prediction_latency_target_ms': 100,  # Real-time prediction requirement
        'batch_processing_target': 1000,      # Minimum batch processing capacity
        'model_accuracy_regression_rmse': 15.0, # Maximum acceptable RMSE (minutes)
        'model_accuracy_classification': 0.85,  # Minimum classification accuracy
        
        # Healthcare regulatory compliance
        'hipaa_compliance_required': True,     # HIPAA privacy validation
        'audit_trail_required': True,          # Healthcare audit documentation
        'clinical_interpretability': 'high'    # Required model interpretability level
    }

@pytest.fixture
def sample_tat_data(healthcare_test_config):
    """
    Generate realistic TAT dataset matching Dana Farber operational patterns.
    
    Creates comprehensive healthcare dataset including patient demographics,
    clinical context, operational variables, laboratory values, and realistic
    TAT outcomes supporting thorough model validation and bottleneck analysis.
    
    Args:
        healthcare_test_config: Healthcare validation configuration parameters
        
    Returns:
        pd.DataFrame: Realistic TAT dataset for healthcare analytics validation
    """
    np.random.seed(42)  # Reproducible test data generation
    n_samples = healthcare_test_config['test_samples_medium']
    
    # Generate realistic patient demographics matching healthcare populations
    demographics_data = {
        'age': np.clip(np.random.normal(65, 15, n_samples), 18, 95),  # Oncology patient age distribution
        'sex': np.random.choice(['F', 'M'], n_samples, p=[0.52, 0.48]),  # Realistic gender distribution
        'race_ethnicity': np.random.choice(
            ['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other/Unknown'], 
            n_samples, p=[0.65, 0.15, 0.10, 0.07, 0.03]  # US healthcare demographics
        ),
        'insurance_type': np.random.choice(
            ['Commercial', 'Medicare', 'Medicaid', 'Self-pay'], 
            n_samples, p=[0.45, 0.35, 0.15, 0.05]  # Healthcare insurance distribution
        )
    }
    
    # Generate clinical context variables reflecting oncology care complexity
    clinical_data = {
        'diagnosis_type': np.random.choice(
            ['SolidTumor', 'Hematologic', 'Autoimmune', 'Other'], 
            n_samples, p=[0.60, 0.25, 0.10, 0.05]  # Dana Farber diagnosis distribution
        ),
        'severity': np.random.choice(
            ['Low', 'Medium', 'High'], 
            n_samples, p=[0.30, 0.50, 0.20]  # Clinical severity distribution
        ),
        'treatment_type': np.random.choice(
            ['Chemotherapy', 'Immunotherapy', 'TargetedTherapy', 'SupportiveCare'], 
            n_samples, p=[0.45, 0.25, 0.20, 0.10]  # Treatment modality distribution
        ),
        'patient_readiness_score': np.random.choice(
            [1, 2, 3], n_samples, p=[0.15, 0.70, 0.15]  # Patient preparation status
        ),
        'premed_required': np.random.binomial(1, 0.35, n_samples),  # Pre-medication requirement
        'stat_order': np.random.binomial(1, 0.08, n_samples)        # STAT order urgency
    }
    
    # Generate operational context variables affecting TAT workflow
    operational_data = {
        'floor': np.random.choice([1, 2, 3], n_samples, p=[0.40, 0.35, 0.25]),  # Treatment floor distribution
        'shift': np.random.choice(['Day', 'Evening', 'Night'], n_samples, p=[0.55, 0.30, 0.15]),  # Shift distribution
        'floor_occupancy_pct': np.clip(np.random.beta(6, 2, n_samples) * 100, 20, 98),  # Realistic occupancy
        'queue_length_at_order': np.random.poisson(4.5, n_samples),  # Workload queue realistic distribution
        'nurse_credential': np.random.choice(
            ['RN', 'BSN', 'MSN', 'NP'], 
            n_samples, p=[0.25, 0.45, 0.25, 0.05]  # Nursing credential distribution
        ),
        'pharmacist_credential': np.random.choice(
            ['RPh', 'PharmD', 'BCOP'], 
            n_samples, p=[0.15, 0.65, 0.20]  # Pharmacy credential distribution
        ),
        'nurse_employment_years': np.clip(np.random.exponential(8, n_samples), 0, 35),  # Experience distribution
        'pharmacist_employment_years': np.clip(np.random.exponential(10, n_samples), 0, 40)  # Experience distribution
    }
    
    # Generate realistic staffing levels affecting workflow efficiency
    staffing_data = {
        'pharmacists_on_duty': np.random.uniform(2, 8, n_samples),  # Staffing level variation
        'nurses_on_duty': np.random.uniform(8, 20, n_samples),      # Floor nursing staff
        'nurse_workload': np.random.uniform(2, 12, n_samples)       # Individual nurse patient load
    }
    
    # Generate realistic laboratory values with clinical context
    lab_data = {}
    for lab_col in LAB_COLS:
        if lab_col in LAB_NORMALS:
            low, high = LAB_NORMALS[lab_col]
            # Generate values centered on normal range with realistic variation
            lab_data[lab_col] = np.random.normal((low + high) / 2, (high - low) / 6, n_samples)
            
            # Apply clinical clipping bounds
            if lab_col in LAB_CLIPS:
                clip_low, clip_high = LAB_CLIPS[lab_col]
                lab_data[lab_col] = np.clip(lab_data[lab_col], clip_low, clip_high)
    
    # Combine all data components
    test_data = pd.DataFrame({
        **demographics_data,
        **clinical_data,
        **operational_data,
        **staffing_data,
        **lab_data
    })
    
    # Add unique identifiers for healthcare analytics
    test_data['order_id'] = [f"ORD_{i:06d}" for i in range(n_samples)]
    test_data['patient_id'] = [f"PT_{i:05d}" for i in np.random.randint(10000, 99999, n_samples)]
    test_data['ordering_physician'] = [f"MD_{i:03d}" for i in np.random.randint(100, 300, n_samples)]
    
    # Generate realistic TAT outcomes with clinical workflow dependencies
    base_tat = 35  # Base TAT for standard workflow
    
    # Clinical complexity factors affecting TAT
    complexity_adjustment = (
        (test_data['severity'] == 'High') * 8 +        # High severity adds processing time
        (test_data['severity'] == 'Medium') * 3 +      # Medium severity moderate increase
        test_data['premed_required'] * 12 +            # Pre-medication adds significant time
        test_data['stat_order'] * (-15) +              # STAT orders get priority (faster)
        (test_data['treatment_type'] == 'Chemotherapy') * 5  # Chemotherapy complexity
    )
    
    # Operational workflow factors
    operational_adjustment = (
        (test_data['pharmacists_on_duty'] - 4) * (-2) +     # More pharmacists = faster
        (test_data['floor_occupancy_pct'] - 70) * 0.15 +    # Higher occupancy = slower
        (test_data['queue_length_at_order'] - 4) * 1.8 +    # Longer queue = delays
        (test_data['shift'] == 'Night') * 8 +               # Night shift slower
        (test_data['shift'] == 'Evening') * 4               # Evening shift moderate delay
    )
    
    # Staff experience factors
    experience_adjustment = (
        -((test_data['nurse_employment_years'] - 5) * 0.3).clip(-5, 2) +      # Experience helps
        -((test_data['pharmacist_employment_years'] - 8) * 0.4).clip(-6, 3)   # Pharmacy experience
    )
    
    # Calculate realistic TAT with stochastic variation
    predicted_tat = (base_tat + complexity_adjustment + operational_adjustment + 
                    experience_adjustment + np.random.exponential(12, n_samples))
    
    # Ensure clinically realistic TAT bounds
    test_data['TAT_minutes'] = np.clip(predicted_tat, 8, 200)
    
    # Create binary threshold indicator for classification
    test_data['TAT_over_60'] = (test_data['TAT_minutes'] > healthcare_test_config['tat_threshold_minutes']).astype(int)
    
    # Introduce realistic missing data patterns
    missing_rate = healthcare_test_config['missing_data_rate']
    
    # Lab values most commonly missing
    for lab_col in LAB_COLS:
        missing_mask = np.random.random(n_samples) < healthcare_test_config['lab_missing_rate']
        test_data.loc[missing_mask, lab_col] = np.nan
    
    # Employment years occasionally missing
    for exp_col in ['nurse_employment_years', 'pharmacist_employment_years']:
        missing_mask = np.random.random(n_samples) < 0.05
        test_data.loc[missing_mask, exp_col] = np.nan
    
    logger.info(f"Generated realistic healthcare TAT dataset: {len(test_data):,} samples, "
                f"{test_data['TAT_over_60'].mean():.1%} exceed 60-minute threshold")
    
    return test_data

@pytest.fixture
def sample_tat_data_small(healthcare_test_config):
    """
    Generate small TAT dataset for unit testing and rapid validation.
    
    Returns:
        pd.DataFrame: Small realistic TAT dataset for unit testing
    """
    np.random.seed(42)
    n_samples = healthcare_test_config['test_samples_small']
    
    # Simplified but realistic data generation for unit testing
    test_data = pd.DataFrame({
        'age': np.random.uniform(18, 85, n_samples),
        'sex': np.random.choice(['F', 'M'], n_samples),
        'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic'], n_samples),
        'severity': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'floor': np.random.choice([1, 2, 3], n_samples),
        'shift': np.random.choice(['Day', 'Evening', 'Night'], n_samples),
        'pharmacists_on_duty': np.random.uniform(2, 6, n_samples),
        'floor_occupancy_pct': np.random.uniform(30, 90, n_samples),
        'premed_required': np.random.binomial(1, 0.3, n_samples),
        'stat_order': np.random.binomial(1, 0.1, n_samples),
        'TAT_minutes': np.random.exponential(40, n_samples) + 15
    })
    
    test_data['TAT_over_60'] = (test_data['TAT_minutes'] > 60).astype(int)
    return test_data

@pytest.fixture
def clinical_validation_data():
    """
    Generate edge case and clinical validation datasets for comprehensive testing.
    
    Returns:
        Dict[str, pd.DataFrame]: Clinical validation scenarios for edge case testing
    """
    np.random.seed(42)
    
    validation_scenarios = {}
    
    # Scenario 1: All high-complexity cases (should have longer TAT)
    high_complexity_data = pd.DataFrame({
        'age': [75, 80, 68, 72, 85],
        'sex': ['F', 'M', 'F', 'M', 'F'],
        'diagnosis_type': ['Hematologic'] * 5,
        'severity': ['High'] * 5,
        'floor': [1, 2, 3, 1, 2],
        'shift': ['Night', 'Evening', 'Night', 'Day', 'Evening'],
        'pharmacists_on_duty': [2.0, 2.5, 2.0, 3.0, 2.5],
        'floor_occupancy_pct': [85, 90, 88, 92, 87],
        'premed_required': [1, 1, 1, 1, 1],
        'stat_order': [0, 0, 0, 0, 0],
        'TAT_minutes': [95, 105, 88, 78, 92]  # All exceed 60-minute threshold
    })
    high_complexity_data['TAT_over_60'] = 1
    validation_scenarios['high_complexity'] = high_complexity_data
    
    # Scenario 2: All low-complexity cases (should have shorter TAT)
    low_complexity_data = pd.DataFrame({
        'age': [45, 52, 38, 41, 49],
        'sex': ['M', 'F', 'M', 'F', 'M'],
        'diagnosis_type': ['SolidTumor'] * 5,
        'severity': ['Low'] * 5,
        'floor': [1, 1, 2, 2, 3],
        'shift': ['Day', 'Day', 'Day', 'Day', 'Evening'],
        'pharmacists_on_duty': [5.0, 6.0, 5.5, 6.5, 5.0],
        'floor_occupancy_pct': [45, 50, 40, 55, 48],
        'premed_required': [0, 0, 0, 0, 0],
        'stat_order': [1, 0, 1, 0, 0],  # Some STAT orders for faster processing
        'TAT_minutes': [28, 35, 25, 42, 38]  # All under 60-minute threshold
    })
    low_complexity_data['TAT_over_60'] = 0
    validation_scenarios['low_complexity'] = low_complexity_data
    
    # Scenario 3: Missing data patterns for robustness testing
    missing_data_scenario = pd.DataFrame({
        'age': [65, np.nan, 72, 68, np.nan],
        'sex': ['F', 'M', None, 'F', 'M'],
        'diagnosis_type': ['SolidTumor', None, 'Hematologic', 'SolidTumor', None],
        'severity': ['Medium', 'High', None, 'Low', 'Medium'],
        'floor': [1, 2, 3, np.nan, 2],
        'shift': ['Day', None, 'Evening', 'Day', None],
        'pharmacists_on_duty': [4.0, np.nan, 3.5, 5.0, np.nan],
        'floor_occupancy_pct': [65, 78, np.nan, 55, 82],
        'premed_required': [1, 0, np.nan, 0, 1],
        'stat_order': [0, np.nan, 0, 1, 0],
        'TAT_minutes': [55, 78, 45, 32, 68]
    })
    missing_data_scenario['TAT_over_60'] = (missing_data_scenario['TAT_minutes'] > 60).astype(int)
    validation_scenarios['missing_data'] = missing_data_scenario
    
    return validation_scenarios

@pytest.fixture
def trained_xgb_model(sample_tat_data_small):
    """
    Pre-trained XGBoost model for testing prediction and analysis capabilities.
    
    Args:
        sample_tat_data_small: Small training dataset for model fitting
        
    Returns:
        XGBoostTATRegressor: Trained XGBoost model ready for testing
    """
    # Prepare training data
    feature_cols = [col for col in sample_tat_data_small.columns 
                   if col not in ['TAT_minutes', 'TAT_over_60']]
    X_train = sample_tat_data_small[feature_cols]
    y_train = sample_tat_data_small['TAT_minutes']
    
    # Initialize and train XGBoost model with test configuration
    xgb_model = XGBoostTATRegressor(
        random_state=42,
        n_estimators=50,  # Smaller for faster testing
        max_depth=4,      # Conservative for testing
        learning_rate=0.1
    )
    
    # Fit model on training data
    xgb_model.fit(X_train, y_train)
    
    return xgb_model

@pytest.fixture
def trained_rf_model(sample_tat_data_small):
    """
    Pre-trained Random Forest model for testing ensemble capabilities.
    
    Args:
        sample_tat_data_small: Small training dataset for model fitting
        
    Returns:
        RandomForestTATRegressor: Trained Random Forest model ready for testing
    """
    # Prepare training data
    feature_cols = [col for col in sample_tat_data_small.columns 
                   if col not in ['TAT_minutes', 'TAT_over_60']]
    X_train = sample_tat_data_small[feature_cols]
    y_train = sample_tat_data_small['TAT_minutes']
    
    # Initialize and train Random Forest model
    rf_model = RandomForestTATRegressor(
        random_state=42,
        n_estimators=50,  # Smaller for faster testing
        max_depth=5       # Conservative for testing
    )
    
    # Fit model on training data
    rf_model.fit(X_train, y_train)
    
    return rf_model

@pytest.fixture
def trained_linear_model(sample_tat_data_small):
    """
    Pre-trained Linear model for testing interpretability capabilities.
    
    Args:
        sample_tat_data_small: Small training dataset for model fitting
        
    Returns:
        LinearTATRegressor: Trained Linear model ready for testing
    """
    # Prepare training data
    feature_cols = [col for col in sample_tat_data_small.columns 
                   if col not in ['TAT_minutes', 'TAT_over_60']]
    X_train = sample_tat_data_small[feature_cols]
    y_train = sample_tat_data_small['TAT_minutes']
    
    # Initialize and train Linear model
    linear_model = RidgeTATRegressor(random_state=42)
    
    # Fit model on training data
    linear_model.fit(X_train, y_train)
    
    return linear_model

@pytest.fixture
def model_comparison_suite(trained_xgb_model, trained_rf_model, trained_linear_model):
    """
    Suite of trained models for comparative testing and ensemble validation.
    
    Returns:
        Dict[str, Any]: Dictionary of trained models for comprehensive testing
    """
    return {
        'xgboost': trained_xgb_model,
        'random_forest': trained_rf_model,
        'linear': trained_linear_model
    }

@pytest.fixture
def temp_data_directory():
    """
    Temporary directory for testing file I/O operations.
    
    Returns:
        Path: Temporary directory path for test data storage
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create subdirectories matching project structure
        (temp_path / "raw").mkdir()
        (temp_path / "processed").mkdir()
        (temp_path / "models").mkdir()
        
        yield temp_path

@pytest.fixture
def mock_logger():
    """
    Mock logger for testing logging functionality without output.
    
    Returns:
        MagicMock: Mock logger for testing log calls
    """
    return MagicMock()

@pytest.fixture
def performance_benchmark_data(healthcare_test_config):
    """
    Large dataset for performance and scalability testing.
    
    Returns:
        pd.DataFrame: Large healthcare dataset for performance validation
    """
    np.random.seed(42)
    n_samples = healthcare_test_config['test_samples_large']
    
    # Generate large-scale dataset for performance testing
    performance_data = pd.DataFrame({
        'age': np.random.uniform(18, 85, n_samples),
        'sex': np.random.choice(['F', 'M'], n_samples),
        'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic', 'Autoimmune'], n_samples),
        'severity': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'floor': np.random.choice([1, 2, 3], n_samples),
        'shift': np.random.choice(['Day', 'Evening', 'Night'], n_samples),
        'pharmacists_on_duty': np.random.uniform(2, 8, n_samples),
        'floor_occupancy_pct': np.random.uniform(20, 95, n_samples),
        'premed_required': np.random.binomial(1, 0.3, n_samples),
        'stat_order': np.random.binomial(1, 0.1, n_samples),
        'TAT_minutes': np.random.exponential(40, n_samples) + 10
    })
    
    performance_data['TAT_over_60'] = (performance_data['TAT_minutes'] > 60).astype(int)
    
    return performance_data

@pytest.fixture(autouse=True)
def setup_test_logging():
    """
    Automatically configure logging for all tests.
    
    Ensures consistent logging behavior across test execution.
    """
    # Configure test-specific logging
    logging.getLogger('src.tat').setLevel(logging.WARNING)  # Reduce noise during testing
    
    # Yield control to test
    yield
    
    # Cleanup after test
    logging.getLogger('src.tat').setLevel(logging.INFO)  # Reset to default

# Test utility functions available to all test modules
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error for TAT prediction validation."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

@pytest.fixture
def healthcare_tat_dataset(sample_tat_data):
    """
    Healthcare TAT dataset formatted for model training and testing.
    
    Provides feature matrix (X) and target variable (y) extracted from
    comprehensive healthcare TAT dataset supporting model validation,
    prediction accuracy testing, and clinical performance evaluation.
    
    Args:
        sample_tat_data: Complete healthcare TAT dataset with all variables
        
    Returns:
        Dict[str, Any]: Dataset dictionary with 'X' (features) and 'y' (target)
        for healthcare model training and validation workflows.
    """
    # Extract feature columns (exclude target and identifier columns)
    feature_cols = [col for col in sample_tat_data.columns 
                   if col not in ['TAT_minutes', 'TAT_over_60', 'order_id', 'patient_id', 'ordering_physician']]
    
    X = sample_tat_data[feature_cols].copy()
    y = sample_tat_data['TAT_minutes'].copy()
    
    # Apply categorical encodings to match config.py mappings
    categorical_mappings = {
        'sex': {"F": 0, "M": 1},
        'severity': {"Low": 0, "Medium": 1, "High": 2},
        'nurse_credential': {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3},
        'pharmacist_credential': {"RPh": 0, "PharmD": 1, "BCOP": 2},
    }
    
    # Encode categorical variables that have defined mappings
    for col, mapping in categorical_mappings.items():
        if col in X.columns:
            X[col] = X[col].map(mapping)
    
    # Use label encoding for other categorical columns
    categorical_cols = ['race_ethnicity', 'insurance_type', 'diagnosis_type', 
                       'treatment_type', 'shift']
    
    for col in categorical_cols:
        if col in X.columns:
            X[col] = pd.Categorical(X[col]).codes
    
    # Handle missing values by filling with median for numerical columns
    # and mode for categorical columns (already encoded as integers)
    for col in X.columns:
        if X[col].dtype in ['object']:  # Any remaining string columns
            X[col] = pd.Categorical(X[col]).codes
        if X[col].isnull().any():
            if X[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
    
    return {
        'X': X,
        'y': y,
        'feature_names': feature_cols,
        'target_name': 'TAT_minutes'
    }

def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error for clinical accuracy assessment."""
    return np.mean(np.abs(y_true - y_pred))

def calculate_threshold_accuracy(y_true, y_pred, threshold=60):
    """Calculate accuracy for TAT threshold classification."""
    true_binary = (y_true > threshold).astype(int)
    pred_binary = (y_pred > threshold).astype(int)
    return (true_binary == pred_binary).mean()

def validate_clinical_predictions(predictions, min_value=5, max_value=300):
    """Validate that TAT predictions fall within clinically plausible ranges."""
    return np.all((predictions >= min_value) & (predictions <= max_value))

# Export test utilities for use across test modules
__all__ = [
    'calculate_rmse',
    'calculate_mae', 
    'calculate_threshold_accuracy',
    'validate_clinical_predictions'
]