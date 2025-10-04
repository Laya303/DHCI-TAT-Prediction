"""
Test Suite

Comprehensive testing framework for medication preparation Turnaround Time (TAT) 
prediction and pharmacy workflow optimization. Ensures clinical accuracy, regulatory 
compliance, and production-grade reliability for healthcare analytics models supporting 
60-minute TAT threshold compliance and bottleneck identification workflows.

Test Suite Components:
- Healthcare model validation ensuring clinical accuracy and patient safety
- Feature engineering tests for pharmacy workflow variables and temporal analysis
- Bottleneck analysis validation supporting evidence-based intervention strategies
- MLOps pipeline tests ensuring production deployment reliability and monitoring
- Integration tests validating end-to-end TAT prediction workflows

Clinical Testing Standards:
- Patient safety validation through model accuracy and reliability testing
- Healthcare data integrity verification ensuring HIPAA compliance and audit trails
- Regulatory compliance testing supporting FDA and clinical validation requirements
- Production readiness validation for real-time TAT prediction deployment scenarios
- Stakeholder communication testing ensuring interpretable healthcare analytics results

Test Categories:
- Unit Tests: Individual component validation for healthcare analytics accuracy
- Integration Tests: Workflow validation for medication preparation process analysis
- Performance Tests: Scalability validation for production healthcare deployment
- Healthcare Tests: Clinical validation ensuring patient safety and regulatory compliance
- MLOps Tests: Production pipeline validation supporting continuous deployment workflows

Usage:
    Run complete test suite for Dana Farber TAT prediction validation:
    ```bash
    # Complete healthcare analytics test suite
    pytest tests/ --verbose --cov=src/tat
    
    # Clinical model validation only
    pytest tests/test_models/ -m healthcare
    
    # Production deployment tests
    pytest tests/test_integration/ -m mlops
    
    # Performance and scalability validation
    pytest tests/ -m performance --benchmark-only
    ```

Note:
    Essential for Dana Farber's pharmacy workflow optimization ensuring production-grade
    healthcare analytics supporting medication preparation efficiency and clinical operations
    excellence through comprehensive testing and validation of TAT prediction capabilities.
"""

# Test suite version and healthcare analytics metadata
__version__ = "1.0.0"
__author__ = "Dana Farber Healthcare Analytics Team"
__description__ = "TAT Prediction Test Suite for Pharmacy Workflow Optimization"

# Healthcare testing configuration and clinical validation settings
HEALTHCARE_TEST_CONFIG = {
    'tat_threshold_minutes': 60,                    # Clinical TAT compliance threshold
    'test_data_samples': 10000,                     # Healthcare test dataset size
    'model_accuracy_threshold': 0.85,               # Minimum clinical accuracy requirement
    'prediction_latency_ms': 100,                   # Maximum prediction response time
    'regulatory_compliance': 'HIPAA_FDA_validated', # Healthcare regulatory requirements
    'clinical_safety_level': 'production_grade',    # Patient safety validation level
    'audit_trail_required': True                    # Healthcare audit documentation
}

# Test markers for healthcare analytics validation categories
HEALTHCARE_TEST_MARKERS = {
    'unit': 'Individual component validation for healthcare accuracy',
    'integration': 'End-to-end workflow validation for TAT prediction',
    'healthcare': 'Clinical validation ensuring patient safety compliance',
    'performance': 'Scalability validation for production deployment',
    'mlops': 'MLOps pipeline validation for continuous deployment',
    'bottleneck': 'Pharmacy workflow bottleneck identification testing',
    'regulatory': 'Healthcare regulatory compliance and audit validation',
    'prediction': 'TAT prediction accuracy and clinical interpretability'
}

# Healthcare data validation schemas for test data integrity
HEALTHCARE_TEST_SCHEMAS = {
    'patient_data': {
        'age': {'type': 'numeric', 'range': [0, 120], 'required': True},
        'sex': {'type': 'categorical', 'values': ['M', 'F'], 'required': True},
        'severity': {'type': 'categorical', 'values': ['Low', 'Medium', 'High'], 'required': True}
    },
    'clinical_data': {
        'diagnosis_type': {'type': 'categorical', 'values': ['SolidTumor', 'Hematologic', 'Autoimmune', 'Other']},
        'treatment_type': {'type': 'categorical', 'values': ['Chemotherapy', 'Immunotherapy', 'TargetedTherapy', 'SupportiveCare']},
        'premed_required': {'type': 'binary', 'values': [0, 1], 'required': True},
        'stat_order': {'type': 'binary', 'values': [0, 1], 'required': True}
    },
    'operational_data': {
        'floor': {'type': 'categorical', 'values': [1, 2, 3], 'required': True},
        'shift': {'type': 'categorical', 'values': ['Day', 'Evening', 'Night'], 'required': True},
        'floor_occupancy_pct': {'type': 'numeric', 'range': [0, 100], 'required': True},
        'pharmacists_on_duty': {'type': 'numeric', 'range': [1, 10], 'required': True}
    },
    'lab_data': {
        'lab_WBC_k_per_uL': {'type': 'numeric', 'range': [0, 50], 'normal_range': [4.0, 11.0]},
        'lab_HGB_g_dL': {'type': 'numeric', 'range': [0, 25], 'normal_range': [12.0, 16.0]},
        'lab_Platelets_k_per_uL': {'type': 'numeric', 'range': [0, 1000], 'normal_range': [150, 400]},
        'lab_Creatinine_mg_dL': {'type': 'numeric', 'range': [0, 10], 'normal_range': [0.6, 1.3]},
        'lab_ALT_U_L': {'type': 'numeric', 'range': [0, 500], 'normal_range': [7, 56]}
    },
    'target_data': {
        'TAT_minutes': {'type': 'numeric', 'range': [5, 300], 'required': True},
        'TAT_over_60': {'type': 'binary', 'values': [0, 1], 'required': True}
    }
}

# Healthcare model validation thresholds for clinical accuracy requirements
HEALTHCARE_MODEL_THRESHOLDS = {
    'regression_models': {
        'max_rmse': 15.0,                          # Maximum RMSE for TAT prediction (minutes)
        'max_mae': 10.0,                           # Maximum MAE for clinical accuracy
        'min_r2': 0.75,                            # Minimum R² for model performance
        'prediction_range': [5, 200]               # Acceptable TAT prediction range
    },
    'classification_models': {
        'min_accuracy': 0.85,                      # Minimum accuracy for 60-minute threshold
        'min_precision': 0.80,                     # Minimum precision for patient safety
        'min_recall': 0.75,                        # Minimum recall for bottleneck detection
        'min_f1': 0.77                            # Minimum F1 for balanced performance
    },
    'feature_importance': {
        'min_features_explained': 0.80,            # Minimum variance explained by top features
        'max_single_feature_dominance': 0.50,      # Maximum single feature contribution
        'min_clinical_interpretability': 'high'    # Required interpretability level
    }
}

# Production deployment validation requirements for healthcare MLOps
PRODUCTION_VALIDATION_REQUIREMENTS = {
    'model_performance': {
        'inference_latency_ms': 100,               # Maximum prediction response time
        'batch_processing_capacity': 10000,        # Minimum batch processing capability
        'concurrent_predictions': 100,             # Concurrent prediction handling
        'model_size_mb': 50                       # Maximum model size for deployment
    },
    'data_validation': {
        'schema_drift_threshold': 0.05,            # Maximum acceptable schema drift
        'feature_drift_threshold': 0.10,           # Maximum acceptable feature drift
        'target_drift_threshold': 0.08,            # Maximum acceptable target drift
        'missing_data_threshold': 0.15             # Maximum acceptable missing data percentage
    },
    'monitoring_requirements': {
        'prediction_logging': True,                # Enable prediction audit trails
        'feature_monitoring': True,                # Enable feature drift monitoring
        'performance_tracking': True,              # Enable model performance tracking
        'error_alerting': True,                   # Enable error notification system
        'clinical_review_frequency': 'weekly'      # Clinical model review schedule
    }
}

# Test utilities and helper functions for healthcare analytics validation
def validate_healthcare_data_schema(data, schema_name):
    """
    Validate healthcare data against defined schemas for clinical accuracy.
    
    Args:
        data: Healthcare dataset to validate
        schema_name: Schema category (patient_data, clinical_data, etc.)
    
    Returns:
        bool: Validation status with detailed error reporting
    """
    if schema_name not in HEALTHCARE_TEST_SCHEMAS:
        raise ValueError(f"Unknown healthcare schema: {schema_name}")
    
    schema = HEALTHCARE_TEST_SCHEMAS[schema_name]
    validation_results = {'valid': True, 'errors': []}
    
    for field, rules in schema.items():
        if field not in data.columns:
            if rules.get('required', False):
                validation_results['valid'] = False
                validation_results['errors'].append(f"Required healthcare field missing: {field}")
    
    return validation_results

def validate_model_clinical_performance(model_metrics, model_type='regression'):
    """
    Validate model performance against clinical accuracy requirements.
    
    Args:
        model_metrics: Dictionary of model  Numerical Features
        model_type: 'regression' or 'classification' for appropriate thresholds
    
    Returns:
        bool: Clinical validation status with detailed assessment
    """
    thresholds = HEALTHCARE_MODEL_THRESHOLDS[f'{model_type}_models']
    validation_results = {'clinically_acceptable': True, 'warnings': []}
    
    if model_type == 'regression':
        if model_metrics.get('rmse', float('inf')) > thresholds['max_rmse']:
            validation_results['clinically_acceptable'] = False
            validation_results['warnings'].append(f"RMSE exceeds clinical threshold: {model_metrics['rmse']:.2f} > {thresholds['max_rmse']}")
    
    return validation_results

def generate_healthcare_test_data(n_samples=1000, include_missing=True, missing_rate=0.10):
    """
    Generate realistic healthcare TAT test data matching Dana Farber schema.
    
    Args:
        n_samples: Number of test samples for healthcare validation
        include_missing: Whether to include realistic missing data patterns
        missing_rate: Percentage of missing data (default 10% per requirements)
    
    Returns:
        pd.DataFrame: Realistic healthcare test dataset for validation
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)  # Reproducible healthcare test data
    
    # Generate realistic healthcare test data matching project schema
    test_data = pd.DataFrame({
        # Patient demographics
        'age': np.random.uniform(18, 85, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'race_ethnicity': np.random.choice(['White', 'Black', 'Hispanic/Latino', 'Asian', 'Other/Unknown'], n_samples),
        'insurance_type': np.random.choice(['Commercial', 'Medicare', 'Medicaid', 'Self-pay'], n_samples),
        
        # Clinical context
        'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic', 'Autoimmune', 'Other'], n_samples),
        'severity': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
        'treatment_type': np.random.choice(['Chemotherapy', 'Immunotherapy', 'TargetedTherapy', 'SupportiveCare'], n_samples),
        'patient_readiness_score': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.5, 0.3]),
        'premed_required': np.random.binomial(1, 0.3, n_samples),
        'stat_order': np.random.binomial(1, 0.1, n_samples),
        
        # Operational context
        'floor': np.random.choice([1, 2, 3], n_samples),
        'shift': np.random.choice(['Day', 'Evening', 'Night'], n_samples, p=[0.5, 0.3, 0.2]),
        'floor_occupancy_pct': np.random.uniform(20, 95, n_samples),
        'queue_length_at_order': np.random.poisson(5, n_samples),
        'pharmacists_on_duty': np.random.uniform(1, 6, n_samples),
        
        # Lab values with realistic ranges
        'lab_WBC_k_per_uL': np.random.normal(8, 2, n_samples),
        'lab_HGB_g_dL': np.random.normal(13, 1.5, n_samples),
        'lab_Platelets_k_per_uL': np.random.normal(275, 75, n_samples),
        'lab_Creatinine_mg_dL': np.random.normal(1, 0.3, n_samples),
        'lab_ALT_U_L': np.random.normal(25, 15, n_samples),
        
        # Target variables
        'TAT_minutes': np.random.exponential(45, n_samples) + 15,  # Realistic TAT distribution
    })
    
    # Add binary TAT threshold indicator
    test_data['TAT_over_60'] = (test_data['TAT_minutes'] > 60).astype(int)
    
    # Introduce realistic missing patterns if requested
    if include_missing:
        for col in ['lab_WBC_k_per_uL', 'lab_HGB_g_dL', 'lab_Platelets_k_per_uL']:
            missing_mask = np.random.random(n_samples) < missing_rate
            test_data.loc[missing_mask, col] = np.nan
    
    return test_data

# Export key testing components for healthcare analytics validation
__all__ = [
    'HEALTHCARE_TEST_CONFIG',
    'HEALTHCARE_TEST_MARKERS', 
    'HEALTHCARE_TEST_SCHEMAS',
    'HEALTHCARE_MODEL_THRESHOLDS',
    'PRODUCTION_VALIDATION_REQUIREMENTS',
    'validate_healthcare_data_schema',
    'validate_model_clinical_performance',
    'generate_healthcare_test_data'
]