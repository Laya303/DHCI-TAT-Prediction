"""
Temporal Feature Engineering for Pharmacy TAT Workflow Analysis

Comprehensive time-based feature engineering components for medication preparation
turnaround time analysis and healthcare workflow optimization. Provides robust
temporal feature extraction, delay calculation, and missing data imputation
specifically designed for pharmacy operations and clinical decision-making.

Core Components:
- DelayEngineer: Step-wise delay computation with sequential imputation strategies
- TemporalEngineer: Time-based pattern recognition and feature extraction  
- TimeReconstructor: Missing timestamp imputation for healthcare workflow integrity
- Healthcare-optimized temporal processing for production TAT analytics pipelines

Feature Categories:
- Step Delays: Inter-step processing times for bottleneck identification
- Temporal Patterns: Hour-of-day, day-of-week, shift patterns for operational insights
- Business Context: Work hours, weekend effects, holiday impacts on TAT performance
- Sequential Features: Workflow progression indicators and completion patterns

Production Features:
- Robust missing data handling for healthcare EHR integration challenges
- Scalable processing suitable for 100k+ order datasets in production
- MLOps integration with structured feature outputs for model training
- Clinical interpretability with healthcare domain knowledge embedded

Usage Example:
    from tat.features.temporal import DelayEngineer, TemporalEngineer
    
    # Step-wise delay computation with missing data handling
    delay_eng = DelayEngineer(impute_missing=True)
    delay_features = delay_eng.transform(tat_df)
    
    # Temporal pattern extraction for predictive modeling
    temporal_eng = TemporalEngineer()
    time_features = temporal_eng.add_time_features(delay_features)
    
    # Combined feature engineering pipeline
    complete_features = temporal_eng.fit_transform(tat_df)
"""

# Core temporal feature engineering components
from .delays import DelayEngineer
from .temporal import TemporalEngineer  
from .time_reconstruct import TimeReconstructor

# Module version for healthcare analytics pipeline tracking
__version__ = "1.0.0"

# Public API for temporal feature engineering in pharmacy TAT analysis
__all__ = [
    # Primary feature engineering classes
    'DelayEngineer',
    'TemporalEngineer',
    'TimeReconstructor',
    
    # Module metadata
    '__version__',
]

# Healthcare workflow configuration for temporal feature engineering
MEDICATION_PREP_WORKFLOW_STEPS = [
    ('doctor_order_time', 'nurse_validation_time', 'delay_order_to_validation'),
    ('nurse_validation_time', 'prep_complete_time', 'delay_validation_to_prep'), 
    ('prep_complete_time', 'second_validation_time', 'delay_prep_to_complete'),
    ('second_validation_time', 'floor_dispatch_time', 'delay_complete_to_dispatch'),
    ('floor_dispatch_time', 'patient_infusion_time', 'delay_dispatch_to_infusion'),
]

# Clinical time thresholds for healthcare feature engineering
BUSINESS_HOURS_START = 7  # 7 AM start for pharmacy operations
BUSINESS_HOURS_END = 19   # 7 PM end for standard pharmacy operations  
SHIFT_BOUNDARIES = [7, 15, 23]  # Day, Evening, Night shift boundaries

# Data quality parameters for healthcare temporal processing
DEFAULT_IMPUTATION_METHOD = 'sequential'  # Healthcare workflow-aware imputation
MAX_MISSING_TIMESTAMP_PCT = 15.0  # Acceptable missing data threshold
TEMPORAL_VALIDATION_TOLERANCE_MINUTES = 1.0  # Chronological sequence validation