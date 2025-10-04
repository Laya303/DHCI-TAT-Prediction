"""
Configuration constants for TAT prediction system.

Defines step columns, target variables, laboratory ranges, categorical mappings,
and training configurations for medication preparation workflow analysis.
"""
from typing import List, Dict, Optional, Tuple

# Sequential medication preparation workflow step columns (order matters for TAT analysis)
STEP_COLS: List[str] = [
    "nurse_validation_time",      # Initial nursing validation and order review
    "prep_complete_time",         # Pharmacy preparation completion timestamp
    "second_validation_time",     # Secondary validation and quality assurance
    "floor_dispatch_time",        # Medication dispatch to treatment floor
    "patient_infusion_time",      # Final medication administration to patient
]

# Step-to-step delay calculation pairs for bottleneck identification
# Format: (previous_step, current_step, delay_feature_name)
DELAY_PAIRS: List[Tuple[Optional[str], str, str]] = [
    (None, "nurse_validation_time_mins_unwrapped", "delay_order_to_nurse"),           # Order → Nursing
    ("nurse_validation_time_mins_unwrapped", "prep_complete_time_mins_unwrapped", "delay_nurse_to_prep"),          # Nursing → Preparation  
    ("prep_complete_time_mins_unwrapped", "second_validation_time_mins_unwrapped", "delay_prep_to_second"),        # Preparation → Validation
    ("second_validation_time_mins_unwrapped", "floor_dispatch_time_mins_unwrapped", "delay_second_to_dispatch"),   # Validation → Dispatch
    ("floor_dispatch_time_mins_unwrapped", "patient_infusion_time_mins_unwrapped", "delay_dispatch_to_infusion"), # Dispatch → Administration
]

# Temporary column suffixes for preprocessing cleanup
HELPER_SUFFIXES: Tuple[str, ...] = ("_mins_unwrapped", "_dt")

# Sequential delay visualization order for healthcare stakeholder reporting
DELAY_PLOT_ORDER: List[str] = [
    "delay_order_to_nurse",        # Initial workflow bottleneck assessment
    "delay_nurse_to_prep",         # Nursing-to-pharmacy handoff analysis
    "delay_prep_to_second",        # Preparation completion efficiency
    "delay_second_to_dispatch",    # Quality validation workflow timing
    "delay_dispatch_to_infusion",  # Floor delivery and administration delays
]

# Healthcare shift boundaries for temporal analysis (24-hour format)
SHIFT_BOUNDARIES = {
    "Day": (7, 15),      # 07:00-14:59 - Primary care hours with full staffing
    "Evening": (15, 23), # 15:00-22:59 - Transition period with reduced resources
    "Night": (23, 7)     # 23:00-06:59 - Overnight coverage with minimal staffing
}

# Business operations configuration for workflow analysis
BUSINESS_HOURS: Tuple[int, int] = (9, 17)  # 09:00-16:59 - Standard healthcare operations
WEEKEND_DAYS: Tuple[int, ...] = (5, 6)     # Saturday and Sunday - Reduced staffing periods

# Non-feature identifier columns excluded from model training
NON_FEATURE_COLS: List[str] = [
    "order_id",           # Unique medication order identifier
    "patient_id",         # Patient identifier (privacy and uniqueness)
    "ordering_physician"  # Physician identifier (high cardinality categorical)
]

# Primary prediction targets for TAT analysis and threshold compliance
TARGETS: List[str] = [
    "TAT_minutes",    # Continuous TAT prediction for workflow optimization
    "TAT_over_60"     # Binary threshold classification for quality compliance
]

# Canonical timestamp column anchoring all temporal calculations
ORDER_TIME_COL: str = "doctor_order_time"

# Healthcare categorical encoding prefixes for feature naming consistency
CATEGORICAL_PREFIX_MAP: Dict[str, str] = {
    'sex': 'sex',                    # Patient demographic encoding
    'race_ethnicity': 'race',        # Demographic analysis prefix
    'insurance_type': 'ins',         # Insurance category encoding
    'diagnosis_type': 'dx',          # Clinical diagnosis classification
    'severity': 'sev',               # Clinical severity assessment
    'treatment_type': 'tx',          # Treatment modality classification
    'nurse_credential': 'nurse',     # Nursing credential hierarchy
    'pharmacist_credential': 'pharm', # Pharmacy credential specialization
    'ordering_department': 'dept',    # Clinical department classification
    'shift': 'shift',                # Temporal shift period encoding
    'floor': 'floor',                # Treatment facility location
}

# Clinical categorical value mappings with healthcare hierarchy
CATEGORICAL_MAPPINGS = {
    'sex': {
        "F": 0,  # Female patient demographic
        "M": 1   # Male patient demographic
    },
    'severity': {
        "Low": 0,     # Low clinical complexity - routine workflow
        "Medium": 1,  # Moderate complexity - standard protocols
        "High": 2     # High complexity - expedited processing required
    },
    'nurse_credential': {
        "RN": 0,   # Registered Nurse - basic clinical competency
        "BSN": 1,  # Bachelor's prepared - enhanced clinical skills
        "MSN": 2,  # Master's prepared - advanced clinical expertise
        "NP": 3    # Nurse Practitioner - advanced practice capability
    },
    'pharmacist_credential': {
        "RPh": 0,     # Registered Pharmacist - foundational practice
        "PharmD": 1,  # Doctor of Pharmacy - clinical practice standard
        "BCOP": 2     # Board Certified Clinical - specialized expertise
    }, 
}

# Laboratory test columns for clinical context integration
LAB_COLS: List[str] = [
    "lab_WBC_k_per_uL",      # White Blood Cell count - infection/immune status
    "lab_HGB_g_dL",          # Hemoglobin - oxygen carrying capacity
    "lab_Platelets_k_per_uL", # Platelet count - bleeding risk assessment
    "lab_Creatinine_mg_dL",   # Creatinine - kidney function evaluation
    "lab_ALT_U_L",           # ALT - liver function assessment
]

# Clinical normal ranges for laboratory values (healthcare standard references)
LAB_NORMALS: Dict[str, Tuple[float, float]] = {
    "lab_WBC_k_per_uL": (4.0, 11.0),        # Normal WBC range (×10³/μL)
    "lab_HGB_g_dL": (12.0, 16.0),           # Normal hemoglobin range (g/dL)
    "lab_Platelets_k_per_uL": (150.0, 400.0), # Normal platelet range (×10³/μL)
    "lab_Creatinine_mg_dL": (0.6, 1.3),      # Normal creatinine range (mg/dL)
    "lab_ALT_U_L": (7.0, 56.0),             # Normal ALT range (U/L)
}

# Defensive clipping ranges preventing extreme outliers in clinical data
LAB_CLIPS: Dict[str, Tuple[float, float]] = {
    "lab_WBC_k_per_uL": (0, 200),      # Physiologically plausible WBC bounds
    "lab_HGB_g_dL": (0, 25),           # Hemoglobin survival compatibility range
    "lab_Platelets_k_per_uL": (0, 600), # Platelet count clinical extremes
    "lab_Creatinine_mg_dL": (0, 20),    # Creatinine dialysis consideration range
    "lab_ALT_U_L": (0, 200),           # ALT hepatotoxicity assessment range
}

# Essential columns required for healthcare analytics pipeline validation
REQUIRED_COLUMNS: set[str] = {
    ORDER_TIME_COL,              # Temporal anchor for TAT calculations
    'age',                       # Patient demographic for complexity assessment
    'sex',                       # Demographic factor for clinical context
    'race_ethnicity',           # Demographic analysis for healthcare equity
    'insurance_type',           # Operational factor affecting workflow
    'diagnosis_type',           # Clinical classification for complexity assessment
    'severity',                 # Clinical severity impacting processing priority
    'treatment_type',           # Treatment modality affecting preparation complexity
    'patient_readiness_score',  # Patient preparation status for scheduling
    'premed_required',          # Pre-medication requirement flag
    'stat_order',               # Urgency flag for expedited processing
    'floor',                    # Treatment location for logistics planning
    'shift',                    # Temporal context for staffing analysis
    'floor_occupancy_pct',      # Operational load affecting workflow efficiency
    'queue_length_at_order',    # Workload context for delay prediction
    'nurse_credential',         # Staff competency affecting processing efficiency
    'pharmacist_credential'     # Pharmacy expertise affecting preparation timing
}

# Data quality and cleaning configuration for healthcare data integrity
CLEANING_CONFIG = {
    'age_bounds': (0, 120),          # Physiologically reasonable patient age range
    'years_bounds': (0, 40),         # Professional experience range validation
    'years_cols': [                  # Employment experience columns for validation
        "nurse_employment_years",
        "pharmacist_employment_years"
    ],
    'binary_cols': [                 # Boolean flags requiring 0/1 validation
        "premed_required",           # Pre-medication requirement flag
        "stat_order"                 # STAT order urgency indicator
    ]
}

# Multi-configuration model training setups for diverse healthcare scenarios
TRAINING_CONFIGS = {
    'linear_interpretable': {
        'scaling_strategy': 'linear',
        'description': 'Linear models optimized for clinical interpretability and coefficient analysis',
        'focus': 'Healthcare stakeholder communication through transparent linear relationships',
        'clinical_applications': [
            'Clinical workflow coefficient interpretation',
            'Evidence-based policy development through linear insights',
            'Healthcare stakeholder communication with transparent predictions'
        ],
        'deployment_scenarios': [
            'Regulatory compliance requiring model explainability',
            'Clinical committee review and approval workflows',
            'Healthcare quality improvement committee presentations'
        ]
    },
    'tree_based_models': {
        'scaling_strategy': 'tree',
        'description': 'Tree-based ensemble models optimized for healthcare feature interactions',
        'focus': 'Complex workflow pattern recognition and non-linear bottleneck identification',
        'clinical_applications': [
            'Advanced bottleneck detection through feature interactions',
            'Operational complexity modeling for diverse clinical scenarios',
            'Workflow optimization through non-linear pattern recognition'
        ],
        'deployment_scenarios': [
            'Real-time TAT prediction for operational decision-making',
            'Automated bottleneck alerts and workflow recommendations',
            'Resource allocation optimization based on predictive insights'
        ]
    },
    'comprehensive_ensemble': {
        'scaling_strategy': 'mixed',
        'description': 'Full ensemble approach maximizing predictive performance for production deployment',
        'focus': 'Maximum accuracy and robustness for critical healthcare operations',
        'clinical_applications': [
            'Production TAT prediction with maximum accuracy requirements',
            'Critical workflow decisions requiring highest confidence predictions',
            'Comprehensive bottleneck analysis across diverse operational scenarios'
        ],
        'deployment_scenarios': [
            'Production healthcare analytics with performance guarantees',
            'Mission-critical workflow optimization requiring maximum reliability',
            'Comprehensive analytics supporting C-suite healthcare decision-making'
        ]
    }
}

# Healthcare quality thresholds and performance standards
HEALTHCARE_THRESHOLDS = {
    'tat_target_minutes': 60,        # Primary clinical quality threshold
    'tat_warning_minutes': 45,       # Early warning threshold for intervention
    'tat_critical_minutes': 90,      # Critical delay requiring immediate attention
    'acceptable_violation_rate': 0.1, # 10% maximum threshold violation rate
    'model_accuracy_minimum': 0.85,   # Minimum prediction accuracy for production deployment
    'bottleneck_significance_threshold': 5.0  # Minutes delay threshold for bottleneck classification
}

# Production monitoring and alerting configuration
MONITORING_CONFIG = {
    'performance_metrics': [
        'RMSE_minutes',               # Root mean square error for continuous TAT prediction
        'MAE_minutes',                # Mean absolute error for interpretable accuracy
        'threshold_compliance_rate',   # Percentage of predictions meeting 60-minute threshold
        'bottleneck_detection_accuracy' # Accuracy of bottleneck identification
    ],
    'data_drift_metrics': [
        'feature_distribution_shift',  # Statistical distribution changes in input features
        'target_distribution_shift',   # Changes in TAT outcome distributions
        'categorical_proportion_drift' # Changes in categorical feature proportions
    ],
    'alert_thresholds': {
        'rmse_degradation_pct': 15,   # 15% RMSE increase triggers retraining alert
        'accuracy_drop_threshold': 0.05, # 5% accuracy drop requires investigation
        'data_drift_significance': 0.01   # Statistical significance for drift detection
    }
}

# MLOps deployment configuration for production healthcare analytics
MLOPS_CONFIG = {
    'model_refresh_frequency': 'monthly',  # Recommended retraining schedule
    'validation_split_strategy': 'temporal', # Time-based validation for healthcare data
    'feature_selection_threshold': 0.01,    # Minimum feature importance for inclusion
    'ensemble_voting_strategy': 'weighted',  # Performance-weighted ensemble predictions
    'production_latency_target_ms': 100,    # Real-time prediction latency requirement
    'batch_prediction_schedule': 'hourly',  # Automated batch prediction frequency
    'model_versioning_strategy': 'semantic', # Semantic versioning for model releases
    'rollback_criteria': {
        'performance_degradation_threshold': 0.1,  # 10% performance drop triggers rollback
        'error_rate_threshold': 0.05,             # 5% error rate maximum tolerance
        'latency_threshold_ms': 200                # Maximum acceptable prediction latency
    }
}