"""
Configuration Management for TAT EDA Summary Generation

Centralized configuration system for pharmacy turnaround time exploratory data
analysis and medication preparation workflow bottleneck identification. Provides
immutable, production-ready parameter management for healthcare analytics teams.

Key Configuration Areas:
- Statistical analysis parameters (percentiles, binning, correlation thresholds)
- Healthcare data type classification (temporal, categorical, numeric)
- Visualization settings optimized for clinical stakeholder consumption
- Column exclusion rules for HIPAA compliance and analytical focus
- TAT-specific workflow step identification and processing directives

"""
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Set


@dataclass(frozen=True)
class SummaryConfig:
    """
    Immutable configuration for TAT dataset analysis and pharmacy workflow optimization.
    
    Centralizes all analysis parameters to ensure consistent behavior across EDA,
    model development, and production monitoring systems. Designed specifically
    for medication preparation workflow analysis and bottleneck identification.
    
    Configuration Categories:
    - Statistical Analysis: Percentiles, binning, and correlation parameters
    - Healthcare Data Types: Column classification for appropriate analysis methods
    - Visualization Controls: Encoding and display options for clinical stakeholders
    - Workflow Analysis: TAT-specific column handling and exclusion rules
    - HIPAA Compliance: Identifier column management and sensitive data handling
    """

    # Statistical analysis parameters for healthcare TAT metrics
    percentiles: Tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    """Percentile values for TAT distribution analysis and SLA monitoring.
    
    Default includes standard healthcare analytics percentiles:
    - p1, p5: Lower tail for exceptional performance identification
    - p25, p50, p75: Quartiles for standard clinical interpretation  
    - p95, p99: Upper tail for bottleneck severity assessment and SLA violations
    """
    
    # Data quality and missing data assessment parameters
    missing_top_n: int = 15
    """Maximum columns to display in missing data quality reports.
    
    Healthcare datasets commonly have 10-15% missingness in workflow timestamps
    due to EHR integration gaps and manual data entry inconsistencies.
    """
    
    # Categorical and visualization analysis parameters
    cat_top: int = 4
    """Top-K categories to display for pharmacy operational factors.
    
    Optimized for clinical team consumption - displays dominant shift patterns,
    credential types, and department distributions without overwhelming detail.
    """
    
    hist_bins: int = 20
    """Default histogram bins for TAT delay distribution analysis.
    
    Balances statistical granularity with clinical interpretability for
    medication preparation timing visualizations and bottleneck identification.
    """

    # Visualization and reporting display controls
    sort_missing: bool = True
    """Sort missing data reports by severity for data quality prioritization."""
    
    force_ascii: Optional[bool] = None
    """Unicode/ASCII control for healthcare IT environment compatibility.
    
    Options:
    - True: ASCII-only for legacy healthcare terminal systems
    - False: Unicode for modern analytics dashboards and stakeholder reports
    - None: Auto-detection based on deployment environment capabilities
    """

    # Healthcare domain-specific column classification overrides
    categorical_overrides: Set[str] = field(default_factory=lambda: {
        "patient_readiness_score",  # Ordinal clinical assessment (1-3 scale)
        "floor",                    # Treatment floor location identifier (1-3)
        "premed_required",          # Binary clinical protocol indicator
        "stat_order",              # Binary urgency classification
        "TAT_over_60",             # Binary TAT SLA violation indicator
    })
    """Numeric columns that should be treated as categorical for clinical analysis.
    
    Ensures appropriate statistical analysis for ordinal scales, binary indicators,
    and location identifiers common in pharmacy workflow datasets.
    """

    # Categorical prefix patterns for one-hot encoded healthcare variables
    categorical_prefixes: Tuple[str, ...] = ("race_", "ins_", "dx_", "treat_", "shift_", "dept_")
    """Column prefixes indicating categorical one-hot encodings from data pipeline.
    
    Supports automated detection of:
    - race_*: Patient demographics for stratified analysis
    - ins_*: Insurance type categories for operational planning
    - dx_*: Diagnosis type classifications for acuity-based workflow analysis
    - treat_*: Treatment modality categories for preparation complexity assessment
    - shift_*: Temporal shift patterns for staffing optimization
    - dept_*: Ordering department categories for workflow source analysis
    """

    # Correlation analysis exclusion rules for focused TAT driver identification
    corr_exclude_prefixes: Tuple[str, ...] = (
        "race_", "ins_", "dx_", "treat_", "shift_", "dept_",
    )
    """Exclude one-hot encoded categoricals from correlation matrices.
    
    Prevents correlation analysis pollution from binary encoding artifacts
    while focusing on continuous operational drivers of TAT variation.
    """
    
    corr_exclude_columns: Tuple[str, ...] = (
        "sex",                        # Demographic categorical
        "severity",                   # Clinical acuity categorical  
        "patient_readiness_score",    # Ordinal clinical assessment
        "premed_required",           # Binary protocol indicator
        "stat_order",                # Binary urgency flag
        "order_dayofweek",           # Temporal categorical
        "order_month",               # Seasonal categorical
        "order_on_weekend",          # Binary temporal indicator
        "TAT_over_60",               # Target variable for supervised learning
        "floor",                     # Location categorical
        "order_id",                  # Primary key identifier
        "patient_id",                # Patient identifier (HIPAA sensitive)
    )
    """Specific columns excluded from numeric correlation analysis.
    
    Removes identifiers, target variables, and categorical encodings to focus
    correlation analysis on actionable operational and clinical drivers of TAT.
    """

    # Pharmacy workflow timestamp column identification
    known_time_cols: Set[str] = field(default_factory=lambda: {
        "doctor_order_time",         # Physician order initiation timestamp
        "nurse_validation_time",     # Clinical review and validation timestamp
        "prep_complete_time",        # Medication preparation completion timestamp
        "second_validation_time",    # Quality assurance validation timestamp
        "floor_dispatch_time",       # Floor delivery dispatch timestamp
        "patient_infusion_time",     # Patient administration initiation timestamp
    })
    """Sequential workflow timestamps for medication preparation process analysis.
    
    Defines the complete TAT workflow pipeline from physician order through
    patient administration for step-wise bottleneck identification and delay analysis.
    """

    # Histogram visualization suppression for workflow timestamp columns
    no_hist_cols: Set[str] = field(default_factory=lambda: {
        "second_validation_time",    
        "nurse_validation_time",     
        "prep_complete_time",        
        "floor_dispatch_time",       
        "patient_infusion_time",     
        "doctor_order_time",         
    })
    """Suppress histograms for raw timestamps in favor of step-wise delay analysis.
    
    Raw timestamps provide limited clinical insight compared to calculated
    delay intervals between workflow steps for bottleneck identification.
    """

    # HIPAA-compliant identifier column management
    id_columns: Set[str] = field(default_factory=lambda: {"order_id", "patient_id"})
    """Patient and order identifiers requiring special handling for HIPAA compliance.
    
    Separated from analytical features to ensure appropriate data governance
    and prevent accidental inclusion in model training or stakeholder reports.
    """
    
    # TAT-specific analysis thresholds and clinical parameters
    tat_threshold_minutes: float = 60.0
    """Standard TAT SLA threshold for pharmacy operations (60 minutes).
    
    Clinical benchmark for medication preparation workflow optimization
    and patient care quality assessment. Used for violation rate calculations
    and bottleneck severity classification throughout EDA and modeling.
    """
    
    # Healthcare analytics quality thresholds
    data_quality_warning_threshold: float = 0.10
    """Missing data warning threshold (10%) for healthcare data quality assessment."""
    
    data_quality_critical_threshold: float = 0.25
    """Missing data critical threshold (25%) requiring immediate data quality intervention."""
    
    correlation_significance_threshold: float = 0.1
    """Minimum correlation coefficient for clinical significance in TAT driver analysis."""
    
    # Production monitoring and MLOps configuration
    enable_production_logging: bool = True
    """Enable detailed logging for production TAT monitoring and model performance tracking."""
    
    max_categorical_cardinality: int = 50
    """Maximum unique values for categorical analysis to prevent memory issues with high-cardinality variables."""