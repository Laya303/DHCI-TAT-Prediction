"""
TAT EDA Summary Generation Package

Comprehensive exploratory data analysis summary tools for pharmacy turnaround time 
optimization and medication preparation workflow analysis. Provides automated 
statistical summaries, data quality assessments, and visualization capabilities
for healthcare analytics teams.

Key Components:
- DataSummary: Primary interface for comprehensive dataset analysis
- SummaryConfig: Configuration management for customizable analysis parameters
- Automated reporting tools for pharmacy leadership and clinical stakeholders

Usage:
    from tat.eda.summary import DataSummary, SummaryConfig
    
    # Basic TAT dataset analysis
    summary = DataSummary(df)
    report = summary.generate_comprehensive_report()
    
    # Custom configuration for pharmacy workflow analysis
    config = SummaryConfig(focus_columns=['TAT_minutes', 'delay_*'])
    summary = DataSummary(df, config=config)
"""

import pandas as pd

# Core summary generation components
from .summary import DataSummary
from .summary_config import SummaryConfig

# Advanced analysis modules for healthcare data
try:
    from . import summary_analyze
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False

try:
    from . import summary_render
    RENDERER_AVAILABLE = True
except ImportError:
    RENDERER_AVAILABLE = False

try:
    from . import summary_tables
    TABLES_AVAILABLE = True
except ImportError:
    TABLES_AVAILABLE = False

# Version and metadata for MLOps pipeline tracking
__version__ = "1.0.0"
__author__ = "TAT Analytics Team"

# Public API exports for healthcare data analysis
__all__ = [
    'DataSummary',
    'SummaryConfig',
]

# Conditionally add advanced components if available
if ANALYZER_AVAILABLE:
    __all__.append('summary_analyze')

if RENDERER_AVAILABLE:
    __all__.append('summary_render')
    
if TABLES_AVAILABLE:
    __all__.append('summary_tables')

# Healthcare-specific configuration constants
TAT_THRESHOLD_MINUTES = 60  # Standard TAT SLA for pharmacy operations
CRITICAL_COLUMNS = [
    'TAT_minutes',
    'delay_order_to_nurse',
    'delay_nurse_to_prep', 
    'delay_prep_to_second',
    'delay_second_to_dispatch',
    'delay_dispatch_to_infusion'
]

# Data quality thresholds for automated assessment
DATA_QUALITY_THRESHOLDS = {
    'missing_data_warning': 0.10,    # 10% missing data threshold
    'missing_data_critical': 0.25,   # 25% missing data critical threshold
    'outlier_detection_zscore': 3.0, # Z-score threshold for outlier detection
    'correlation_threshold': 0.7     # High correlation threshold for feature analysis
}

# Healthcare analytics configuration presets
PHARMACY_ANALYSIS_PRESET = {
    'focus_areas': ['workflow_bottlenecks', 'staffing_impact', 'temporal_patterns'],
    'key_metrics': ['median_tat', 'violation_rate', 'bottleneck_severity'],
    'visualization_types': ['distribution', 'correlation', 'temporal', 'categorical']
}

def get_default_config() -> SummaryConfig:
    """Get default configuration for TAT summary analysis.
    
    Returns:
        SummaryConfig: Default configuration optimized for healthcare TAT analysis
    """
    return SummaryConfig()

def create_pharmacy_summary(df: pd.DataFrame) -> SummaryConfig:
    """Create summary configuration tailored for pharmacy workflow analysis.
    
    Args:
        df: TAT dataset with pharmacy workflow and patient data
        
    Returns:
        SummaryConfig: Configuration optimized for pharmacy TAT analysis
    """
    config = SummaryConfig(
        max_categorical_cardinality=min(50, len(df) // 100)
    )
    return config

# Module availability status for conditional feature usage
MODULE_STATUS = {
    'analyzer_available': ANALYZER_AVAILABLE,
    'renderer_available': RENDERER_AVAILABLE, 
    'tables_available': TABLES_AVAILABLE,
    'core_functionality': True
}