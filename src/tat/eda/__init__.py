"""
Exploratory Data Analysis (EDA) Utilities for Pharmacy TAT Workflow Optimization

Core EDA components supporting medication preparation turnaround time
analysis and bottleneck identification initiatives. Designed specifically for healthcare
analytics teams conducting TAT workflow optimization and automated monitoring.

Key Components:
- DataSummary: Comprehensive TAT dataset analysis with multi-format reporting
- StepDelayVisualizer: Medication preparation step-by-step delay analysis and visualization
- Healthcare-optimized configuration management for clinical interpretation
- Production-ready analysis suitable for automated pharmacy performance monitoring

Usage Example:
    from tat.eda import DataSummary, StepDelayVisualizer
    
    # Comprehensive TAT workflow analysis
    summary = DataSummary.default()
    analysis_artifacts = summary.print_report(tat_df)
    
    # Step-by-step delay analysis and visualization
    visualizer = StepDelayVisualizer()
    # For full TAT datasets with all required workflow columns:
    # delay_data = visualizer.compute_delays(tat_df)  
    # delay_stats = visualizer.available_delay_stats(tat_df)
"""

from .summary.summary import DataSummary
from .step_delay_plots import StepDelayVisualizer

# Version for MLOps tracking and model lifecycle management
__version__ = "1.0.0"

__all__ = [
    'DataSummary',
    'StepDelayVisualizer',
    '__version__',
]

# Healthcare analytics metadata for production deployment
SUPPORTED_TAT_COLUMNS = [
    'doctor_order_time',
    'nurse_validation_time', 
    'prep_complete_time',
    'second_validation_time',
    'floor_dispatch_time',
    'patient_infusion_time',
]

TAT_SLA_THRESHOLD_MINUTES = 60.0  # Healthcare acceptable TAT threshold