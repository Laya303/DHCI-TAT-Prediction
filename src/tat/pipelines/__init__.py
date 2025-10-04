"""
TAT Dataset Creation Pipeline Module

Specialized dataset creation pipeline for medication preparation turnaround time
analysis and pharmacy workflow optimization. Provides automated dataset generation
capabilities designed specifically for healthcare analytics environments and
TAT prediction modeling workflows.

Key Components:
- DatasetBuilder: Automated dataset creation for F0 (real-time prediction) and diagnostics (comprehensive analysis)

Dataset Types:
- F0 Dataset: Production-ready features excluding future information for real-time TAT prediction deployment
- Diagnostics Dataset: Comprehensive analytical dataset including delay features for workflow bottleneck identification

Technical Features:
- Automated feature engineering pipeline with healthcare-optimized transformations
- Scaling strategy customization for production ML model requirements
- Clinical validation and data integrity checks throughout processing
- Comprehensive metadata generation for model deployment and monitoring

Usage Example:
    from tat.pipelines import DatasetBuilder
    
    # Create F0 dataset for real-time prediction deployment
    builder = DatasetBuilder()
    f0_dataset = builder.create_f0(raw_tat_df, scaling_strategy="mixed")
    
    # Create diagnostics dataset for workflow analysis
    diag_dataset = builder.create_diagnostics(raw_tat_df, scaling_strategy="mixed")
"""

from .make_dataset import DatasetBuilder

__all__ = ['DatasetBuilder']