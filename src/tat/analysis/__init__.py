"""
TAT Analysis Module

Key Components:
- BottleneckAnalyzer: Multi-dimensional workflow bottleneck detection
- FeatureImportanceAnalyzer: SHAP-based model interpretability for clinical insights

Usage:
    from tat.analysis import BottleneckAnalyzer, FeatureImportanceAnalyzer
    
    # Workflow bottleneck analysis
    analyzer = BottleneckAnalyzer(tat_threshold=60.0)
    report = analyzer.generate_bottleneck_report(df)
    
    # ML model interpretability
    importance_analyzer = FeatureImportanceAnalyzer(model, X_train)
    shap_results = importance_analyzer.shap_summary(X_test)
"""

from .bottleneck_analysis import BottleneckAnalyzer
from .feature_importance import FeatureImportanceAnalyzer

# Version information for MLOps pipeline tracking
__version__ = "1.0.0"

# Public API exports for healthcare analytics teams
__all__ = [
    'BottleneckAnalyzer',
    'FeatureImportanceAnalyzer',
]

# Module-level constants for healthcare TAT analysis
DEFAULT_TAT_THRESHOLD = 60.0  # Standard 60-minute TAT SLA threshold
SUPPORTED_MODEL_TYPES = [
    'xgboost', 
    'random_forest', 
    'linear', 
    'ensemble'
]

# Clinical workflow steps for bottleneck analysis
WORKFLOW_STEPS = [
    'delay_order_to_nurse',
    'delay_nurse_to_prep', 
    'delay_prep_to_second',
    'delay_second_to_dispatch',
    'delay_dispatch_to_infusion'
]