"""
TAT Prediction Package for Pharmacy Workflow Optimization

Core package for analyzing and predicting medication preparation turnaround times (TAT).
Provides tools for feature engineering, modeling, and bottleneck analysis to optimize
pharmacy workflows and meet 60-minute threshold compliance.

Key Modules:
- config: Core configuration and constants
- data_io: Data loading and validation utilities  
- features: Temporal, clinical, and operational feature engineering
- eda: Exploratory data analysis and visualization
- models: Machine learning models for TAT prediction
- analysis: Bottleneck identification and feature importance
- pipelines: Data processing pipelines
- scripts: Command-line tools for workflow analysis
"""

# Core configuration and I/O for healthcare analytics infrastructure
from .config import (
    STEP_COLS,           # Sequential medication preparation step columns for workflow analysis
    TARGETS,             # TAT prediction targets supporting 60-minute threshold compliance
    ORDER_TIME_COL,      # Canonical order timestamp column for temporal analysis
    LAB_COLS,           # Laboratory value columns for clinical context integration
    TRAINING_CONFIGS,    # Multi-configuration training setups for diverse healthcare scenarios
)
from .data_io import DataIO

# Comprehensive feature engineering modules for healthcare analytics
from .features.temporal.time_reconstruct import TimeReconstructor
from .features.temporal.delays import DelayEngineer  
from .features.temporal.temporal import TemporalEngineer
from .features.categoricals import CategoricalEncoder
from .features.labs import LabProcessor
from .features.cleaners import Cleaner
from .features.operational import OperationalEngineer

# Advanced EDA and visualization for pharmacy workflow analysis  
from .eda.summary.summary import DataSummary
from .eda.step_delay_plots import StepDelayVisualizer

# Comprehensive analysis modules for bottleneck identification
from .analysis.feature_importance import FeatureImportanceAnalyzer
from .analysis.bottleneck_analysis import BottleneckAnalyzer

# Production-ready modeling infrastructure for TAT prediction
from .models.factory import TATModelFactory, TATTrainingOrchestrator
from .models.base import BaseTATModel
from .models.xgboost_model import XGBoostTATRegressor
from .models.random_forest_model import RandomForestTATRegressor
from .models.linear_model import RidgeTATRegressor
from .models.ensemble_model import StackingTATRegressor

# Data pipeline modules for consistent preprocessing
from .pipelines import DatasetBuilder

__all__ = [
    # Core Healthcare Analytics Configuration
    "STEP_COLS", "TARGETS", "ORDER_TIME_COL", "LAB_COLS", "TRAINING_CONFIGS",
    "DataIO",
    
    # Comprehensive Feature Engineering for Healthcare Analytics
    "TimeReconstructor",
    "DelayEngineer", 
    "TemporalEngineer",
    "CategoricalEncoder",
    "LabProcessor", 
    "Cleaner",
    "OperationalEngineer",
    
    # Advanced Analysis and Visualization for Pharmacy Workflow Optimization
    "DataSummary",
    "StepDelayVisualizer", 
    "FeatureImportanceAnalyzer",
    "BottleneckAnalyzer",
    
    # Production-Ready Modeling Infrastructure for TAT Prediction
    "TATModelFactory",
    "TATTrainingOrchestrator",
    "BaseTATModel",
    "XGBoostTATRegressor",
    "RandomForestTATRegressor", 
    "RidgeTATRegressor",
    "StackingTATRegressor",
    
    # Healthcare Data Pipeline Modules
    "DatasetBuilder",
]

# Package metadata for healthcare analytics deployment
__version__ = "1.0.0"
__description__ = "Advanced TAT Prediction System for Pharmacy Workflow Optimization"
__healthcare_focus__ = "Medication Preparation Turnaround Time Analysis & Bottleneck Identification"
__clinical_threshold__ = "60-minute TAT Compliance Requirement"
__target_application__ = "Healthcare Pharmacy Operations Excellence Initiative"

# Healthcare analytics package information
__package_info__ = {
    "name": "tat",
    "version": __version__,
    "description": __description__,
    "healthcare_domain": "Pharmacy Operations & Medication Preparation Workflow",
    "clinical_application": "TAT Prediction & Bottleneck Analysis",  
    "quality_threshold": "60 minutes",
    "analytics_focus": [
        "Step-to-step delay analysis",
        "Workflow bottleneck identification", 
        "TAT threshold compliance monitoring",
        "Predictive resource allocation",
        "Evidence-based workflow optimization"
    ],
    "production_capabilities": [
        "Real-time TAT prediction",
        "Automated bottleneck detection",
        "MLOps-ready deployment", 
        "Continuous model monitoring",
        "Healthcare IT integration"
    ],
    "stakeholder_value": [
        "Pharmacy operations teams",
        "Clinical leadership",
        "Healthcare quality improvement",
        "Patient care throughput enhancement",
        "Evidence-based decision making"
    ]
}

def get_package_info() -> dict:
    """
    Retrieve comprehensive package information for healthcare analytics deployment.
    
    Provides detailed metadata supporting healthcare stakeholder communication,
    deployment planning, and clinical documentation requirements for pharmacy
    workflow optimization and TAT prediction system implementation.
    
    Returns:
        dict: Comprehensive package information including healthcare context,
    clinical applications, and production capabilities supporting healthcare
    pharmacy operations excellence and workflow optimization.    Example:
        For healthcare deployment documentation:
        ```python
        import tat
        package_info = tat.get_package_info()
        print(f"Healthcare Application: {package_info['clinical_application']}")
        print(f"Quality Threshold: {package_info['quality_threshold']}")
        ```
    
    Note:
        Essential for healthcare IT integration supporting pharmacy workflow
        optimization and medication preparation efficiency through comprehensive
        TAT analysis enabling clinical operations excellence and patient care improvement.
    """
    return __package_info__.copy()