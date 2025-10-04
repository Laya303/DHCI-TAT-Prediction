"""
TAT Analysis Command-Line Scripts Module

Command-line interface collection for medication preparation turnaround time analysis
and pharmacy workflow optimization. Provides healthcare analytics CLI tools designed
specifically for pharmacy operations teams and clinical analysts conducting TAT
analysis, bottleneck identification, and workflow optimization initiatives.

Key Components:
- Dataset Preparation: Automated F0 and diagnostics dataset creation (prepare_dataset_main)
- Exploratory Analysis: Comprehensive TAT data exploration and statistical summaries (get_eda_main)
- Bottleneck Analysis: Step-to-step delay identification and workflow optimization (run_bottleneck_analysis_main)
- Visualization Tools: Delay plotting and clinical insight generation (get_delay_plots_main)

Usage Examples:
    # Dataset preparation for modeling
    python -m src.tat.scripts.prepare_dataset --input raw_data.csv --output processed/
    
    # Exploratory data analysis
    python -m src.tat.scripts.get_eda --input dataset.csv --save-dir reports/
    
    # Bottleneck analysis
    python -m src.tat.scripts.run_bottleneck_analysis --input dataset.csv --output analysis/
    
    # Delay visualization
    python -m src.tat.scripts.get_delay_plots --input dataset.csv --save-dir plots/

Note: train_model.py exists but uses different CLI pattern (no main function) and is not imported here.
"""

from .prepare_dataset import main as prepare_dataset_main
from .get_eda import main as get_eda_main
from .run_bottleneck_analysis import main as run_bottleneck_analysis_main
from .get_delay_plots import main as get_delay_plots_main

# Module version for healthcare analytics pipeline tracking
__version__ = "1.0.0"

__all__ = [
    # Dataset Preparation
    'prepare_dataset_main',
    
    # Exploratory Data Analysis
    'get_eda_main',
    
    # Bottleneck Analysis
    'run_bottleneck_analysis_main',
    
    # Visualization and Reporting
    'get_delay_plots_main'
]