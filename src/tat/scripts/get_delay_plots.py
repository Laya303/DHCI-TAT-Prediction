""" 
Step-by-step delay visualization script.

Generates delay analysis plots to identify bottlenecks in
medication preparation workflow.ep-to-Step Delay Visualization Script for Medication Preparation TAT Analysis

Command-line interface for generating comprehensive step-to-step delay visualizations
supporting pharmacy workflow bottleneck identification and turnaround time optimization.
Provides production-ready delay analysis capabilities designed specifically for healthcare
analytics environments and clinical operations excellence initiatives.

Key Features:
- Consistent data processing using diagnostics pipeline for reproducible bottleneck analysis
- Enhanced box plot visualizations highlighting workflow step delays and bottleneck patterns
- SLA threshold integration supporting clinical quality standards and operational monitoring
- Export capabilities for clinical reporting and healthcare operations documentation
- Interactive visualization options for real-time analysis and clinical decision support

Technical Features:
- Automated delay feature extraction from medication preparation timestamp sequences
- Consistent processing pipeline ensuring reproducible analysis results across analyses
- Healthcare-optimized visualization design emphasizing clinical interpretation and usability
- Flexible output options supporting diverse healthcare reporting and documentation requirements
- Error handling and validation ensuring robust operation in clinical environments

Usage Examples:
    # Generate delay analysis for pharmacy workflow optimization
    python get_delay_plots.py --input tat_data.csv --save delay_analysis.png --sla 60
    
    # Interactive analysis for real-time clinical decision support
    python get_delay_plots.py --input tat_data.csv --show --sla 45
    
    # Comprehensive bottleneck reporting with custom SLA thresholds
    python get_delay_plots.py -i pharmacy_tat.csv -s bottleneck_report.png --sla 75

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from tat.eda.step_delay_plots import StepDelayVisualizer
from tat.pipelines.make_dataset import make_diagnostics


def main(argv: Optional[list[str]] = None) -> int:
    """
    Generate comprehensive step-to-step delay visualizations for pharmacy workflow analysis.

    Creates production-ready delay box plots supporting medication preparation bottleneck
    identification and TAT optimization through consistent data processing and healthcare-
    optimized visualization design. Essential for clinical operations supporting pharmacy
    workflow efficiency assessment and healthcare quality improvement initiatives.

    Clinical Workflow Analysis:
    - Step-to-step delay quantification: Identifies specific bottlenecks in medication preparation
    - SLA threshold analysis: Compares delays against clinical quality standards and thresholds
    - Workflow optimization insight: Visual patterns supporting pharmacy operations improvement
    - Quality monitoring capabilities: Delay trend analysis for healthcare operations excellence

    Data Processing Integration:
    - Consistent diagnostics pipeline: Ensures reproducible analysis across healthcare workflows
    - Healthcare data validation: Clinical bounds checking and quality assurance throughout processing
    - Feature engineering consistency: Standardized delay calculation supporting accurate analysis
    - Production-ready processing: Scalable analysis suitable for large healthcare datasets

    Args:
        argv: Command-line arguments supporting flexible analysis configuration for diverse
              healthcare analytics requirements and clinical workflow optimization scenarios.

    Returns:
        int: Exit code indicating analysis success (0) or specific failure modes for
        healthcare operations monitoring and automated pipeline integration requirements.

    Exit Codes:
        0: Successful delay analysis and visualization generation for clinical use
        2: Input file access issues preventing healthcare data processing and analysis
        3: Data processing failures compromising delay calculation and bottleneck identification
        4: Visualization generation failures affecting clinical reporting and decision support
        5: Output saving issues preventing healthcare documentation and operations integration

    Example:
        For comprehensive pharmacy workflow bottleneck analysis:
        ```python
        # Automated healthcare analytics pipeline integration
        exit_code = main(['--input', 'tat_data.csv', '--save', 'delays.png', '--sla', '60'])
        
        # Interactive clinical decision support analysis
        exit_code = main(['--input', 'pharmacy_data.csv', '--show', '--sla', '45'])
        ```
    """
    # Configure comprehensive argument parser for healthcare analytics workflows
    p = argparse.ArgumentParser(
        description="Generate step-to-step delay visualizations for medication preparation TAT analysis",
        epilog="Essential for pharmacy workflow bottleneck identification and clinical operations optimization"
    )
    p.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to CSV dataset containing medication preparation TAT data for delay analysis"
    )
    p.add_argument(
        "--save", "-s", 
        default=None, 
        help="Output path for delay visualization (PNG format) supporting clinical reporting"
    )
    p.add_argument(
        "--show", 
        action="store_true", 
        help="Display interactive delay visualization for real-time clinical analysis"
    )
    p.add_argument(
        "--sla", 
        type=float, 
        default=60.0, 
        help="SLA threshold in minutes for clinical quality assessment (default: 60 minutes)"
    )
    args = p.parse_args(argv)

    # Prepare output directory for healthcare reporting and documentation
    if args.save:
        save_path = Path(args.save).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Configured output path: {save_path}")

    # Load healthcare dataset with comprehensive validation and error handling
    csv_path = Path(args.input).resolve()
    if not csv_path.exists():
        print(f"❌ Healthcare dataset not found: {csv_path}", file=sys.stderr)
        print("   Please verify the input file path for TAT analysis", file=sys.stderr)
        return 2

    try:
        df_raw = pd.read_csv(csv_path)
        print(f"✅ Loaded healthcare dataset: {df_raw.shape[0]:,} medication orders, {df_raw.shape[1]} variables")
        
        # Validate essential columns for TAT analysis
        required_time_cols = ['doctor_order_time', 'patient_infusion_time']
        missing_cols = [col for col in required_time_cols if col not in df_raw.columns]
        if missing_cols:
            print(f"⚠️  Missing essential timestamp columns: {missing_cols}", file=sys.stderr)
            print("   Dataset may require preprocessing for complete delay analysis", file=sys.stderr)
        
    except Exception as e:
        print(f"❌ Failed to load healthcare dataset: {e}", file=sys.stderr)
        print("   Please verify CSV format and data integrity", file=sys.stderr)
        return 2

    try:
        print("\n📊 Processing healthcare data using comprehensive diagnostics pipeline...")
        print("   Applying consistent feature engineering for reproducible delay analysis")
        
        # Apply production-ready diagnostics pipeline for consistent delay analysis
        X_diag, y_reg, y_clf, scaler_info, removal_info = make_diagnostics(
            df_raw, 
            scaling_strategy="mixed"  # Healthcare-optimized scaling preserving delay interpretability
        )
        
        # Reconstruct comprehensive dataset with delay features for visualization
        df_processed = X_diag.copy()
        df_processed['TAT_minutes'] = y_reg
        df_processed['TAT_over_60'] = y_clf
        
        print(f"✅ Healthcare data processing complete: {df_processed.shape[0]:,} samples, {df_processed.shape[1]} features")
        
        # Identify delay features for bottleneck analysis
        delay_cols = [col for col in df_processed.columns if col.startswith('delay_')]
        print(f"📈 Identified {len(delay_cols)} step-to-step delay features for analysis:")
        for delay_col in delay_cols:
            delay_stats = df_processed[delay_col].describe()
            print(f"   • {delay_col}: mean={delay_stats['mean']:.1f}min, "
                  f"median={delay_stats['50%']:.1f}min, max={delay_stats['max']:.1f}min")
        
        if not delay_cols:
            print("❌ No delay columns found in processed healthcare data!", file=sys.stderr)
            print("   Dataset may not contain sufficient timestamp information for delay analysis", file=sys.stderr)
            return 3
        
        print(f"\n🎯 Generating delay visualization with SLA threshold: {args.sla} minutes")
        print("   Using healthcare-optimized visualization design for clinical interpretation")
        
        # Initialize healthcare-optimized delay visualizer for bottleneck analysis
        vis = StepDelayVisualizer(
            delay_engineer=None, 
            figsize=(14, 8),  # Clinical reporting format
            impute_missing=False  # Preserve data integrity for accurate analysis
        )
        
        # Generate comprehensive delay visualization using consistent methodology
        fig = vis._plot_box_from_processed_delays(
            df_processed,
            delay_cols=delay_cols,
            sla_minutes=args.sla,
            show=args.show,
            save_path=str(save_path) if args.save else None,
            title="Medication Preparation Step-to-Step Delays: Workflow Bottleneck Analysis"
        )

        # Provide comprehensive analysis summary for clinical decision support
        print(f"\n📋 Delay Analysis Summary:")
        print(f"   • Dataset: {len(df_processed):,} medication preparation orders analyzed")
        print(f"   • Delay Features: {len(delay_cols)} step-to-step delays quantified")
        print(f"   • SLA Threshold: {args.sla} minutes (clinical quality standard)")
        
        # Calculate SLA compliance statistics for healthcare quality monitoring
        sla_violations = {}
        for delay_col in delay_cols:
            violations = (df_processed[delay_col] > args.sla).sum()
            violation_rate = violations / len(df_processed) * 100
            sla_violations[delay_col] = violation_rate
            if violation_rate > 10:  # Highlight significant bottlenecks
                print(f"   ⚠️  {delay_col}: {violation_rate:.1f}% exceed SLA (potential bottleneck)")
        
        if args.save:
            print(f"\n✅ Delay visualization saved: {save_path}")
            print("   Ready for clinical reporting and healthcare operations documentation")
        
        if args.show:
            print("🖥️  Interactive visualization displayed for real-time clinical analysis")
        
        print(f"\n🎉 Delay analysis complete: {len(df_processed):,} orders analyzed with consistent pipeline")
        print("   Results support pharmacy workflow optimization and clinical operations excellence")
            
    except ValueError as e:
        print(f"❌ Delay visualization generation failed: {e}", file=sys.stderr)
        print("   Please verify data format and visualization requirements", file=sys.stderr)
        return 5
    except Exception as e:
        print(f"❌ Healthcare data processing failed: {e}", file=sys.stderr)
        print("   Please verify data integrity and processing requirements", file=sys.stderr)
        return 4

    return 0


if __name__ == "__main__":
    """
    Execute step-to-step delay visualization for medication preparation TAT analysis.
    
    Provides command-line interface for healthcare analytics supporting pharmacy
    workflow bottleneck identification and clinical operations optimization through
    comprehensive delay analysis and visualization capabilities.
    """
    raise SystemExit(main())