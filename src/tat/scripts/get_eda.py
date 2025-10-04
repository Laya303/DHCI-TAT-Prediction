""" 
Exploratory data analysis script for TAT datasets.

Generates comprehensive EDA reports and statistical summaries
for medication preparation analysis.mprehensive Exploratory Data Analysis Script for Medication Preparation TAT Analysis

Command-line interface for generating detailed exploratory data analysis reports
supporting medication preparation turnaround time analysis and pharmacy workflow
optimization. Provides production-ready EDA capabilities designed specifically for
healthcare analytics environments and clinical operations excellence initiatives.

Key Features:
- Comprehensive statistical summaries with clinical context and healthcare interpretation
- Automated report generation supporting healthcare documentation and quality assessment
- Configurable analysis parameters enabling customized exploration for diverse clinical needs
- Healthcare-optimized visualizations emphasizing clinical relevance and operational insights
- Export capabilities for clinical reporting and healthcare operations documentation

Usage Examples:
    # Standard EDA for pharmacy workflow analysis
    python get_eda.py --input tat_data.csv --save-dir eda_results --bins 15 --top 8
    
    # Quick exploration for clinical data quality assessment
    python get_eda.py -i pharmacy_orders.csv -b 20 -t 10
    
    # Comprehensive analysis for healthcare operations documentation
    python get_eda.py --input medication_tat.csv --save-dir comprehensive_eda --bins 25

"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from tat.data_io import DataIO
from tat.eda.summary.summary import DataSummary
from tat.eda.summary.summary_config import SummaryConfig


def main(argv: Optional[List[str]] = None) -> int:
    """
    Generate comprehensive exploratory data analysis for medication preparation TAT analysis.

    Creates detailed statistical summaries and visualizations supporting healthcare analytics
    and pharmacy workflow optimization through automated EDA report generation. Essential
    for clinical operations supporting evidence-based decision-making and healthcare quality
    improvement through comprehensive data exploration and pattern identification.

    Healthcare Analytics Workflow:
    - Clinical data loading: Reads medication preparation datasets with healthcare validation
    - Statistical analysis: Comprehensive summaries with healthcare domain context and interpretation
    - Healthcare visualization: Clinical-relevant charts supporting operational decision-making
    - Quality assessment: Data integrity evaluation ensuring accurate TAT analysis capabilities
    - Documentation export: Professional reports supporting healthcare operations and compliance

    Clinical Decision Support:
    - Variable distribution analysis: Understanding clinical and operational data patterns
    - Missing data assessment: Healthcare data quality evaluation supporting accurate modeling
    - Correlation identification: Clinical variable relationships supporting workflow optimization
    - Outlier detection: Healthcare data anomaly identification ensuring analysis reliability
    - Trend analysis: Temporal patterns supporting pharmacy operations improvement initiatives

    Args:
        argv: Command-line arguments supporting flexible analysis configuration for diverse
              healthcare analytics requirements and clinical workflow optimization scenarios.

    Returns:
        int: Exit code indicating analysis success (0) or specific failure modes for
        healthcare operations monitoring and automated pipeline integration requirements.

    Exit Codes:
        0: Successful EDA generation and report export for clinical use and documentation
        2: Input file access issues preventing healthcare data processing and analysis
        3: Data loading failures compromising exploratory analysis and clinical insights
        4: EDA generation failures affecting healthcare reporting and decision support

    Example:
        For comprehensive healthcare analytics data exploration:
        ```python
        # Automated healthcare analytics pipeline integration
        exit_code = main(['--input', 'tat_data.csv', '--save-dir', 'eda_reports', '--bins', '20'])
        
        # Clinical data quality assessment workflow
        exit_code = main(['-i', 'pharmacy_data.csv', '-b', '15', '-t', '8'])
        ```
    """
    # Configure comprehensive argument parser for healthcare analytics workflows
    p = argparse.ArgumentParser(
        description="Generate comprehensive EDA summary report for medication preparation TAT analysis",
        epilog="Essential for healthcare analytics supporting pharmacy workflow optimization and clinical operations excellence"
    )
    p.add_argument(
        "--input", "-i", 
        required=True, 
        help="Path to CSV dataset containing medication preparation TAT data for exploratory analysis"
    )
    p.add_argument(
        "--save-dir", "-s",
        default=None,
        help="Output directory for EDA artifacts and reports (default: <input_stem>_eda next to input file)"
    )
    p.add_argument(
        "--bins", "-b", 
        type=int, 
        default=15, 
        help="Number of bins for numeric histograms in healthcare data visualization (default: 15)"
    )
    p.add_argument(
        "--top", "-t", 
        type=int, 
        default=8, 
        help="Top-N categorical values displayed inline for clinical interpretation (default: 8)"
    )
    args = p.parse_args(argv)

    # Validate healthcare dataset accessibility with comprehensive error handling
    csv_path = Path(args.input).resolve()
    if not csv_path.exists():
        print(f"❌ Healthcare dataset not found: {csv_path}", file=sys.stderr)
        print("   Please verify the input file path for TAT analysis", file=sys.stderr)
        return 2

    print(f"📊 Loading healthcare dataset for exploratory analysis: {csv_path.name}")
    print(f"   Full path: {csv_path}")

    # Load healthcare dataset using production-ready data I/O system
    io = DataIO()
    try:
        df = io.read_csv(str(csv_path))
        print(f"✅ Healthcare dataset loaded successfully: {df.shape[0]:,} medication orders, {df.shape[1]} variables")
        
        # Provide basic dataset overview for clinical context
        print(f"   • Data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Identify key healthcare variable categories for analysis planning
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        lab_cols = [col for col in df.columns if col.startswith('lab_')]
        clinical_cols = [col for col in df.columns if any(term in col.lower() 
                        for term in ['age', 'diagnosis', 'severity', 'treatment'])]
        
        print(f"   • Timestamp variables: {len(time_cols)} (workflow analysis)")
        print(f"   • Laboratory values: {len(lab_cols)} (clinical assessment)")
        print(f"   • Clinical variables: {len(clinical_cols)} (patient context)")
        
    except Exception as e:
        print(f"❌ Failed to load healthcare dataset: {e}", file=sys.stderr)
        print("   Please verify CSV format and data integrity for TAT analysis", file=sys.stderr)
        return 3

    # Configure healthcare-optimized EDA parameters for clinical analysis
    print(f"\n🔧 Configuring EDA analysis parameters:")
    print(f"   • Histogram bins: {args.bins} (optimized for healthcare data distributions)")
    print(f"   • Top categorical values: {args.top} (clinical interpretation focus)")
    
    config = SummaryConfig(
        hist_bins=args.bins,  # Healthcare-optimized binning for clinical data patterns
        cat_top=args.top      # Clinical relevance focus for categorical analysis
    )
    
    # Initialize comprehensive data summary system with healthcare configuration
    ds = DataSummary(config=config)
    print("✅ EDA system initialized with healthcare-optimized configuration")

    # Determine output directory for healthcare reporting and documentation
    if args.save_dir:
        save_dir = Path(args.save_dir).resolve()
    else:
        save_dir = csv_path.parent / f"{csv_path.stem}_eda"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 EDA artifacts will be saved to: {save_dir}")

    # Generate comprehensive EDA report for healthcare analytics
    try:
        print(f"\n📈 Generating comprehensive EDA report for medication preparation TAT analysis...")
        print("   • Computing statistical summaries with healthcare context")
        print("   • Creating clinical data visualizations and distribution analysis")
        print("   • Assessing data quality and completeness for accurate modeling")
        print("   • Generating professional healthcare documentation")
        
        report_path = ds.report(df, save_dir=str(save_dir), export="html")
        
        print(f"\n✅ EDA report generation complete!")
        print(f"   • HTML report: {save_dir / 'summary_report.html'}")
        print(f"   • Supporting artifacts saved in: {save_dir}")
        print(f"   • Analysis covers {df.shape[0]:,} medication orders across {df.shape[1]} variables")
        
        # Provide guidance for clinical interpretation and next steps
        print(f"\n📋 Analysis Summary for Clinical Review:")
        print(f"   • Dataset: {df.shape[0]:,} medication preparation orders analyzed")
        print(f"   • Variables: {df.shape[1]} healthcare and operational features assessed")
        print(f"   • Configuration: {args.bins} histogram bins, top {args.top} categorical values")
        print(f"   • Output: Professional HTML report ready for clinical documentation")
        
        print(f"\n🎯 Next Steps for Healthcare Analytics:")
        print(f"   1. Review HTML report for clinical insights and data quality assessment")
        print(f"   2. Identify workflow bottlenecks and operational improvement opportunities") 
        print(f"   3. Plan targeted feature engineering for TAT prediction modeling")
        print(f"   4. Share findings with pharmacy operations team for workflow optimization")
        
    except Exception as e:
        print(f"❌ EDA report generation failed: {e}", file=sys.stderr)
        print("   Please verify data format and system requirements", file=sys.stderr)
        print("   Check available disk space and file permissions", file=sys.stderr)
        return 4

    print(f"\n🎉 Healthcare analytics EDA complete: {save_dir.resolve()}")
    print("   Ready for clinical review and pharmacy workflow optimization planning")
    return 0


if __name__ == "__main__":
    """
    Execute comprehensive exploratory data analysis for medication preparation TAT analysis.
    
    Provides command-line interface for healthcare analytics supporting pharmacy
    workflow optimization and clinical operations excellence through detailed data
    exploration and professional reporting capabilities.
    """
    raise SystemExit(main())