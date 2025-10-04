""" 
Bottleneck analysis script for workflow optimization.

Identifies delays and inefficiencies in medication preparation
steps with recommendations for improvement.T Bottleneck Analysis Script for Medication Preparation Workflow Optimization

Command-line interface for comprehensive bottleneck identification in medication 
preparation turnaround time analysis supporting pharmacy workflow optimization.
Analyzes step-to-step delays to identify workflow inefficiencies and provide
actionable recommendations for improving patient care throughput and meeting
the 60-minute TAT threshold requirement.

Key Features:
- Comprehensive step-to-step delay analysis identifying specific workflow bottlenecks
- Operational context analysis examining shift, floor, and staffing impact on TAT
- Temporal pattern identification revealing time-based workflow optimization opportunities
- Clinical workflow recommendations supporting pharmacy operations improvement initiatives
- Production-ready visualizations for healthcare stakeholder communication and reporting

Technical Features:
- Automated bottleneck detection using comprehensive delay feature analysis algorithms
- Multi-dimensional analysis examining temporal, operational, and clinical factors affecting TAT
- Healthcare-optimized visualization generation supporting clinical decision-making processes
- Comprehensive reporting with actionable recommendations for pharmacy workflow improvement
- Error handling and validation ensuring robust operation in healthcare environments

Usage Examples:
    # Standard bottleneck analysis for pharmacy workflow optimization
    python run_bottleneck_analysis.py
    
    # Detailed analysis with comprehensive logging for troubleshooting
    python run_bottleneck_analysis.py --verbose
    
    # Automated pipeline integration for continuous workflow monitoring
    analyze-bottlenecks  # if installed via setup.py

"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

# Import project modules for comprehensive TAT analysis
from tat.data_io import DataIO
from tat.pipelines.make_dataset import make_diagnostics
from tat.analysis.bottleneck_analysis import BottleneckAnalyzer
from tat.eda.step_delay_plots import StepDelayVisualizer

# Configure logging for healthcare analytics bottleneck analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Healthcare analytics project structure configuration
PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_DIR / "data"
REPORTS_DIR = PROJECT_DIR / "reports"

def setup_output_directories() -> Dict[str, Path]:
    """
    Create organized output directory structure for bottleneck analysis results.
    
    Establishes healthcare analytics reporting structure supporting clinical
    documentation requirements and pharmacy workflow optimization reporting
    needs. Essential for maintaining organized analysis artifacts and
    supporting healthcare stakeholder communication and decision-making.
    
    Returns:
        Dict[str, Path]: Directory structure mapping for healthcare analytics
        reporting including comprehensive analysis artifacts and visualizations.
    """
    dirs = {
        'reports': REPORTS_DIR,
        'figures': REPORTS_DIR / "figures", 
        'bottleneck_analysis': REPORTS_DIR / "bottleneck_analysis"
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✅ Healthcare analytics output directories configured:")
    for name, path in dirs.items():
        logger.info(f"   • {name.title()}: {path}")
    
    return dirs

def load_and_prepare_data() -> pd.DataFrame:
    """
    Load and prepare medication preparation TAT data for comprehensive bottleneck analysis.
    
    Processes healthcare dataset using production-ready diagnostics pipeline ensuring
    consistent feature engineering and delay calculation for accurate bottleneck
    identification. Essential for healthcare analytics supporting pharmacy workflow
    optimization and medication preparation efficiency analysis.
    
    Data Preparation Pipeline:
    - Raw healthcare dataset loading with comprehensive validation and error handling
    - Diagnostics feature engineering including step-to-step delay calculation algorithms
    - Operational context preservation supporting conditional bottleneck analysis workflows
    - Temporal information reconstruction enabling time-based pattern identification analysis
    - Data quality validation ensuring accurate bottleneck identification and clinical insights
    
    Returns:
        pd.DataFrame: Comprehensive dataset with delay features and operational context
        suitable for bottleneck analysis and pharmacy workflow optimization research.
    
    Raises:
        FileNotFoundError: When required healthcare dataset is not accessible for analysis
        ValueError: When data processing fails compromising bottleneck analysis capabilities
    
    """
    logger.info("🏥 Loading medication preparation TAT dataset for bottleneck analysis...")
    
    # Load healthcare dataset using production-ready data I/O system
    io = DataIO()
    dataset_path = DATA_DIR / 'raw' / 'DFCI_TAT_Dataset_100k.csv'
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Healthcare dataset not found: {dataset_path}")
    
    df_raw = io.read_csv(dataset_path)
    logger.info(f"✅ Raw healthcare dataset loaded: {df_raw.shape[0]:,} medication orders, {df_raw.shape[1]} variables")
    
    # Apply comprehensive diagnostics pipeline for consistent delay analysis
    logger.info("📊 Processing healthcare data using diagnostics pipeline for bottleneck analysis...")
    X_diag, y_reg, y_clf, scaler_info, removal_info = make_diagnostics(
        df_raw, 
        scaling_strategy="mixed"  # Healthcare-optimized scaling preserving delay interpretability
    )
    
    # Reconstruct comprehensive dataset preserving clinical context for analysis
    df_prepared = X_diag.copy()
    df_prepared['TAT_minutes'] = y_reg
    df_prepared['TAT_over_60'] = y_clf
    
    # Preserve essential operational context for conditional bottleneck analysis
    key_operational_columns = [
        'shift', 'floor', 'severity', 'pharmacists_on_duty', 
        'nurse_credential', 'pharmacist_credential', 'treatment_type',
        'diagnosis_type', 'stat_order', 'premed_required'
    ]
    
    for col in key_operational_columns:
        if col in df_raw.columns and col not in df_prepared.columns:
            df_prepared[col] = df_raw[col].values
            logger.debug(f"   • Preserved operational context: {col}")
    
    # Add temporal information for time-based bottleneck pattern analysis
    if 'doctor_order_time' in df_raw.columns:
        df_prepared['doctor_order_time_dt'] = pd.to_datetime(
            df_raw['doctor_order_time'], 
            errors='coerce'
        )
        logger.info("   • Temporal analysis capability enabled")
    
    # Comprehensive delay feature validation for bottleneck analysis accuracy
    delay_cols = [col for col in df_prepared.columns if col.startswith('delay_')]
    logger.info(f"📈 Delay feature validation: {len(delay_cols)} step-to-step delay columns identified")
    
    if not delay_cols:
        logger.warning("❌ No delay columns found! Bottleneck analysis may be limited.")
    else:
        # Validate delay data quality for accurate bottleneck identification
        for col in delay_cols:
            non_null_count = df_prepared[col].notna().sum()
            if non_null_count > 0:
                stats = df_prepared[col].describe()
                zero_pct = (df_prepared[col] == 0).sum() / non_null_count * 100
                logger.info(f"   • {col}: {non_null_count:,} valid values, "
                          f"mean={stats['mean']:.1f}min, median={stats['50%']:.1f}min, "
                          f"{zero_pct:.1f}% zero values")
            else:
                logger.warning(f"   ❌ {col}: All values missing - bottleneck analysis affected")
    
    # Validate TAT threshold compliance for clinical quality assessment
    tat_over_60_pct = df_prepared['TAT_over_60'].mean() * 100
    logger.info(f"🎯 TAT threshold analysis: {tat_over_60_pct:.1f}% orders exceed 60-minute threshold")
    
    logger.info(f"✅ Healthcare data preparation complete: {df_prepared.shape[0]:,} orders ready for analysis")
    return df_prepared

def perform_bottleneck_analysis(df: pd.DataFrame, output_dirs: Dict[str, Path]) -> None:
    """
    Execute comprehensive bottleneck analysis for medication preparation workflow optimization.
    
    Identifies specific delays in medication preparation process that contribute to TAT
    threshold violations and provides actionable recommendations for pharmacy workflow
    improvement. Essential for healthcare operations supporting patient care throughput
    enhancement and clinical operations excellence through evidence-based optimization.
    
    Bottleneck Analysis Components:
    - Step-to-step delay quantification identifying specific workflow bottlenecks affecting TAT
    - Operational context analysis examining shift, floor, and staffing impact on delays
    - Temporal pattern identification revealing time-based optimization opportunities
    - Clinical workflow recommendations supporting pharmacy operations improvement initiatives
    - Comprehensive reporting with visualization supporting stakeholder communication
    
    Healthcare Quality Impact:
    - TAT threshold compliance assessment supporting clinical quality monitoring standards
    - Workflow efficiency evaluation enabling targeted improvement interventions and initiatives
    - Resource allocation optimization based on temporal patterns and operational demands
    - Patient care throughput enhancement through evidence-based bottleneck elimination
    - Clinical operations excellence support through comprehensive workflow understanding
    
    Args:
        df: Prepared healthcare dataset with delay features and operational context
           supporting comprehensive bottleneck analysis and workflow optimization.
        output_dirs: Directory structure for healthcare analytics reporting and
                    visualization artifacts supporting clinical documentation requirements.
    """
    logger.info("🔍 Starting comprehensive medication preparation bottleneck analysis...")
    
    # Initialize bottleneck analyzer with healthcare quality threshold requirements
    analyzer = BottleneckAnalyzer(tat_threshold=60.0)  # Per project requirements
    
    # Generate comprehensive bottleneck analysis report for healthcare stakeholders
    logger.info("📊 Generating comprehensive bottleneck analysis report...")
    report_path = output_dirs['bottleneck_analysis'] / "bottleneck_report.json"
    report = analyzer.generate_bottleneck_report(df, save_path=str(report_path))
    
    # Display comprehensive analysis results for clinical stakeholders
    print(f"\n" + "="*80)
    print("🏥 MEDICATION PREPARATION TAT BOTTLENECK ANALYSIS")
    print("="*80)
    print("Pharmacy Workflow Optimization for Enhanced Patient Care Throughput")
    print("="*80)
    
    # Healthcare dataset overview with clinical context
    summary = report['dataset_summary']
    print(f"\n📊 Analysis Overview:")
    print(f"   • Medication Orders Analyzed: {summary['total_orders']:,}")
    print(f"   • Average TAT: {summary['avg_tat']:.1f} minutes")
    print(f"   • TAT Threshold (60 min) Violations: {summary['tat_violation_rate']:.1%}")
    print(f"   • Clinical Impact: {int(summary['total_orders'] * summary['tat_violation_rate']):,} orders exceed threshold")
    
    # Primary bottleneck identification in medication preparation workflow
    step_bottlenecks = report['step_bottlenecks']
    primary = step_bottlenecks.get('primary_bottleneck')
    
    if primary:
        primary_name = primary.replace('delay_', '').replace('_', ' → ').title()
        print(f"\n🚨 Primary Workflow Bottleneck: {primary_name}")
        print(f"   Critical intervention point for TAT improvement")
        
        print(f"\n📈 Top 5 Medication Preparation Bottlenecks:")
        print(f"   {'Workflow Step':<35} | {'Median Delay':<12} | {'Threshold Violations'}")
        print(f"   {'-'*35} | {'-'*12} | {'-'*18}")
        
        for i, (step, metrics) in enumerate(list(step_bottlenecks['step_analysis'].items())[:5], 1):
            step_name = step.replace('delay_', '').replace('_', ' → ').title()
            median = metrics['median_delay']
            violation = metrics['violation_rate']
            print(f"   {i}. {step_name:<33} | {median:>6.1f} min    | {violation:>6.1%}")
    else:
        print(f"\n⚠️  No primary bottleneck identified - multiple workflow issues detected")
    
    # Operational context analysis for targeted interventions
    conditional = report['conditional_bottlenecks']
    if conditional:
        print(f"\n⚕️  Operational Context Analysis (Intervention Opportunities):")
        
        for condition, analysis in conditional.items():
            if analysis and len(analysis) > 1:
                # Identify highest and lowest performing operational contexts
                worst = max(analysis.items(), key=lambda x: x[1]['avg_tat'])
                best = min(analysis.items(), key=lambda x: x[1]['avg_tat'])
                improvement_opportunity = worst[1]['avg_tat'] - best[1]['avg_tat']
                
                print(f"   • {condition.replace('_', ' ').title()}:")
                print(f"     - Highest TAT: {worst[0]} ({worst[1]['avg_tat']:.1f} min avg)")
                print(f"     - Lowest TAT:  {best[0]} ({best[1]['avg_tat']:.1f} min avg)")
                print(f"     - Improvement Potential: {improvement_opportunity:.1f} minutes")
    
    # Temporal workflow patterns affecting patient care delivery
    temporal = report['temporal_bottlenecks']
    if 'hourly' in temporal and temporal['hourly']:
        hourly_data = temporal['hourly']
        if len(hourly_data) > 1:
            peak_hour = max(hourly_data.items(), key=lambda x: x[1]['avg_tat'])
            optimal_hour = min(hourly_data.items(), key=lambda x: x[1]['avg_tat'])
            
            print(f"\n⏰ Temporal Workflow Patterns:")
            print(f"   • Peak Delay Period: {peak_hour[0]:02d}:00 hours ({peak_hour[1]['avg_tat']:.1f} min avg)")
            print(f"   • Optimal Period: {optimal_hour[0]:02d}:00 hours ({optimal_hour[1]['avg_tat']:.1f} min avg)")
            print(f"   • Staffing Optimization Opportunity: Focus resources during peak periods")
    
    # Specialized analysis for identified primary bottlenecks
    if primary:
        if 'nurse_to_prep' in primary.lower():
            logger.info("🔍 Conducting specialized nurse-to-prep bottleneck analysis...")
            analyzer.generate_detailed_nurse_prep_analysis(df)
            print(f"   • Specialized nurse workflow analysis completed")
        
        if 'prep_to_validation' in primary.lower():
            print(f"   • Focus area: Pharmacy preparation completion and validation workflow")
        
        if 'validation_to_dispatch' in primary.lower():
            print(f"   • Focus area: Second validation and floor dispatch coordination")

    # Seasonal and trend analysis for long-term workflow planning
    logger.info("📅 Analyzing seasonal patterns for workflow planning...")
    analyzer.analyze_seasonal_patterns(df)

    # Evidence-based workflow optimization recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 EVIDENCE-BASED WORKFLOW OPTIMIZATION RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
    else:
        # Generate standard recommendations based on analysis
        print(f"\n💡 WORKFLOW OPTIMIZATION RECOMMENDATIONS:")
        print(f"   1. Target primary bottleneck: {primary_name if primary else 'Multiple workflow points'}")
        print(f"   2. Implement temporal staffing adjustments during peak delay periods")
        print(f"   3. Establish continuous monitoring of TAT threshold compliance")
        print(f"   4. Develop staff training programs focusing on identified bottleneck areas")
    
    # Generate comprehensive bottleneck visualizations for stakeholder communication
    logger.info("📊 Generating bottleneck analysis visualizations...")
    
    try:
        # Primary bottleneck intensity heatmap for workflow understanding
        fig = analyzer.plot_bottleneck_heatmap(
            df,
            save_path=str(output_dirs['figures'] / "medication_prep_bottleneck_heatmap.png")
        )
        
        if fig:
            logger.info("✅ Bottleneck heatmap generated successfully")
            print(f"   ✅ Workflow bottleneck heatmap: {output_dirs['figures'] / 'medication_prep_bottleneck_heatmap.png'}")
            plt.close(fig)
        
    except Exception as e:
        logger.error(f"❌ Bottleneck heatmap generation failed: {e}")
        print(f"   ❌ Visualization generation issue: {e}")
    
    # Analysis completion summary for healthcare stakeholders
    print(f"\n" + "="*80)
    print("✅ BOTTLENECK ANALYSIS COMPLETE")
    print("="*80)
    print(f"📁 Comprehensive Report: {report_path}")
    print(f"📊 Visualizations Directory: {output_dirs['figures']}")
    
    # Verify generated artifacts for healthcare documentation
    viz_files = list(output_dirs['figures'].glob('*.png'))
    report_files = list(output_dirs['bottleneck_analysis'].glob('*.json'))
    
    print(f"📈 Analysis Artifacts Generated:")
    print(f"   • Visualization files: {len(viz_files)}")
    print(f"   • Analysis reports: {len(report_files)}")
    
    if viz_files:
        print(f"💡 Additional delay visualizations available via: get_delay_plots.py")
    
    # Clinical impact summary and next steps
    focus_area = primary.replace('delay_', '').replace('_', ' → ').title() if primary else 'Multiple bottlenecks'
    print(f"\n🎯 CLINICAL FOCUS AREA: {focus_area}")
    print(f"📋 Next Steps:")
    print(f"   1. Share analysis with pharmacy operations team")
    print(f"   2. Implement targeted interventions for identified bottlenecks")
    print(f"   3. Establish monitoring for TAT threshold compliance")
    print(f"   4. Schedule follow-up analysis to measure improvement")
    
    logger.info("🎉 Comprehensive bottleneck analysis completed successfully")

def main():
    """
    Execute comprehensive TAT bottleneck analysis for medication preparation workflow optimization.
    
    Main entry point for healthcare analytics bottleneck identification supporting
    pharmacy workflow optimization and patient care throughput improvement. Analyzes
    medication preparation delays to identify specific workflow inefficiencies and
    provide evidence-based recommendations for clinical operations excellence.
    
    Clinical Workflow Analysis:
    - Comprehensive bottleneck identification in medication preparation process workflows
    - Operational context analysis examining shift, staffing, and facility factors affecting TAT
    - Temporal pattern analysis revealing time-based optimization opportunities for scheduling
    - Evidence-based recommendations supporting pharmacy workflow improvement initiatives
    - Healthcare quality assessment focused on 60-minute TAT threshold compliance requirements
    
    Returns:
        int: Exit code indicating analysis success (0) or failure modes for
        automated healthcare pipeline integration and monitoring workflows.
    
    Exit Codes:
        0: Successful bottleneck analysis completion with comprehensive reporting
        1: Analysis interrupted or general failure affecting healthcare workflow insights

    """
    parser = argparse.ArgumentParser(
        description="TAT Bottleneck Analysis - Identify medication preparation workflow delays",
        epilog="Essential for pharmacy workflow optimization and patient care throughput improvement",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable detailed logging output for comprehensive analysis troubleshooting'
    )
    
    args = parser.parse_args()
    
    # Configure logging level based on user requirements
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔧 Verbose logging enabled for comprehensive analysis tracking")
    
    try:
        print("🏥 TAT BOTTLENECK ANALYSIS SYSTEM")
        print("Advanced Analytics for Pharmacy Workflow Optimization")
        print("Supporting Patient Care Throughput Enhancement")
        print()
        
        # Initialize healthcare analytics output structure
        logger.info("🔧 Setting up healthcare analytics infrastructure...")
        output_dirs = setup_output_directories()
        
        # Load and prepare medication preparation TAT data
        logger.info("📊 Loading medication preparation dataset...")
        df_prepared = load_and_prepare_data()
        
        # Execute comprehensive bottleneck analysis workflow
        logger.info("🔍 Executing bottleneck analysis...")
        perform_bottleneck_analysis(df_prepared, output_dirs)
        
        logger.info("🎉 Bottleneck analysis pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("⚠️  Analysis interrupted by user")
        print("\n⚠️  Analysis interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Bottleneck analysis failed: {e}")
        print(f"\n❌ Analysis failed: {e}")
        
        if args.verbose:
            logger.exception("Detailed error information:")
            raise
        else:
            print("   Use --verbose flag for detailed error information")
        
        return 1

if __name__ == "__main__":
    """
    Execute TAT bottleneck analysis for medication preparation workflow optimization.
    
    Provides command-line interface for healthcare analytics supporting pharmacy
    operations improvement and patient care throughput enhancement through
    comprehensive bottleneck identification and evidence-based recommendations.

    """
    sys.exit(main())