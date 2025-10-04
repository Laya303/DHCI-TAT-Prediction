"""
Primary interface for TAT dataset exploratory data analysis.

Provides comprehensive analysis and reporting for medication preparation
workflow optimization.
"""
import os
import sys
from typing import Any, Dict, Optional

import pandas as pd

from .summary_config import SummaryConfig
from . import summary_tables as T
from .summary_render import ConsoleRenderer, HtmlRenderer

# Ensure UTF-8 console output for healthcare environments supporting Unicode
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    # Graceful fallback for legacy healthcare terminal systems
    # ASCII rendering will be used automatically based on SummaryConfig detection
    pass


class DataSummary:
    """
    Comprehensive exploratory data analysis for TAT datasets.
    
    Provides statistical summaries, visualizations, and reporting
    for medication preparation workflow analysis.
    """

    def __init__(self, config: Optional[SummaryConfig] = None):
        """
        Initialize DataSummary with optional configuration.
        
        Args:
            config: Optional SummaryConfig for analysis parameters.
        """
        self.cfg = config or SummaryConfig()
        self._console = ConsoleRenderer()
        self._html = HtmlRenderer()

    # ------- Core Analysis Methods for Pharmacy Workflow Optimization -------
    
    def print_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive console-based TAT analysis for interactive pharmacy team review.
        
        Provides immediate, detailed analysis suitable for pharmacy team interactive
        exploration, shift handoff briefings, and real-time bottleneck identification.
        Optimized for terminal-based healthcare analytics environments and quick
        clinical decision-making support during operational workflow optimization.
        
        Analysis Components:
        - Executive summary with dataset overview and key operational metrics
        - Missing data assessment for workflow timestamp completeness and quality
        - Workflow step analysis for medication preparation process evaluation
        - Operational factor distributions (staffing, shifts, departments, clinical context)
        - TAT  Numerical Features with statistical summaries and distribution patterns
        - Correlation analysis for bottleneck driver identification and improvement targeting
        
        Args:
            df: TAT dataset from pharmacy operations system containing workflow timestamps,
               operational factors, clinical variables, and  Numerical Features
        
        Returns:
            Dict containing comprehensive analysis artifacts:
            - time_table: Workflow timestamp completeness and quality assessment
            - categorical_table: Operational factor analysis for workflow optimization
            - numeric_table: TAT  Numerical Features with embedded distribution data
            - correlations: Feature correlation matrix for bottleneck driver identification
            - missing_table: Data quality assessment for workflow integrity monitoring
            - counts: Executive summary metrics for pharmacy leadership reporting
            - df_processed: Preprocessed dataset for downstream analysis and modeling
        
        Example:
            For pharmacy shift briefing and bottleneck review:
            summary = DataSummary.default()
            analysis = summary.print_report(daily_tat_df)
            
            # Access specific insights for operational decisions
            workflow_issues = analysis['time_table']  # Missing timestamp patterns
            staffing_insights = analysis['categorical_table']  # Shift/credential patterns
            performance_metrics = analysis['numeric_table']  # TAT distributions and trends
            improvement_drivers = analysis['correlations']  # Bottleneck correlation analysis
        
        Note:
            Designed for pharmacy team consumption with clinical terminology and
            healthcare workflow interpretation. Console format suitable for terminal-based
            analysis and integration with automated pharmacy monitoring systems.
        """
        arts = T.build_artifacts(df, self.cfg)

        # Executive summary header for pharmacy operations team orientation
        sep = "=" * 78
        cnt = arts["counts"]
        print(sep)
        print("PHARMACY TAT DATASET ANALYSIS SUMMARY".center(78))
        print(sep)
        print(f"📊 Orders Analyzed: {cnt['rows']:,}  |  📈 Total Features: {cnt['cols']}  |  "
              f"⏰ Workflow Steps: {cnt['time']}  |  🏥 Operational Factors: {cnt['categorical']}  |  "
              f"📋  Numerical Features: {cnt['numeric']}")

        # Data quality assessment - critical for healthcare workflow analysis integrity
        if not arts["missing_table_console"].empty:
            print(f"\n[🔍 DATA QUALITY: MISSING WORKFLOW TIMESTAMPS — {cnt['missing_cols']} features with gaps]")
            print("Missing data assessment for medication preparation workflow integrity and bottleneck analysis")
            with pd.option_context("display.max_rows", 200, "display.max_colwidth", 40):
                print(arts["missing_table_console"].to_string(index=False))

        # Medication preparation workflow timeline analysis
        if not arts["time_table"].empty:
            print(f"\n[⏱️ MEDICATION PREPARATION WORKFLOW TIMELINE — {cnt['time']} sequential steps]")
            print("Timestamp completeness analysis from physician order to patient administration")
            print(arts["time_table"].to_string(index=False))

        # Operational context and staffing factor analysis
        if not arts["categorical_table"].empty:
            print(f"\n[🏥 OPERATIONAL CONTEXT & STAFFING ANALYSIS — {cnt['categorical']} factors]")
            print("Shift patterns, credentials, departments, and clinical factors affecting TAT performance")
            print(arts["categorical_table"].to_string(index=False))

        # TAT  Numerical Features and clinical indicators
        if not arts["numeric_table"].empty:
            print(f"\n[📈 TAT  Numerical Features & CLINICAL INDICATORS — {cnt['numeric']} measures]")
            print("Turnaround times, queue metrics, occupancy rates, laboratory values, and operational indicators")
            # Remove internal distribution artifacts for clean console display
            display_table = arts["numeric_table"].drop(
                columns=["_dist_counts", "_dist_labels"], errors="ignore"
            )
            print(display_table.to_string(index=False))

        # TAT driver correlation analysis for bottleneck identification and optimization
        if not arts["correlations"].empty:
            print("\n[🔗 TAT DRIVER CORRELATION ANALYSIS — Bottleneck identification for workflow optimization]")
            print("Pearson correlations for medication preparation delay drivers and improvement opportunities")
            
            # Generate actionable correlation insights for pharmacy workflow optimization
            correlation_pairs = T.correlation_pairs_table(
                arts["df_processed"], self.cfg, min_abs=0.10, top_k=50
            )
            
            if not correlation_pairs.empty:
                print("Top correlation pairs for workflow optimization targeting (|r| ≥ 0.10):")
                with pd.option_context("display.max_rows", 200, "display.max_colwidth", 40):
                    print(correlation_pairs.round({"r": 3, "|r|": 3}).to_string(index=False))
                
                print("\n💡 Focus on highest absolute correlations for maximum TAT improvement impact")
            else:
                print("No correlation pairs found meeting clinical significance threshold (|r| ≥ 0.10)")

        print(sep)
        print("✅ Analysis Complete: Review findings for workflow bottlenecks and optimization opportunities")
        print("📋 Next Steps: Focus on high-correlation drivers and missing data improvement initiatives")
        print(sep)
        
        return arts

    def to_html(self, df: pd.DataFrame) -> str:
        """
        Generate comprehensive HTML report for TAT analysis stakeholder communication.
        
        Produces professional, standalone HTML reports suitable for pharmacy leadership
        review, clinical stakeholder presentations, healthcare analytics dashboards,
        and regulatory documentation. Optimized for medication preparation workflow
        analysis with embedded visualizations and clinical interpretation guidance.
        
        Report Features:
        - Executive summary with key operational metrics and dataset overview
        - Interactive histogram visualizations for TAT distribution analysis
        - Color-coded data quality assessment with healthcare-appropriate thresholds
        - Professional styling suitable for clinical stakeholder consumption
        - Embedded CSS for standalone deployment across healthcare IT environments
        - Correlation heatmaps for bottleneck driver identification and prioritization
        
        Args:
            df: TAT dataset from pharmacy operations system for comprehensive analysis
        
        Returns:
            str: Complete HTML document with embedded styling ready for:
            - File system storage and stakeholder email distribution
            - Web server deployment for pharmacy leadership dashboard integration
            - Clinical presentation and healthcare quality improvement documentation
            - Integration with automated TAT monitoring and reporting systems
        
        Example:
            For pharmacy leadership reporting and clinical stakeholder presentations:
            summary = DataSummary.default()
            html_report = summary.to_html(monthly_tat_df)
            
            # Save for stakeholder distribution
            with open('pharmacy_tat_analysis_report.html', 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            # Deploy to healthcare analytics dashboard
            dashboard.deploy_report(html_report, 'tat_monthly_analysis')
        
        Note:
            HTML output includes professional healthcare analytics styling with
            clinical terminology and workflow optimization focus. Suitable for
            both technical and clinical audiences in pharmacy operations environments.
        """
        return self._html.to_html(df, self.cfg)

    def report(
        self,
        df: pd.DataFrame,
        save_dir: Optional[str] = None,
        export: Optional[str] = "html",
        console: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute comprehensive TAT analysis with multi-format output for diverse stakeholder needs.
        
        Orchestrates complete pharmacy workflow analysis pipeline with flexible output
        options supporting interactive team analysis, executive reporting, automated
        monitoring systems, and downstream model development workflows. Designed for
        production deployment in healthcare analytics environments with robust error handling.
        
        Analysis Pipeline:
        1. Healthcare data preprocessing with timestamp standardization and encoding normalization
        2. Column classification with clinical domain knowledge and TAT workflow context
        3. Comprehensive statistical analysis with healthcare-appropriate thresholds
        4. Multi-format report generation (console, HTML, CSV) for stakeholder distribution
        5. Artifact persistence for downstream analytics and model development integration
        
        Args:
            df: TAT dataset from pharmacy operations system for comprehensive analysis
            save_dir: Optional directory path for report persistence and stakeholder distribution.
                     Creates directory structure suitable for healthcare analytics file organization.
            export: Report format for stakeholder consumption ("html" for rich visualizations).
                   HTML format includes embedded styling and interactive elements.
            console: Enable detailed console output for interactive pharmacy team analysis.
                    Suitable for shift briefings and real-time bottleneck identification.
        
        Returns:
            Dict containing comprehensive analysis artifacts for downstream consumption:
            - Complete analysis tables for workflow optimization and clinical interpretation
            - Processed dataset for model development and advanced analytics workflows
            - Executive summary metrics for pharmacy leadership reporting and KPI tracking
        
        File Outputs (when save_dir specified):
            - summary_time.csv: Workflow timestamp analysis for data quality monitoring
            - summary_categorical.csv: Operational factor distributions for staffing insights
            - summary_numeric.csv: TAT  Numerical Features for trend analysis and benchmarking
            - summary.html: Comprehensive stakeholder report with visualizations and insights
        
        Example:
            For comprehensive pharmacy workflow analysis with stakeholder distribution:
            summary = DataSummary.default()
            
            # Full analysis with multi-format output
            artifacts = summary.report(
                tat_df,
                save_dir="pharmacy_tat_analysis_2025_q1",
                export="html",
                console=True  # For immediate team review
            )
            
            # Access structured artifacts for advanced analytics
            workflow_analysis = artifacts['time_table']
            performance_metrics = artifacts['numeric_table']
            correlation_matrix = artifacts['correlations']
            
            # Integrate with downstream model development
            model_features = artifacts['df_processed']
        
        Note:
            Production-ready method suitable for automated TAT monitoring pipelines,
            scheduled pharmacy analytics workflows, and integration with healthcare
            analytics platforms. Robust error handling ensures reliable operation
            in production healthcare IT environments with diverse data quality patterns.
        """
        # Generate comprehensive analysis artifacts with healthcare-specific processing
        arts = T.build_artifacts(df, self.cfg)

        # Interactive console analysis for pharmacy team immediate review
        if console:
            print("🏥 Generating comprehensive TAT analysis for pharmacy workflow optimization...")
            self._console.print_from_arts(arts)
            print("📊 Console analysis complete. See detailed insights above.")

        # Multi-format report persistence for stakeholder distribution and archival
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"💾 Saving analysis artifacts to: {save_dir}")
            
            # Structured CSV exports for downstream analytics and model development
            arts["time_table"].to_csv(
                os.path.join(save_dir, "summary_time.csv"), 
                index=False
            )
            arts["categorical_table"].to_csv(
                os.path.join(save_dir, "summary_categorical.csv"), 
                index=False
            )
            arts["numeric_table"].to_csv(
                os.path.join(save_dir, "summary_numeric.csv"), 
                index=False
            )
            
            # Rich HTML report for stakeholder communication and presentation
            if export == "html":
                html_path = os.path.join(save_dir, "summary.html")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(self._html.to_html(df, self.cfg))
                print(f"📄 Comprehensive HTML report saved: {html_path}")
                print("🎯 Report ready for pharmacy leadership review and clinical stakeholder distribution")

        return arts

    @classmethod
    def default(cls) -> "DataSummary":
        """
        Create DataSummary instance with healthcare-optimized default configuration.
        
        Convenience factory method for pharmacy analytics teams requiring immediate
        TAT analysis capabilities without custom configuration. Provides sensible
        defaults optimized for medication preparation workflow analysis, healthcare
        data quality assessment, and clinical stakeholder reporting.
        
        Default Configuration Includes:
        - 60-minute TAT SLA threshold for pharmacy operations benchmark
        - Healthcare analytics percentiles (p1, p5, p25, p50, p75, p95, p99)
        - Clinical data quality thresholds (10% warning, 25% critical missing data)
        - Pharmacy workflow column classifications and healthcare domain knowledge
        - Professional visualization settings for clinical stakeholder consumption
        
        Returns:
            DataSummary: Configured instance ready for immediate TAT dataset analysis
        
        Example:
            For quick pharmacy workflow analysis with industry-standard parameters:
            summary = DataSummary.default()
            artifacts = summary.print_report(tat_df)
            
            # Equivalent to:
            config = SummaryConfig()  # Healthcare-optimized defaults
            summary = DataSummary(config)
            artifacts = summary.print_report(tat_df)
        
        Note:
            Default configuration suitable for most pharmacy TAT analysis use cases
            including bottleneck identification, workflow optimization, and clinical
            performance monitoring. Custom configurations available for specialized
            healthcare analytics requirements or advanced model development workflows.
        """
        return cls()