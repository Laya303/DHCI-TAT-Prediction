"""
Production-Grade TAT Analysis Report Renderers

Comprehensive rendering system for pharmacy turnaround time exploratory data analysis
reports. Provides both console and HTML output formats optimized for healthcare
stakeholder consumption and clinical decision-making workflows.

Key Components:
- ConsoleRenderer: Streamlined text output for pharmacy team interactive analysis
- HtmlRenderer: Rich HTML reports with embedded visualizations for leadership review
- Healthcare-optimized formatting for medication preparation workflow insights
- Production-ready rendering suitable for automated TAT monitoring dashboards

"""
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .summary_config import SummaryConfig
from . import summary_tables as T

# Ensure UTF-8 console output for healthcare environments supporting Unicode
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    # Graceful fallback for legacy healthcare terminal systems
    # ASCII rendering will be used automatically based on SummaryConfig detection
    pass


class ConsoleRenderer:
    """
    Plain-text console renderer for interactive TAT analysis workflows.
    
    Optimized for pharmacy team interactive data exploration and quick bottleneck
    identification. Provides compact, readable summaries suitable for terminal-based
    healthcare analytics environments and automated monitoring system outputs.
    
    Key Features:
    - Compact tabular format for workflow step analysis
    - Missing data highlighting for data quality assessment
    - TAT correlation insights for bottleneck identification
    - Clinical terminology and healthcare-focused presentation
    """

    @staticmethod
    def print_from_arts(arts: Dict[str, Any]) -> None:
        """
        Render comprehensive TAT analysis summary to console from processed artifacts.

        Displays structured analysis of medication preparation workflow data including:
        - Missing data quality assessment for workflow timestamp completeness
        - Temporal column analysis for step-by-step delay identification
        - Categorical distributions for operational factors (shift, credentials, departments)
        - Numeric summaries for TAT metrics, queue lengths, and clinical indicators
        - Correlation matrix for TAT driver identification and bottleneck analysis

        Args:
            arts: Processed analysis artifacts from build_artifacts containing:
                - missing_table: Data quality assessment results
                - time_table: Workflow timestamp analysis
                - categorical_table: Operational factor distributions  
                - numeric_table: TAT metrics and clinical measure summaries
                - correlations: Feature correlation matrix for driver identification

        Note:
            Designed for pharmacy team consumption with clinical terminology and
            healthcare workflow interpretation. Compact format suitable for
            terminal-based analysis and automated monitoring system integration.
        """
        sep = "=" * 78
        print(sep)
        print("TAT DATASET ANALYSIS SUMMARY".center(78))
        print(sep)

        # Data quality assessment - critical for healthcare workflow analysis
        mt = arts.get("missing_table", pd.DataFrame())
        if not mt.empty:
            print("\n[DATA QUALITY: MISSING WORKFLOW TIMESTAMPS (>0%)]")
            print("Missing data assessment for medication preparation workflow integrity")
            # Enhanced readability for healthcare team review
            with pd.option_context("display.max_rows", 200, "display.max_colwidth", 40):
                print(mt.to_string(index=False))

        # Workflow timestamp analysis for step-by-step bottleneck identification
        tt = arts.get("time_table", pd.DataFrame())
        if not tt.empty:
            print("\n[MEDICATION PREPARATION WORKFLOW TIMESTAMPS]")
            print("Sequential workflow steps from physician order to patient administration")
            print(tt.to_string(index=False))

        # Operational factors affecting TAT performance
        ct = arts.get("categorical_table", pd.DataFrame())
        if not ct.empty:
            print("\n[OPERATIONAL FACTORS: STAFFING & WORKFLOW CONTEXT]")
            print("Shift patterns, credentials, departments affecting medication preparation TAT")
            print(ct.to_string(index=False))

        # TAT metrics and clinical performance indicators
        nt = arts.get("numeric_table", pd.DataFrame())
        if not nt.empty:
            print("\n[Numerical Features & CLINICAL INDICATORS]")
            print("Turnaround times, queue metrics, occupancy rates, and laboratory values")
            # Remove distribution artifacts for clean console display
            display_cols = [col for col in nt.columns if not col.startswith('_dist_')]
            print(nt[display_cols].to_string(index=False))

        # TAT driver correlation analysis for bottleneck identification
        corr = arts.get("correlations", pd.DataFrame())
        if not corr.empty:
            print("\n[TAT DRIVER CORRELATION ANALYSIS]")
            print("Pearson correlations for medication preparation bottleneck identification")
            print("Focus on TAT_minutes relationships for workflow optimization insights")
            print(corr.round(3).to_string())

        print(sep)
        print("Analysis Complete: Review for workflow bottlenecks and optimization opportunities")
        print(sep)


class HtmlRenderer:
    """
    Professional HTML report renderer for TAT analysis stakeholder communication.
    
    Generates comprehensive, visually-rich reports suitable for pharmacy leadership
    review, clinical stakeholder presentations, and healthcare analytics dashboards.
    Optimized for medication preparation workflow analysis and bottleneck identification.
    
    Key Features:
    - Embedded CSS styling for standalone report deployment
    - Interactive histogram visualizations for TAT distribution analysis
    - Color-coded missing data assessment for workflow data quality
    - Correlation heatmaps for TAT driver identification
    - Healthcare-focused terminology and clinical interpretation guidance
    - Production-ready output suitable for automated reporting pipelines
    """

    # Professional healthcare analytics stylesheet for stakeholder consumption
    _STYLE = """
<meta charset="utf-8">
<style>
body { 
    font-family: ui-sans-serif, system-ui, Segoe UI, Roboto, Arial, sans-serif; 
    padding: 24px; 
    background: #fafbfc;
    color: #1f2937;
    line-height: 1.5;
}

h1 { 
    margin: 0 0 12px 0; 
    text-align: center; 
    color: #1e40af;
    font-size: 2.2em;
    font-weight: 700;
}

h2 { 
    margin: 32px 0 16px 0; 
    text-align: center; 
    color: #1e40af;
    font-size: 1.4em;
    font-weight: 600;
}

.meta { 
    color: #4b5563; 
    margin: 12px 0 24px; 
    text-align: center; 
    font-size: 1.1em;
    background: #ffffff;
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.section { 
    padding: 16px 0 8px; 
    margin: 20px 0;
}

.section h2::after {
    content: "";
    display: block;
    width: 160px;
    height: 4px;
    margin: 12px auto 0;
    background: linear-gradient(90deg, #e5e7eb, #3b82f6, #e5e7eb);
    border-radius: 2px;
    opacity: 0.7;
}

.tablewrap { 
    overflow-x: auto; 
    padding: 16px 0; 
    background: #ffffff;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    margin: 16px 0;
}

.tablewrap > table { margin: 0 auto; }

table.dataframe { 
    border-collapse: collapse; 
    font-size: 14px; 
    font-variant-numeric: tabular-nums;
}

table.dataframe.compact { 
    width: max-content; 
    table-layout: auto; 
    margin: 0 auto; 
}

table.dataframe.compact th,
table.dataframe.compact td {
    border-bottom: 1px solid #e5e7eb;
    padding: 10px 14px;
    white-space: nowrap;
    text-align: center;
    vertical-align: middle;
}

table.dataframe.compact th { 
    background: #f8fafc; 
    font-weight: 600;
    color: #374151;
    border-bottom: 2px solid #d1d5db;
}

table.dataframe.compact tbody tr:nth-child(even) { background: #fafbfc; }
table.dataframe.compact tbody tr:hover { 
    background: #eff6ff; 
    transition: background-color 0.2s ease;
}

/* Healthcare-specific table styling */
table.dataframe.time th { background: #fef3c7; color: #92400e; }
table.dataframe.categorical th { background: #dbeafe; color: #1e40af; }
table.dataframe.numeric th { background: #dcfce7; color: #166534; }
table.dataframe.missing th { background: #fee2e2; color: #991b1b; }

table.dataframe.corr td, table.dataframe.corr th { min-width: 80px; font-size: 12px; }
table.dataframe.corr-pairs td, table.dataframe.corr-pairs th { min-width: 120px; }

/* TAT distribution histogram styling */
.vhist { 
    display: flex; 
    align-items: flex-end; 
    gap: 2px; 
    width: 220px; 
    height: 70px;
    background: #ffffff; 
    border-radius: 6px; 
    padding: 8px; 
    box-sizing: border-box;
    margin: 0 auto;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}

.vhist.empty { 
    background: repeating-linear-gradient(45deg, #f9fafb, #f9fafb 8px, #f3f4f6 8px, #f3f4f6 16px);
    border: 1px dashed #d1d5db;
}

.vhist .bar { 
    flex: 1 1 auto; 
    background: linear-gradient(to top, #3b82f6, #60a5fa);
    border-top-left-radius: 2px; 
    border-top-right-radius: 2px;
    transition: opacity 0.2s ease;
}

.vhist .bar:hover { opacity: 0.8; }

/* Healthcare data quality indicators */
.miss-good { color: #059669; font-weight: 600; }
.miss-bad { color: #dc2626; font-weight: 600; }
.miss-warning { color: #d97706; font-weight: 600; }

/* TAT correlation highlighting */
.corr-target { 
    background: #fef3c7 !important; 
    font-weight: 700; 
    border-radius: 4px;
    color: #92400e;
}

.corr-target-cell { 
    background: #fffbeb !important; 
    font-weight: 600; 
    color: #92400e;
}

/* Professional healthcare report styling */
.insight-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    padding: 16px;
    margin: 20px 0;
    border-radius: 0 6px 6px 0;
    font-style: italic;
    color: #1e40af;
}
</style>
"""

    @staticmethod
    def _vhist(counts: List[int], labels: Optional[List[str]]) -> str:
        """
        Generate interactive HTML histogram visualization for TAT distribution analysis.

        Creates compact, visually-appealing distribution charts suitable for pharmacy
        stakeholder reports and clinical dashboard integration. Optimized for displaying
        medication preparation timing patterns and bottleneck identification.

        Visualization Features:
        - Responsive bar scaling relative to maximum count for consistent comparison
        - Tooltip information showing bin ranges and frequencies
        - Professional healthcare analytics styling with gradient effects
        - Empty state handling for robust report generation

        Args:
            counts: Integer frequency counts per histogram bin
            labels: Optional bin range labels for tooltip display

        Returns:
            str: HTML div element containing interactive histogram visualization

        Example:
            For TAT_minutes distribution bins:
            - Tooltip: "(20, 40] — 350 orders"
            - Visual: Scaled bars showing relative frequency distribution
        """
        if not counts:
            return '<div class="vhist empty" title="No data available for distribution analysis"></div>'
        
        mx = max(counts) or 1
        bars = []
        for i, c in enumerate(counts):
            # Scale height for 70px container with minimum visibility (3px)
            h = max(3.0, 66.0 * (c / mx))
            
            # Generate informative tooltip for clinical interpretation
            if labels and i < len(labels):
                bin_range = labels[i]
                tooltip = f"{bin_range} — {c:,} orders"
            else:
                tooltip = f"Bin {i+1} — {c:,} orders"
            
            bars.append(f'<div class="bar" style="height:{h:.1f}px" title="{tooltip}"></div>')
        
        return f'<div class="vhist" title="Distribution Analysis">{"".join(bars)}</div>'

    @staticmethod
    def _color_missing_column(df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Apply healthcare data quality color coding to missing data columns.

        Transforms numeric missing data percentages into color-coded HTML spans
        for immediate visual assessment of workflow data completeness. Essential
        for pharmacy team identification of data collection gaps affecting
        medication preparation bottleneck analysis.

        Color Coding Strategy:
        - Green (miss-good): 0% missing - excellent data quality
        - Orange (miss-warning): 1-15% missing - acceptable healthcare range  
        - Red (miss-bad): >15% missing - critical data quality issue

        Args:
            df_in: DataFrame potentially containing missing data percentage columns

        Returns:
            pd.DataFrame: Enhanced DataFrame with color-coded missing data indicators

        Note:
            Healthcare datasets typically show 5-15% missingness in workflow
            timestamps due to EHR integration challenges and manual entry gaps.
        """
        dfc = df_in.copy()
        colname = None
        
        # Detect missing data columns with flexible naming
        for c in dfc.columns:
            if c.lower() in ("miss%", "missing%", "missing_pct", "missing", "missing_rate"):
                colname = c
                break
        
        if colname is None:
            return dfc

        def format_missing_indicator(v):
            try:
                # Handle percentage strings and numeric values
                fv = float(str(v).replace("%", "")) if isinstance(v, str) else float(v)
            except (ValueError, TypeError):
                return v
            
            # Healthcare data quality thresholds
            if fv == 0:
                cls = "miss-good"
            elif fv <= 15.0:  # Acceptable range for healthcare workflow data
                cls = "miss-warning" 
            else:  # Critical data quality issue
                cls = "miss-bad"
            
            return f'<span class="{cls}">{fv:.1f}%</span>'

        dfc[colname] = dfc[colname].map(format_missing_indicator)
        return dfc

    @staticmethod
    def _add_table_classes(html: str, extra: str) -> str:
        """
        Enhance pandas HTML table output with healthcare analytics styling classes.

        Post-processes pandas to_html output to apply professional healthcare
        report styling suitable for pharmacy leadership consumption and clinical
        stakeholder presentations.

        Args:
            html: Raw pandas HTML table output
            extra: Additional CSS class for table-specific styling (time, categorical, numeric, etc.)

        Returns:
            str: Enhanced HTML with professional healthcare analytics styling
        """
        return html.replace(
            '<table border="1" class="dataframe">',
            f'<table border="1" class="dataframe compact {extra}">'
        )

    @staticmethod
    def _wrap(html: str) -> str:
        """
        Wrap table HTML in responsive container for cross-device healthcare analytics.

        Ensures professional presentation across different healthcare IT environments
        from desktop workstations to tablet-based clinical review systems.

        Args:
            html: Table HTML to wrap

        Returns:
            str: Responsive table container with healthcare styling
        """
        return f'<div class="tablewrap">{html}</div>'

    def to_html(self, df: pd.DataFrame, cfg: SummaryConfig) -> str:
        """
        Generate comprehensive HTML report for TAT dataset analysis and pharmacy workflow optimization.

        Produces production-ready HTML reports suitable for:
        - Pharmacy leadership review and strategic planning
        - Clinical stakeholder presentations and workflow optimization meetings  
        - Healthcare analytics dashboards and automated monitoring systems
        - Regulatory documentation and quality improvement initiatives

        Report Components:
        - Executive summary with dataset overview and key metrics
        - Workflow timestamp analysis for step-by-step bottleneck identification
        - Operational factor distributions (staffing, shifts, departments)
        - TAT  Numerical Features with embedded distribution visualizations
        - Data quality assessment with color-coded missing data indicators
        - Correlation analysis for TAT driver identification and optimization opportunities

        Args:
            df: TAT dataset from pharmacy operations system
            cfg: Analysis configuration with healthcare-specific parameters

        Returns:
            str: Complete HTML document with embedded CSS styling ready for:
                - File system storage and stakeholder distribution
                - Web server deployment for dashboard integration
                - Email-based report distribution to pharmacy leadership
                - Integration with healthcare analytics platforms

        Example:
            html_report = renderer.to_html(tat_df, pharmacy_config)
            with open('tat_analysis_report.html', 'w') as f:
                f.write(html_report)
        """
        # Build comprehensive analysis artifacts from TAT dataset
        arts = T.build_artifacts(df, cfg)
        cnt = arts["counts"]

        # Enhance numeric table with interactive distribution visualizations
        num_tbl = arts["numeric_table"].copy()
        if not num_tbl.empty:
            distribution_charts: List[str] = []
            for _, row in num_tbl.iterrows():
                counts = row.get("_dist_counts", [])
                labels = row.get("_dist_labels", [])
                
                if isinstance(counts, list) and counts:
                    distribution_charts.append(self._vhist(counts, labels if isinstance(labels, list) else []))
                else:
                    distribution_charts.append('<div class="vhist empty" title="Insufficient numeric data for distribution analysis"></div>')
            
            # Replace original distribution column with interactive visualizations
            num_tbl["distribution"] = distribution_charts
            num_tbl = num_tbl.drop(columns=["_dist_counts", "_dist_labels"], errors="ignore")

        # Initialize report with professional healthcare styling
        report_parts: List[str] = [self._STYLE, "<h1>Pharmacy TAT Analysis Report</h1>"]

        # Executive summary with key operational metrics
        report_parts.append(
            f"<div class='meta'>"
            f"<strong>📊 Dataset Overview:</strong> {cnt['rows']:,} medication orders analyzed &nbsp;•&nbsp; "
            f"<strong>📈 Total Features:</strong> {cnt['cols']} variables &nbsp;•&nbsp; "
            f"<strong>⏰ Workflow Steps:</strong> {cnt['time']} timestamps &nbsp;•&nbsp; "
            f"<strong>🏥 Operational Factors:</strong> {cnt['categorical']} categories &nbsp;•&nbsp; "
            f"<strong>📋  Numerical Features:</strong> {cnt['numeric']} measures"
            f"</div>"
        )

        # Medication preparation workflow timestamp analysis
        if not arts["time_table"].empty:
            tt = self._color_missing_column(arts["time_table"])
            tt_html = self._add_table_classes(tt.to_html(index=False, escape=False), "time")
            report_parts.append(f"<div class='section'><h2>⏱️ Medication Preparation Workflow Timeline ({cnt['time']} steps)</h2></div>")
            report_parts.append('<div class="insight-box">Sequential timestamps from physician order to patient administration. Missing data indicates workflow tracking gaps requiring data quality improvement.</div>')
            report_parts.append(self._wrap(tt_html))

        # Operational context and staffing factors
        if not arts["categorical_table"].empty:
            ct = self._color_missing_column(arts["categorical_table"])
            ct_html = self._add_table_classes(ct.to_html(index=False, escape=False), "categorical")
            report_parts.append(f"<div class='section'><h2>🏥 Operational Context & Staffing Factors ({cnt['categorical']} factors)</h2></div>")
            report_parts.append('<div class="insight-box">Shift patterns, staff credentials, departments, and operational conditions affecting medication preparation TAT. Focus on dominant categories for workflow optimization.</div>')
            report_parts.append(self._wrap(ct_html))

        # TAT  Numerical Features with distribution analysis
        if not num_tbl.empty:
            nt = self._color_missing_column(num_tbl)
            nt_html = self._add_table_classes(nt.to_html(index=False, escape=False), "numeric")
            report_parts.append(f"<div class='section'><h2>📈 TAT  Numerical Features & Clinical Indicators ({cnt['numeric']} metrics)</h2></div>")
            report_parts.append('<div class="insight-box">Turnaround times, queue metrics, occupancy rates, and laboratory values. Distribution charts show data patterns for bottleneck identification.</div>')
            report_parts.append(self._wrap(nt_html))

        # Data quality assessment for workflow integrity
        if not arts["missing_table"].empty:
            mt = self._color_missing_column(arts["missing_table"])
            mt_html = self._add_table_classes(mt.to_html(index=False, escape=False), "missing")
            report_parts.append(f"<div class='section'><h2>🔍 Workflow Data Quality Assessment ({cnt['missing_cols']} incomplete features)</h2></div>")
            report_parts.append('<div class="insight-box">Missing data analysis for medication preparation workflow integrity. Red indicates critical data quality issues requiring immediate attention.</div>')
            report_parts.append(self._wrap(mt_html))

        # TAT driver correlation analysis and bottleneck identification
        corr = arts["correlations"]
        if not corr.empty:
            # Generate lower-triangular correlation matrix for clean visualization
            lower_triangular = corr.copy()
            mask = np.tril(np.ones(lower_triangular.shape), k=0).astype(bool)
            correlation_matrix = lower_triangular.values.copy()
            correlation_matrix[~mask] = np.nan
            lower_triangular = pd.DataFrame(
                correlation_matrix, 
                index=lower_triangular.index, 
                columns=lower_triangular.columns
            ).round(3).replace({np.nan: ""})

            # Apply professional healthcare styling with TAT_minutes highlighting
            lower_html = lower_triangular.to_html(escape=False)
            lower_html = lower_html.replace(
                '<table border="1" class="dataframe">',
                '<table border="1" class="dataframe compact corr">'
            )
            
            # Highlight TAT_minutes as primary target variable for bottleneck analysis
            if "TAT_minutes" in lower_triangular.columns or "TAT_minutes" in lower_triangular.index:
                lower_html = lower_html.replace("<th>TAT_minutes</th>", '<th class="corr-target">TAT_minutes</th>')

            # Generate actionable correlation insights for workflow optimization
            high_correlation_pairs: List[Any] = []
            feature_names = list(corr.columns)
            n_features = len(feature_names)
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    correlation_value = corr.iloc[i, j]
                    if pd.isna(correlation_value):
                        continue
                    
                    abs_correlation = abs(float(correlation_value))
                    if abs_correlation >= 0.10:  # Clinical significance threshold
                        high_correlation_pairs.append((
                            feature_names[i], 
                            feature_names[j], 
                            float(correlation_value), 
                            abs_correlation
                        ))
            
            # Display correlation insights for pharmacy workflow optimization
            correlation_insights_html = ""
            if high_correlation_pairs:
                insights_df = pd.DataFrame(
                    high_correlation_pairs, 
                    columns=["Primary Factor", "Secondary Factor", "Correlation", "Absolute Correlation"]
                )
                insights_df = insights_df.sort_values("Absolute Correlation", ascending=False).head(25).reset_index(drop=True)
                correlation_insights_html = insights_df.round({"Correlation": 3, "Absolute Correlation": 3}).to_html(index=False, escape=False)
                correlation_insights_html = correlation_insights_html.replace(
                    '<table border="1" class="dataframe">',
                    '<table border="1" class="dataframe compact corr-pairs">'
                )

            report_parts.append("<div class='section'><h2>🔗 TAT Driver Correlation Analysis</h2></div>")
            report_parts.append('<div class="insight-box">Correlation matrix for medication preparation bottleneck identification. Focus on TAT_minutes relationships for workflow optimization opportunities.</div>')
            report_parts.append(self._wrap(lower_html))
            
            if correlation_insights_html:
                report_parts.append("<div class='section'><h2>🎯 High-Impact Correlation Insights (|r| ≥ 0.10)</h2></div>")
                report_parts.append('<div class="insight-box">Actionable correlation pairs for pharmacy workflow optimization. Prioritize factors with strongest TAT relationships for maximum improvement impact.</div>')
                report_parts.append(self._wrap(correlation_insights_html))

        # Complete comprehensive HTML report
        return "".join(report_parts)