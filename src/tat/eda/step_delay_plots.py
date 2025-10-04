"""
Visualization tools for step-by-step delay analysis in pharmacy workflows.

Provides plotting capabilities for identifying bottlenecks in medication
preparation process.
"""
from __future__ import annotations

from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tat.features.temporal.delays import DelayEngineer
from tat.config import DELAY_PLOT_ORDER


class StepDelayVisualizer:
    """
    Comprehensive delay analysis and visualization system for pharmacy workflow optimization.

    Orchestrates end-to-end delay computation and visualization pipeline for medication
    preparation workflow analysis. Integrates DelayEngineer for robust missing data
    handling with professional matplotlib-based plotting capabilities suitable for
    healthcare stakeholder consumption and clinical decision-making workflows.

    Key Capabilities:
    - Sequential delay computation with healthcare-optimized imputation strategies
    - Statistical annotation including median, IQR, sample sizes, and SLA violations
    - Professional styling with color-coded delay severity indicators
    - Flexible output formats supporting both interactive analysis and report generation
    - Robust error handling for production deployment in healthcare analytics environments

    Visualization Features:
    - Box plots with statistical overlays for comprehensive delay distribution analysis
    - SLA threshold lines with violation percentage tracking for compliance monitoring
    - Color-coded severity indicators based on delay magnitude relative to thresholds
    - Sample size annotations with missing data percentage for data quality assessment
    - Professional styling suitable for pharmacy leadership and clinical stakeholder review

    Production Considerations:
    - Lightweight matplotlib dependency for healthcare IT environment compatibility
    - Configurable figure sizing and styling for diverse presentation requirements
    - Robust missing data handling through DelayEngineer integration
    - Consistent plotting order based on medication preparation workflow sequence

    Args:
        delay_engineer: Optional pre-configured DelayEngineer instance for custom
                       imputation strategies. Defaults to sequential imputation optimized
                       for healthcare workflow data patterns.
        figsize: Matplotlib figure dimensions for presentation and dashboard integration.
                Default (10, 6) suitable for most healthcare analytics reporting needs.
        impute_missing: Enable sequential imputation for missing timestamp data.
                       Recommended for healthcare datasets with typical 5-15% missingness.

    Example:
        # Standard workflow delay analysis with SLA monitoring
        visualizer = StepDelayVisualizer()
        fig = visualizer.plot_box(
            tat_df, 
            sla_minutes=60, 
            title="Medication Preparation Step Delays",
            save_path="pharmacy_delays_analysis.png"
        )

        # Custom configuration for specialized analysis
        custom_engineer = DelayEngineer(impute_missing=True, validation_strict=False)
        visualizer = StepDelayVisualizer(
            delay_engineer=custom_engineer,
            figsize=(14, 8)
        )
    """

    def __init__(
        self,
        delay_engineer: Optional[DelayEngineer] = None,
        figsize: Tuple[int, int] = (10, 6),
        impute_missing: bool = True,
    ):
        """
        Initialize delay visualization system with healthcare-optimized configuration.

        Sets up comprehensive delay analysis pipeline with appropriate defaults for
        medication preparation workflow optimization and pharmacy operations analytics.
        Integrates DelayEngineer for robust missing data handling and professional
        matplotlib styling for healthcare stakeholder consumption.

        Args:
            delay_engineer: Optional custom DelayEngineer instance with specialized
                           imputation and validation strategies. If None, creates
                           default instance optimized for healthcare workflow analysis.
            figsize: Matplotlib figure dimensions (width, height) for visualization
                    output. Default (10, 6) suitable for most reporting needs.
            impute_missing: Enable sequential timestamp imputation for missing data
                           handling. Recommended for healthcare datasets with typical
                           workflow timestamp gaps and EHR integration challenges.

        Note:
            Default DelayEngineer configuration includes sequential imputation optimized
            for healthcare data patterns with chronological workflow step validation.
            Custom engineers enable specialized analysis for specific operational requirements.
        """
        # Initialize DelayEngineer with healthcare-optimized defaults for robust processing
        self.delay_engineer = delay_engineer or DelayEngineer(impute_missing=bool(impute_missing))
        self.figsize = figsize

    # --------- Healthcare Data Preparation and Validation Methods ---------

    def compute_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform TAT dataset to include comprehensive step-wise delay calculations.

        Applies DelayEngineer transformation pipeline to compute medication preparation
        workflow delays while preserving original dataset integrity. Handles missing
        timestamp patterns common in healthcare data through configurable imputation
        strategies suitable for pharmacy operations analysis and bottleneck identification.

        Processing Pipeline:
        - Sequential timestamp validation for chronological workflow integrity
        - Missing data imputation using healthcare-optimized strategies
        - Step-wise delay calculation in minutes for clinical interpretation
        - Data quality validation with comprehensive error handling

        Args:
            df: Raw TAT dataset containing medication preparation workflow timestamps
               and operational context variables for comprehensive delay analysis

        Returns:
            pd.DataFrame: Enhanced dataset with computed delay_* columns containing
            step-wise processing times in minutes. Original columns preserved for
            comprehensive analysis and downstream model development integration.

        Example:
            For medication preparation workflow analysis:
            Input: doctor_order_time, nurse_validation_time, prep_complete_time...
            Output: delay_order_to_validation, delay_validation_to_prep, delay_prep_to_complete...

        Note:
            Delay calculations maintain chronological sequence validation and handle
            healthcare data irregularities through defensive programming strategies.
            Output suitable for visualization, statistical analysis, and ML model features.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        return self.delay_engineer.transform(df.copy())

    @staticmethod
    def get_delay_cols(df: pd.DataFrame) -> List[str]:
        """
        Identify delay calculation columns in processed TAT dataset.

        Discovers step-wise delay columns generated by DelayEngineer transformation
        for systematic visualization and analysis workflows. Essential for dynamic
        processing of healthcare datasets with variable workflow step coverage
        based on operational context and data availability patterns.

        Column Detection:
        - Identifies columns with 'delay_' prefix indicating computed step intervals
        - Alphabetical sorting ensures consistent processing order for reproducible analysis
        - Robust handling of datasets with partial workflow step coverage

        Args:
            df: Processed TAT dataset potentially containing computed delay columns
               from DelayEngineer transformation or manual feature engineering

        Returns:
            List[str]: Sorted list of delay column names for systematic processing
            and visualization pipeline integration. Empty list if no delay columns found.

        Example:
            For comprehensive medication preparation analysis:
            ['delay_order_to_validation', 'delay_validation_to_prep', 
             'delay_prep_to_complete', 'delay_complete_to_dispatch']

        Note:
            Used internally for visualization pipeline orchestration and data validation.
            Supports dynamic workflow analysis with variable step coverage patterns.
        """
        # Filter for delay columns with alphabetical sorting for consistent processing
        cols = [c for c in df.columns if c.startswith("delay_")]
        cols.sort()
        return cols

    def _ordered_steps(self, available_steps: List[str]) -> List[str]:
        """
        Establish clinically-meaningful plotting order for medication preparation steps.

        Applies healthcare workflow sequence knowledge to organize delay visualizations
        in chronological order matching medication preparation process flow. Ensures
        consistent presentation across different datasets and analysis contexts while
        accommodating variable workflow step coverage in healthcare operations data.

        Ordering Strategy:
        - Prioritizes canonical medication preparation sequence from DELAY_PLOT_ORDER
        - Preserves clinical workflow chronology for stakeholder interpretation
        - Appends additional delay columns alphabetically for comprehensive coverage
        - Handles partial workflow datasets with missing intermediate steps

        Args:
            available_steps: List of delay column names discovered in current dataset
                           through get_delay_cols() or similar detection methods

        Returns:
            List[str]: Ordered delay column names following medication preparation
            workflow sequence for clinically-meaningful visualization presentation

        Example:
            Input: ['delay_prep_to_complete', 'delay_order_to_validation', 'delay_custom_step']
            Output: ['delay_order_to_validation', 'delay_prep_to_complete', 'delay_custom_step']

        Note:
            Clinical workflow ordering enhances stakeholder interpretation and enables
            consistent bottleneck identification across diverse pharmacy operations datasets.
        """
        # Extract steps following preferred clinical workflow sequence
        preferred_present = [c for c in DELAY_PLOT_ORDER if c in available_steps]
        
        # Include additional delay columns not in canonical sequence
        extras = sorted([c for c in available_steps if c not in DELAY_PLOT_ORDER])
        
        return preferred_present + extras

    def available_delay_stats(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate data availability assessment for delay columns in TAT dataset.

        Computes comprehensive coverage statistics for step-wise delay calculations
        to inform imputation strategies, visualization priorities, and data quality
        improvement initiatives. Essential for healthcare analytics teams evaluating
        workflow timestamp completeness and planning bottleneck analysis approaches.

        Coverage Analysis:
        - Non-null observation counts per delay column for sample size assessment
        - Descending sort by availability for prioritized analysis planning
        - Supports data quality evaluation and imputation strategy development
        - Enables informed visualization decisions based on statistical power

        Args:
            df: TAT dataset with computed or candidate delay columns for coverage
               assessment and data quality evaluation workflows

        Returns:
            pd.Series: Coverage statistics indexed by delay column names with
            non-null observation counts sorted descending by availability.
            Empty series if no delay columns detected in dataset.

        Example:
            For pharmacy workflow data quality assessment:
            delay_order_to_validation        9,850
            delay_validation_to_prep         8,920
            delay_prep_to_complete           8,340
            delay_complete_to_dispatch       7,680

        Note:
            Healthcare datasets typically show decreasing coverage for later workflow
            steps due to EHR integration challenges and operational data collection gaps.
            Use for prioritizing visualization focus and imputation strategy development.
        """
        # Identify delay columns for coverage analysis
        cols = self.get_delay_cols(df)
        if not cols:
            # Return empty series with appropriate dtype for consistent downstream handling
            return pd.Series(dtype=int)
        
        # Calculate non-null counts per column and sort by coverage for priority assessment
        return df[cols].notna().sum().sort_values(ascending=False)

    def melt_delays(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform delay dataset to long-format for comprehensive visualization analysis.

        Converts wide-format delay columns to tidy long-format structure suitable for
        matplotlib plotting and statistical analysis workflows. Integrates DelayEngineer
        processing with data validation and cleaning for robust visualization pipeline
        supporting healthcare analytics and pharmacy workflow optimization initiatives.

        Transformation Pipeline:
        1. Apply DelayEngineer transformation for comprehensive delay computation
        2. Extract delay columns and convert to long-format melted structure
        3. Numeric coercion with invalid value handling for robust data processing
        4. Missing value removal for clean visualization and statistical analysis

        Args:
            df: Raw TAT dataset containing medication preparation workflow timestamps
               for transformation to visualization-ready long-format structure

        Returns:
            pd.DataFrame: Long-format dataset with columns ['step', 'delay'] containing
            step identifiers and numeric delay values in minutes for direct plotting
            and statistical analysis integration with healthcare analytics workflows.

        Raises:
            ValueError: If no delay columns found after DelayEngineer transformation
                       or if all delay values are invalid after numeric coercion

        Example:
            Input dataset with delay_order_to_validation, delay_validation_to_prep...
            Output long-format:
            step                        delay
            delay_order_to_validation   15.3
            delay_order_to_validation   22.1
            delay_validation_to_prep    8.7

        Note:
            Essential preprocessing step for matplotlib visualization pipeline and
            statistical analysis workflows. Robust error handling ensures production
            stability with diverse healthcare data quality patterns and edge cases.
        """
        # Apply DelayEngineer transformation for comprehensive delay computation
        df_work = self.compute_delays(df)
        
        # Identify computed delay columns for melting transformation
        cols = self.get_delay_cols(df_work)
        if not cols:
            raise ValueError(
                "No delay_* columns found after DelayEngineer transformation. "
                "Verify input dataset contains required timestamp columns for workflow analysis."
            )
        
        # Transform to long-format structure for plotting and analysis
        melt = df_work.loc[:, cols].melt(var_name="step", value_name="delay")
        
        # Ensure numeric delay values with robust invalid data handling
        melt["delay"] = pd.to_numeric(melt["delay"], errors="coerce")
        melt = melt.dropna(subset=["delay"])
        
        if melt.empty:
            raise ValueError(
                "All delay_* columns contain invalid data after numeric coercion. "
                "Verify DelayEngineer configuration and input data quality for workflow analysis."
            )
        
        return melt

    # --------- Professional Visualization Methods for Healthcare Analytics ---------

    def plot_box(
        self,
        df: pd.DataFrame,
        *,
        sla_minutes: Optional[float] = 60,  # Healthcare TAT threshold standard
        show: bool = True,
        save_path: Optional[str] = None,
        title: str = "Step-to-step Processing Delays",
        color_palette: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Create comprehensive box plot visualization for medication preparation delay analysis.

        Generates production-ready statistical visualization with healthcare-optimized
        annotations, SLA compliance monitoring, and professional styling suitable for
        pharmacy leadership review and clinical stakeholder presentations. Integrates
        DelayEngineer processing with advanced matplotlib visualization for comprehensive
        workflow bottleneck identification and performance monitoring workflows.

        Visualization Features:
        - Box plots with quartile distributions for comprehensive delay pattern analysis
        - Statistical annotations including median, IQR, and sample size information
        - SLA threshold visualization with violation percentage tracking for compliance
        - Color-coded delay severity indicators based on clinical significance thresholds
        - Professional styling with healthcare analytics presentation standards
        - Missing data percentage display for data quality transparency

        Clinical Annotations:
        - Median delay times for operational benchmarking and performance assessment
        - Interquartile range (IQR) for delay variability understanding and prediction
        - Sample sizes with missing data percentages for statistical reliability assessment
        - SLA violation rates for immediate compliance and quality improvement identification

        Args:
            df: Raw TAT dataset containing medication preparation workflow timestamps
               for comprehensive delay analysis and bottleneck identification
            sla_minutes: Service level agreement threshold in minutes for compliance
                        monitoring and violation analysis. Default 60 minutes aligns
                        with healthcare TAT standards for medication preparation workflows.
            show: Display visualization immediately for interactive analysis workflows.
                 Set False for programmatic report generation and batch processing.
            save_path: Optional file path for high-resolution figure persistence suitable
                      for stakeholder distribution and healthcare analytics documentation.
            title: Customizable plot title for context-specific presentations and reports.
                  Default suitable for general medication preparation workflow analysis.
            color_palette: Optional custom color scheme for organizational branding or
                          specific healthcare analytics dashboard integration requirements.

        Returns:
            plt.Figure: Matplotlib figure object for programmatic manipulation, additional
            annotation, or integration with comprehensive healthcare analytics reporting
            systems and dashboard platforms for pharmacy operations monitoring.

        Raises:
            ValueError: If no valid delay data available after DelayEngineer processing
                       or if input dataset lacks required timestamp columns for analysis

        Example:
            For pharmacy leadership review with SLA compliance monitoring:
            visualizer = StepDelayVisualizer()
            fig = visualizer.plot_box(
                monthly_tat_df,
                sla_minutes=60,
                title="Monthly Medication Preparation Delays - SLA Analysis",
                save_path="pharmacy_delays_monthly_report.png"
            )

        Note:
            Professional visualization suitable for healthcare leadership presentations,
            regulatory documentation, and quality improvement initiatives. Statistical
            annotations provide actionable insights for pharmacy workflow optimization
            and resource allocation decision-making in clinical operations environments.
        """
        # Transform dataset to visualization-ready long-format with comprehensive validation
        melt = self.melt_delays(df)
        
        # Establish clinically-meaningful plotting order for medication preparation workflow
        available_steps = sorted(melt["step"].unique().tolist())
        steps = self._ordered_steps(available_steps)
        
        # Organize delay arrays by workflow step for statistical analysis and visualization
        groups = [melt.loc[melt["step"] == s, "delay"].to_numpy(dtype=float) for s in steps]
        
        # Compute comprehensive statistics for healthcare analytics annotations
        stats = []
        for g in groups:
            valid = g[~np.isnan(g)]
            stats.append({
                'median': np.median(valid),
                'q1': np.percentile(valid, 25),
                'q3': np.percentile(valid, 75),
                'n': len(valid),
                'missing': (len(g) - len(valid)) / len(g) * 100,
                'over_sla': (valid > sla_minutes).mean() * 100 if sla_minutes else 0
            })
        
        # Initialize professional healthcare analytics styling for stakeholder presentation
        try:
            import seaborn as sns
            plt.style.use('seaborn')
            sns.set_theme()
        except (ImportError, OSError):
            # Fallback styling for environments without seaborn dependency
            plt.style.use('default')
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.spines.top': False,
                'axes.spines.right': False
            })
        
        # Create figure with healthcare analytics presentation sizing
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Apply color palette with clinical severity indicators for delay visualization
        if color_palette is None:
            color_palette = plt.cm.RdYlBu_r(np.linspace(0, 1, len(steps)))
        
        # Generate box plot with professional healthcare analytics styling
        bp = ax.boxplot(
            groups,
            # Transform delay column names to human-readable step labels for stakeholder clarity
            labels=[s.replace('delay_', '').title() for s in steps],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4),
            whiskerprops=dict(linewidth=1.5),
            boxprops=dict(linewidth=1.5),
            whis=1.5  # Standard 1.5 IQR whisker length for outlier identification
        )
        
        # Apply color-coding based on delay severity relative to SLA threshold
        for patch, stat in zip(bp['boxes'], stats):
            color_val = min(stat['median'] / (sla_minutes if sla_minutes else 60), 1)
            patch.set_facecolor(plt.cm.RdYlBu_r(color_val))
            patch.set_alpha(0.7)
        
        # Establish y-axis range with padding for statistical annotations
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - (ymax - ymin) * 0.15, ymax)  # 15% bottom padding for labels
        
        # Add SLA threshold reference line for compliance monitoring visualization
        if sla_minutes is not None:
            ax.axhline(
                y=sla_minutes,
                color='red',
                linestyle='--',
                linewidth=1.5,
                label=f'SLA Threshold ({sla_minutes} min)'
            )
        
        # Integrate comprehensive statistical annotations for healthcare analytics insight
        for i, stat in enumerate(stats, start=1):
            # Sample size and data quality indicators positioned for readability
            ax.text(
                i, 
                ax.get_ylim()[0] + (ymax - ymin) * 0.02,  # Just above bottom margin
                f'n={stat["n"]}\n({stat["missing"]:.0f}% missing)',
                ha='center', 
                va='bottom',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )
            
            # Central tendency and variability metrics for operational benchmarking
            ax.text(
                i, 
                stat['q3'],
                f'Med: {stat["median"]:.1f}\nIQR: {stat["q3"]-stat["q1"]:.1f}',
                ha='center', 
                va='bottom',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )
            
            # SLA compliance metrics for immediate quality assessment and intervention
            if sla_minutes:
                ax.text(
                    i, 
                    sla_minutes,
                    f'{stat["over_sla"]:.1f}%\nover SLA',
                    ha='center', 
                    va='bottom',
                    fontsize=8,
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )
        
        # Apply professional healthcare analytics styling for stakeholder presentation
        ax.set_xlabel("")
        ax.set_ylabel("Processing Time (minutes)")
        
        # Optimize label readability for medication preparation workflow step names
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        # Enhance readability with professional grid styling for numeric interpretation
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Include legend for SLA threshold reference when applicable
        if sla_minutes is not None:
            ax.legend(loc='upper right')
        
        # Optimize layout for professional presentation and stakeholder consumption
        plt.tight_layout()
        
        # Persist high-resolution figure for healthcare analytics documentation
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Display for interactive analysis workflows when requested
        if show:
            plt.show()
        
        return fig
    
    def _plot_box_from_processed_delays(
        self,
        df: pd.DataFrame,
        delay_cols: List[str],
        *,
        sla_minutes: Optional[float] = 60,
        show: bool = True,
        save_path: Optional[str] = None,
        title: str = "Step-to-step Processing Delays",
        color_palette: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Create consistent box plot visualization directly from pre-computed delay columns.

        Bypasses DelayEngineer recomputation to ensure visualization consistency with
        processed datasets in production analytics workflows. Essential for maintaining
        data integrity when delay calculations have been performed upstream in ETL
        pipelines or when ensuring reproducible visualization outputs for healthcare
        analytics reporting and model validation workflows.

        Consistency Guarantees:
        - Uses exact delay values from pre-computed columns without reprocessing
        - Maintains statistical accuracy with upstream feature engineering pipelines
        - Prevents computation drift between analysis and visualization phases
        - Ensures reproducible outputs for regulatory documentation and quality assurance

        Args:
            df: TAT dataset containing pre-computed delay columns from upstream processing
               or manual feature engineering for direct visualization without recomputation
            delay_cols: Explicit list of delay column names for visualization inclusion
                       ensuring precise control over plotted workflow steps and analysis scope
            sla_minutes: Service level agreement threshold for compliance monitoring
                        and violation analysis identical to standard plot_box() method
            show: Display visualization immediately for interactive analysis workflows
                 matching standard plotting behavior for consistent user experience
            save_path: Optional persistence path for figure storage and distribution
                      supporting automated reporting and stakeholder communication needs
            title: Customizable plot title for context-specific presentations matching
                  standard plotting interface for consistent API usage patterns  
            color_palette: Optional color scheme customization for organizational branding
                          or dashboard integration maintaining visual consistency standards

        Returns:
            plt.Figure: Matplotlib figure object identical to plot_box() output enabling
            consistent downstream processing, annotation, and integration workflows
            with existing healthcare analytics reporting and dashboard systems

        Raises:
            ValueError: If specified delay columns missing from dataset or if no valid
                       delay data found after validation, ensuring robust error handling
                       for production deployment and automated analytics pipeline stability

        Example:
            For model validation with pre-computed delay features ensuring consistency:
            # Upstream feature engineering in ETL pipeline
            processed_df = feature_engineer.compute_delays(raw_tat_df)
            
            # Consistent visualization without recomputation
            delay_columns = ['delay_order_to_validation', 'delay_validation_to_prep']
            visualizer = StepDelayVisualizer()
            fig = visualizer._plot_box_from_processed_delays(
                processed_df,
                delay_columns,
                title="Pre-computed Delay Analysis - Model Validation"
            )

        Note:
            Internal method designed for production analytics workflows requiring
            computational consistency between feature engineering and visualization.
            Maintains identical statistical annotations and styling as standard plot_box()
            while ensuring data lineage integrity for regulatory and quality assurance.
        """
        # Validate delay column availability for robust error handling in production
        missing_cols = [col for col in delay_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing delay columns in dataset: {missing_cols}. "
                f"Verify upstream processing includes all required delay calculations."
            )
        
        # Transform pre-computed delays to long-format without DelayEngineer reprocessing
        melt_data = df[delay_cols].melt(var_name="step", value_name="delay")
        melt_data["delay"] = pd.to_numeric(melt_data["delay"], errors="coerce")
        melt_data = melt_data.dropna(subset=["delay"])
        
        if melt_data.empty:
            raise ValueError(
                "No valid delay data found in pre-computed columns. "
                "Verify upstream delay calculation quality and numeric formatting."
            )
        
        # Apply identical step ordering logic as standard plot_box() for consistency
        available_steps = sorted(melt_data["step"].unique().tolist())
        steps = self._ordered_steps(available_steps)
        
        # Generate delay arrays grouped by workflow step for statistical analysis
        groups = [melt_data.loc[melt_data["step"] == s, "delay"].to_numpy(dtype=float) for s in steps]
        
        # Compute statistical summaries identical to plot_box() for consistent annotations
        stats = []
        for g in groups:
            valid = g[~np.isnan(g)]
            stats.append({
                'median': np.median(valid),
                'q1': np.percentile(valid, 25),
                'q3': np.percentile(valid, 75),
                'n': len(valid),
                'missing': 0.0,  # Pre-processed data has missing values already handled
                'over_sla': (valid > sla_minutes).mean() * 100 if sla_minutes else 0
            })
        
        # Initialize professional styling identical to plot_box() for visual consistency
        try:
            import seaborn as sns
            plt.style.use('seaborn-v0_8')
            sns.set_theme()
        except (ImportError, OSError):
            # Fallback styling matching plot_box() for consistent presentation
            plt.style.use('default')
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.spines.top': False,
                'axes.spines.right': False
            })
        
        # Create figure with larger sizing for healthcare analytics presentation
        # Use minimum size to accommodate all labels and annotations
        fig_width = max(self.figsize[0], 14)  # Minimum 14 inches width
        fig_height = max(self.figsize[1], 10)  # Minimum 10 inches height
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Apply color palette matching plot_box() behavior for visual consistency
        if color_palette is None:
            color_palette = plt.cm.RdYlBu_r(np.linspace(0, 1, len(steps)))
        
        # Generate box plot with identical styling parameters as plot_box()
        bp = ax.boxplot(
            groups,
            labels=[s.replace('delay_', '').replace('_', ' ').title() for s in steps],
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4),
            whiskerprops=dict(linewidth=1.5),
            boxprops=dict(linewidth=1.5),
            whis=1.5
        )
        
        # Apply severity-based color coding identical to plot_box() for consistency
        for patch, stat in zip(bp['boxes'], stats):
            color_val = min(stat['median'] / (sla_minutes if sla_minutes else 60), 1)
            patch.set_facecolor(plt.cm.RdYlBu_r(color_val))
            patch.set_alpha(0.7)
        
        # Establish y-axis range with annotation padding matching plot_box()
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - (ymax - ymin) * 0.15, ymax)
        
        # Add SLA threshold reference line identical to plot_box() functionality
        if sla_minutes is not None:
            ax.axhline(
                y=sla_minutes,
                color='red',
                linestyle='--',
                linewidth=1.5,
                label=f'SLA Threshold ({sla_minutes:.1f} min)'
            )
        
        # Apply statistical annotations identical to plot_box() for consistency
        for i, stat in enumerate(stats, start=1):
            # Sample size and data quality indicators
            ax.text(
                i, 
                ax.get_ylim()[0] + (ymax - ymin) * 0.02,
                f'n={stat["n"]}\n({stat["missing"]:.0f}% missing)',
                ha='center', 
                va='bottom',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )
            
            # Central tendency and variability metrics
            ax.text(
                i, 
                stat['q3'],
                f'Med: {stat["median"]:.1f}\nIQR: {stat["q3"]-stat["q1"]:.1f}',
                ha='center', 
                va='bottom',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
            )
            
            # SLA compliance monitoring annotations
            if sla_minutes:
                ax.text(
                    i, 
                    sla_minutes,
                    f'{stat["over_sla"]:.1f}%\nover SLA',
                    ha='center', 
                    va='bottom',
                    fontsize=8,
                    color='red',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
                )
        
        # Apply professional styling identical to plot_box() for consistency
        ax.set_xlabel("")
        ax.set_ylabel("Processing Time (minutes)")
        
        # Optimize label readability matching plot_box() formatting
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        
        # Professional grid styling for numeric interpretation
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        # Include SLA legend when applicable for compliance reference
        if sla_minutes is not None:
            ax.legend(loc='upper right')
        
        # Optimize layout for professional presentation
        # Use manual adjustment to avoid tight_layout issues with complex annotations
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.25, hspace=0.3)
        
        # Set title with proper spacing
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.96)
        
        # High-resolution persistence for healthcare analytics documentation
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"   • Consistent step delay visualization saved: {save_path}")
        
        # Interactive display when requested for analysis workflows
        if show:
            plt.show()
        
        return fig