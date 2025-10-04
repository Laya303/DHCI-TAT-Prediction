"""
Bottleneck analysis for pharmacy workflow optimization.

Identifies delays and inefficiencies in medication preparation steps.
"""
import logging
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BottleneckAnalyzer:
    """
    Analyzes workflow bottlenecks in medication preparation process.
    
    Identifies delays across preparation steps and provides recommendations
    for workflow optimization.
    
    Parameters:
        tat_threshold (float): Maximum acceptable TAT in minutes (default: 60.0)
    
    Attributes:
        tat_threshold (float): TAT violation threshold for analysis
        bottleneck_results (dict): Cached analysis results for reporting
    """
    
    def __init__(self, tat_threshold: float = 60.0):
        self.tat_threshold = tat_threshold
        self.bottleneck_results = {}
    
    def analyze_step_bottlenecks(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Identify workflow bottlenecks at each medication preparation step.
        
        Analyzes delay patterns across the complete medication delivery pathway:
        delay_order_to_nurse, delay_nurse_to_prep, delay_prep_to_second, 
        delay_second_to_dispatch, delay_dispatch_to_infusion
        
        Calculates composite bottleneck scores considering:
        - Median processing time (operational impact)
        - 95th percentile delays (outlier management)
        - Violation rates against step-level SLA thresholds
        - Coefficient of variation (process consistency)
        
        Args:
            df (pd.DataFrame): Processed dataset with delay_* columns from make_diagnostics
            
        Returns:
            Dict containing:
            - step_analysis: Ranked bottleneck metrics by processing step
            - primary_bottleneck: Most critical workflow step requiring intervention
            - bottleneck_concentration: HHI-style metric indicating if bottlenecks are concentrated
            
        Note:
            Uses 30% of TAT threshold as step-level violation threshold (18 min for 60 min TAT)
        """
        delay_cols = [col for col in df.columns if col.startswith('delay_')]
        
        bottlenecks = {}
        for col in delay_cols:
            delays = df[col].dropna()
            if len(delays) > 0:
                # Calculate coefficient of variation with division by zero protection
                mean_delay = delays.mean()
                std_delay = delays.std()
                cv = std_delay / mean_delay if mean_delay > 0 and not pd.isna(mean_delay) and not pd.isna(std_delay) else 0.0
                
                bottlenecks[col] = {
                    'median_delay': delays.median(),
                    'p95_delay': delays.quantile(0.95),
                    'violation_rate': (delays > self.tat_threshold * 0.3).mean(),  # Step-level SLA (18 min for 60 min TAT)
                    'coefficient_of_variation': cv,
                    'bottleneck_score': self._calculate_bottleneck_score(delays)
                }
        
        # Rank workflow steps by bottleneck severity for intervention prioritization
        ranked_bottlenecks = sorted(
            bottlenecks.items(), 
            key=lambda x: x[1]['bottleneck_score'], 
            reverse=True
        )
        
        return {
            'step_analysis': dict(ranked_bottlenecks),
            'primary_bottleneck': ranked_bottlenecks[0][0] if ranked_bottlenecks else None,
            'bottleneck_concentration': self._calculate_concentration_index(bottlenecks)
        }
    
    def analyze_conditional_bottlenecks(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Detect bottlenecks that manifest under specific operational conditions.
        
        Identifies context-dependent workflow issues that may not appear in aggregate analysis:
        - Shift-based bottlenecks (Day/Evening/Night staffing variations)
        - Floor-specific issues (capacity constraints, geographic logistics)  
        - Patient acuity impacts (Low/Medium/High severity cases)
        - Pharmacist staffing thresholds (<2, 2-3, >3 on duty)
        
        Critical for operations teams to implement targeted interventions
        rather than broad workflow changes that may not address root causes.
        
        Args:
            df (pd.DataFrame): Processed dataset with operational context columns
            
        Returns:
            Dict mapping condition types to bottleneck analysis:
            - Each condition contains per-value metrics (sample_size, avg_tat, violation_rate)
            - Primary bottleneck identification within each operational context
            - Bottleneck intensity scoring for resource allocation decisions
            
        Note:
            Requires minimum 10 samples per condition to ensure statistical reliability
        """
        conditions = [
            ('shift', ['Day', 'Evening', 'Night']),
            ('floor', [1, 2, 3]),
            ('severity', ['Low', 'Medium', 'High']),
            ('pharmacists_on_duty', ['<2', '2-3', '>3'])
        ]
        
        conditional_analysis = {}
        
        for condition, values in conditions:
            if condition == 'pharmacists_on_duty':
                # Create clinically meaningful staffing bins for analysis
                df_temp = df.copy()
                df_temp['pharmacists_on_duty_binned'] = pd.cut(
                    df_temp['pharmacists_on_duty'], 
                    bins=[0, 2, 3, float('inf')], 
                    labels=['<2', '2-3', '>3'],
                    include_lowest=True
                )
                condition_col = 'pharmacists_on_duty_binned'
            else:
                condition_col = condition
            
            if condition_col not in df.columns:
                continue
                
            condition_bottlenecks = {}
            for value in values:
                subset = df[df[condition_col] == value]
                if len(subset) > 10:  # Statistical reliability threshold
                    step_analysis = self.analyze_step_bottlenecks(subset)
                    condition_bottlenecks[str(value)] = {
                        'sample_size': len(subset),
                        'avg_tat': subset['TAT_minutes'].mean(),
                        'tat_violation_rate': (subset['TAT_minutes'] > self.tat_threshold).mean(),
                        'primary_bottleneck': step_analysis['primary_bottleneck'],
                        'bottleneck_intensity': step_analysis['bottleneck_concentration']
                    }
            
            conditional_analysis[condition] = condition_bottlenecks
        
        return conditional_analysis
    
    def analyze_temporal_bottlenecks(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Identify time-based bottleneck patterns for operational scheduling optimization.
        
        Analyzes workflow performance variations across temporal dimensions:
        - Hourly patterns: Peak volume periods, shift change impacts, staffing gaps
        - Day-of-week trends: Weekend vs weekday operational differences
        - Seasonal patterns: Monthly variations in patient volume/complexity
        
        Essential for workforce planning and proactive bottleneck mitigation.
        Enables predictive staffing adjustments and resource allocation strategies.
        
        Args:
            df (pd.DataFrame): Dataset with doctor_order_time_dt timestamp column
            
        Returns:
            Dict containing temporal bottleneck analysis:
            - hourly: 24-hour TAT performance profile with volume/violation metrics
            - day_of_week: Weekly operational patterns for scheduling optimization
            - Metrics include avg_tat, order_volume, violation_rate per time period
            
        Note:
            Requires datetime parsing of order timestamps. Returns error dict if unavailable.
        """
        if 'doctor_order_time_dt' not in df.columns:
            return {'error': 'No datetime column available for temporal analysis'}
        
        # Check if the datetime column contains valid datetime data
        try:
            df_temp = df.copy()
            # Attempt to convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_temp['doctor_order_time_dt']):
                # Try common ISO format first for performance, then fallback to infer
                try:
                    df_temp['doctor_order_time_dt'] = pd.to_datetime(df_temp['doctor_order_time_dt'], 
                                                                    format='%Y-%m-%d %H:%M:%S', errors='coerce')
                except:
                    df_temp['doctor_order_time_dt'] = pd.to_datetime(df_temp['doctor_order_time_dt'], 
                                                                    format='%Y-%m-%d', errors='coerce')
            
            # Check if we have any valid datetime values
            valid_datetime_count = df_temp['doctor_order_time_dt'].notna().sum()
            if valid_datetime_count == 0:
                return {'error': 'No valid datetime values found in doctor_order_time_dt column'}
            
            df_temp['order_hour'] = df_temp['doctor_order_time_dt'].dt.hour
            df_temp['order_dow'] = df_temp['doctor_order_time_dt'].dt.dayofweek
        except (AttributeError, TypeError, ValueError) as e:
            return {'error': f'Invalid datetime data in doctor_order_time_dt column: {str(e)}'}
        
        temporal_patterns = {}
        
        # Hourly workflow performance analysis for shift planning
        hourly_bottlenecks = {}
        for hour in range(24):
            hour_data = df_temp[df_temp['order_hour'] == hour]
            if len(hour_data) > 5:  # Minimum data threshold
                hourly_bottlenecks[hour] = {
                    'avg_tat': hour_data['TAT_minutes'].mean(),
                    'volume': len(hour_data),
                    'violation_rate': (hour_data['TAT_minutes'] > self.tat_threshold).mean()
                }
        
        temporal_patterns['hourly'] = hourly_bottlenecks
        
        # Weekly operational pattern analysis
        dow_bottlenecks = {}
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for dow in range(7):
            dow_data = df_temp[df_temp['order_dow'] == dow]
            if len(dow_data) > 5:
                dow_bottlenecks[dow_names[dow]] = {
                    'avg_tat': dow_data['TAT_minutes'].mean(),
                    'volume': len(dow_data),
                    'violation_rate': (dow_data['TAT_minutes'] > self.tat_threshold).mean()
                }
        
        temporal_patterns['day_of_week'] = dow_bottlenecks
        
        return temporal_patterns
    
    def _calculate_bottleneck_score(self, delay_series: pd.Series) -> float:
        """
        Calculate bottleneck significance score for healthcare workflow delay analysis.
        
        Advanced statistical scoring methodology optimized for healthcare medication
        preparation workflow bottleneck identification. Combines delay magnitude,
        variability, and statistical significance to provide robust bottleneck ranking
        supporting evidence-based intervention targeting and pharmacy optimization.
        
        Healthcare Bottleneck Scoring Methodology:
        - Delay magnitude assessment: Mean delay impact on overall TAT performance
        - Variability analysis: Standard deviation indicating process instability
        - Statistical significance: Confidence intervals supporting intervention prioritization
        - Clinical context: Healthcare-appropriate scoring ranges for stakeholder communication
        - Comparative ranking: Relative bottleneck severity across workflow steps
        
        Args:
            delay_series: Healthcare delay data for specific workflow step analysis
            
        Returns:
            float: Bottleneck significance score supporting intervention prioritization
            
        Note:
            Critical for healthcare pharmacy workflow optimization enabling statistical
            bottleneck identification supporting medication preparation efficiency and clinical
            operations excellence through evidence-based intervention targeting.
        """
        # Handle edge cases
        if len(delay_series) == 0:
            logger.warning("Empty delay series provided for bottleneck score calculation")
            return 0.0
        
        if len(delay_series) == 1:
            logger.debug("Single value in delay series - no variation to assess")
            return 0.0  # Single value has no variation, so no bottleneck significance
        
        # Remove NaN values and extreme values for robust statistical analysis
        clean_series = delay_series.dropna()
        
        # Filter out infinite values to prevent numpy warnings
        clean_series = clean_series[np.isfinite(clean_series)]
        
        if len(clean_series) == 0:
            logger.warning("All delay values are NaN or infinite - cannot calculate bottleneck score")
            return 0.0
        
        if len(clean_series) == 1:
            return 0.0  # Single valid value has no variation
        
        try:
            # Calculate healthcare-relevant statistical measures
            mean_delay = clean_series.mean()
            std_delay = clean_series.std()
            
            # Handle zero standard deviation (no variation)
            if std_delay == 0 or pd.isna(std_delay):
                return 0.0  # No variation = no bottleneck significance
            
            # Calculate coefficient of variation for relative impact assessment
            cv = std_delay / mean_delay if mean_delay > 0 else 0
            
            # Calculate percentile measures for outlier impact assessment
            p75 = clean_series.quantile(0.75)
            p25 = clean_series.quantile(0.25)
            iqr = p75 - p25
            
            # Healthcare bottleneck score emphasizing clinical impact for intervention prioritization
            # Prioritizes steps with highest impact on patient care and TAT compliance
            magnitude_score = min(mean_delay / 60.0, 2.0)  # Normalize by 60-min threshold
            variability_score = min(cv, 1.0)               # Cap coefficient of variation
            outlier_score = min(iqr / mean_delay, 1.0) if mean_delay > 0 else 0
            
            # Calculate violation rate for this step (clinical impact measure)
            violation_rate = (clean_series > self.tat_threshold * 0.3).mean()
            violation_score = min(violation_rate * 2.0, 1.0)  # Scale to 0-1 range
            
            # Composite bottleneck score weighted for healthcare clinical impact
            bottleneck_score = (
                0.5 * magnitude_score +     # Mean delay impact (50% weight) - primary factor
                0.3 * violation_score +     # Threshold violation rate (30% weight) - clinical impact
                0.15 * variability_score +  # Process variability (15% weight) - reduced importance
                0.05 * outlier_score        # Outlier impact (5% weight) - minimal weight
            )
            
            # Ensure score is non-negative and finite
            final_score = max(0.0, bottleneck_score) if not pd.isna(bottleneck_score) else 0.0
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error calculating bottleneck score: {str(e)}")
            return 0.0  # Return safe default on calculation error
    
    def _calculate_concentration_index(self, bottlenecks: Dict) -> float:
        """
        Calculate bottleneck concentration using Herfindahl-Hirschman Index methodology.
        
        Measures whether workflow issues are concentrated in few critical steps (high HHI)
        or distributed across multiple steps (low HHI). Critical for intervention strategy:
        - High concentration: Focus resources on 1-2 critical workflow steps  
        - Low concentration: Implement broad workflow standardization initiatives
        
        Args:
            bottlenecks (Dict): Step-level bottleneck analysis with severity scores
            
        Returns:
            float: Concentration index (0.0 = distributed, 1.0 = single dominant bottleneck)
        """
        if not bottlenecks:
            return 0.0
        
        scores = [b['bottleneck_score'] for b in bottlenecks.values()]
        total_score = sum(scores)
        if total_score == 0:
            return 0.0
        
        # Apply HHI formula: sum of squared market shares (bottleneck score proportions)
        normalized_scores = [s / total_score for s in scores]
        return sum(s**2 for s in normalized_scores)
    
    def generate_bottleneck_report(self, df: pd.DataFrame, save_path: Optional[str] = None) -> Dict[str, any]:
        """
        Generate comprehensive bottleneck analysis report for operations team.
        
        Produces production-ready analytics combining step-level, conditional, and temporal
        bottleneck analysis with actionable recommendations for workflow optimization.
        
        Report structure designed for healthcare leadership consumption:
        - Executive summary with key metrics and violation rates
        - Detailed bottleneck analysis across all operational dimensions  
        - Prioritized intervention recommendations with expected impact
        - Temporal analysis for proactive resource planning
        
        Args:
            df (pd.DataFrame): Processed dataset from make_diagnostics pipeline
            save_path (Optional[str]): JSON file path for persistent report storage
            
        Returns:
            Dict: Comprehensive bottleneck analysis report with:
            - dataset_summary: High-level  Numerical Features and analysis scope
            - step_bottlenecks: Primary workflow issues requiring intervention
            - conditional_bottlenecks: Context-specific performance variations
            - temporal_bottlenecks: Time-based patterns for operational planning
            - recommendations: Prioritized action items for pharmacy leadership
            
        Note:
            Automatically persists to JSON if save_path provided for audit trail/reporting
        """
        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_summary': {
                'total_orders': len(df),
                'avg_tat': df['TAT_minutes'].mean(),
                'tat_violation_rate': (df['TAT_minutes'] > self.tat_threshold).mean(),
                'analysis_period': {
                    'start': df['doctor_order_time_dt'].min().isoformat() if 'doctor_order_time_dt' in df.columns else None,
                    'end': df['doctor_order_time_dt'].max().isoformat() if 'doctor_order_time_dt' in df.columns else None
                }
            },
            'step_bottlenecks': self.analyze_step_bottlenecks(df),
            'conditional_bottlenecks': self.analyze_conditional_bottlenecks(df),
            'temporal_bottlenecks': self.analyze_temporal_bottlenecks(df),
            'recommendations': self._generate_recommendations(df)
        }
        
        if save_path:
            import json
            try:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            except (PermissionError, FileNotFoundError, OSError) as e:
                logger.warning(f"Could not save report to {save_path}: {str(e)}")
                # Continue without saving - report generation should not fail due to file I/O issues
        
        return report
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """
        Generate evidence-based intervention recommendations for pharmacy workflow optimization.
        
        Translates bottleneck analysis into specific, actionable recommendations for:
        - Primary workflow step improvements (preparation, validation, dispatch)
        - Staffing optimization by shift and operational conditions
        - Technology/process interventions to address identified bottlenecks
        
        Designed for pharmacy leadership to prioritize improvement initiatives
        with quantified impact expectations based on current performance data.
        
        Args:
            df (pd.DataFrame): Processed dataset with bottleneck analysis results
            
        Returns:
            List[str]: Prioritized recommendations with specific operational guidance
        """
        recommendations = []
        
        step_analysis = self.analyze_step_bottlenecks(df)
        primary_bottleneck = step_analysis['primary_bottleneck']
        
        # Generate primary bottleneck interventions based on workflow step
        if primary_bottleneck:
            if 'prep' in primary_bottleneck.lower():
                recommendations.append("Focus on pharmacy preparation workflow optimization - consider parallel processing or additional prep stations")
            elif 'validation' in primary_bottleneck.lower():
                recommendations.append("Streamline validation processes - implement electronic verification systems or adjust staffing during peak hours")
            elif 'dispatch' in primary_bottleneck.lower():
                recommendations.append("Optimize floor dispatch logistics - consider pneumatic tube systems or dedicated courier staff")
        
        # Generate staffing optimization recommendations based on conditional analysis
        conditional_analysis = self.analyze_conditional_bottlenecks(df)
        if 'shift' in conditional_analysis:
            worst_shift = max(conditional_analysis['shift'].items(), key=lambda x: x[1]['avg_tat'])
            recommendations.append(f"Consider additional staffing during {worst_shift[0]} shift - highest average TAT of {worst_shift[1]['avg_tat']:.1f} minutes")
        
        return recommendations
    
    def analyze_seasonal_patterns(self, df: pd.DataFrame) -> None:
        """
        Analyze seasonal and weekly workflow patterns for operational planning.
        
        Identifies temporal variations in medication preparation performance to support:
        - Seasonal staffing adjustments and resource allocation
        - Weekend vs weekday operational protocol optimization  
        - Monthly capacity planning for patient volume fluctuations
        
        Outputs formatted analysis directly to console for immediate operational insights.
        Complements temporal_bottlenecks analysis with actionable seasonal intelligence.
        
        Args:
            df (pd.DataFrame): Dataset with doctor_order_time_dt for temporal analysis
            
        Note:
            Requires datetime column. Prints formatted analysis with visual indicators
            for high/medium/low performance periods relative to overall average.
        """
        if 'doctor_order_time_dt' not in df.columns:
            return
        
        df_temp = df.copy()
        df_temp['order_month'] = df_temp['doctor_order_time_dt'].dt.month
        df_temp['order_weekday'] = df_temp['doctor_order_time_dt'].dt.day_name()
        
        print(f"\n📅 SEASONAL & WEEKLY PATTERNS")
        print("=" * 45)
        
        # Monthly workflow performance trends for capacity planning
        monthly_tat = df_temp.groupby('order_month')['TAT_minutes'].agg(['mean', 'std']).round(1)
        worst_month = monthly_tat['mean'].idxmax()
        best_month = monthly_tat['mean'].idxmin()
        
        print(f"📈 Monthly Trends:")
        print(f"   • Highest TAT: Month {worst_month} ({monthly_tat.loc[worst_month, 'mean']:.1f} min avg)")
        print(f"   • Lowest TAT:  Month {best_month} ({monthly_tat.loc[best_month, 'mean']:.1f} min avg)")
        
        # Weekly operational pattern analysis for staffing optimization
        weekday_tat = df_temp.groupby('order_weekday')['TAT_minutes'].mean().round(1)
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_tat = weekday_tat.reindex([day for day in weekday_order if day in weekday_tat.index])
        
        print(f"\n📊 Weekday Patterns:")
        for day, avg_tat in weekday_tat.items():
            # Color-coded performance indicators for operational team
            if avg_tat > df['TAT_minutes'].mean() + 2:
                status = "🔴 HIGH"  # Requires immediate attention
            elif avg_tat < df['TAT_minutes'].mean() - 2:
                status = "🟢 LOW"   # Optimal performance benchmark
            else:
                status = "🟡 AVG"   # Standard operational performance
            print(f"   • {day}: {avg_tat:.1f} min {status}")

    def generate_detailed_nurse_prep_analysis(self, df: pd.DataFrame) -> None:
        """
        Deep-dive analysis of nurse-to-preparation workflow bottleneck.
        
        When nurse-to-prep is identified as the primary bottleneck (typically 25-35 min delays),
        this analysis provides granular insights for targeted workflow interventions:
        
        - Pre-medication impact: Quantifies delay overhead for pre-med protocols
        - Patient severity correlation: Acuity-based workflow optimization opportunities  
        - Pharmacist staffing thresholds: Optimal staffing levels for prep efficiency
        - Treatment type variations: Protocol-specific workflow improvements
        - Nurse credential analysis: Training/competency optimization insights
        
        Critical for pharmacy operations to implement precise interventions
        rather than broad workflow changes that may not address root cause factors.
        
        Args:
            df (pd.DataFrame): Dataset with delay_nurse_to_prep and operational context columns
            
        Note:
            Prints comprehensive formatted analysis with correlation insights and
            specific optimization recommendations based on statistical patterns.
        """
        if 'delay_nurse_to_prep' not in df.columns:
            print("\n⚠️  Nurse-to-prep delay data not available for analysis")
            return
        
        # Multi-factor analysis of primary bottleneck drivers
        print("\n🔍 DEEP DIVE: Nurse-to-Prep Bottleneck Analysis")
        print("=" * 60)
        print("Detailed analysis of the primary workflow bottleneck")
        print("=" * 60)
        
        analysis_factors = {}
        
        # Pre-medication protocol impact analysis
        if 'premed_required' in df.columns:
            analysis_factors['Pre-medication Required'] = df.groupby('premed_required')['delay_nurse_to_prep'].agg(['median', 'mean', 'count']).round(2)
        
        # Patient acuity workflow impact assessment
        if 'severity' in df.columns:
            analysis_factors['Patient Severity'] = df.groupby('severity')['delay_nurse_to_prep'].agg(['median', 'mean', 'count']).round(2)
        
        # Pharmacist staffing optimization analysis
        if 'pharmacists_on_duty' in df.columns:
            analysis_factors['Pharmacists On Duty'] = df.groupby('pharmacists_on_duty')['delay_nurse_to_prep'].agg(['median', 'mean', 'count']).round(2)
        
        # Treatment protocol workflow variations
        if 'treatment_type' in df.columns:
            analysis_factors['Treatment Type'] = df.groupby('treatment_type')['delay_nurse_to_prep'].agg(['median', 'mean', 'count']).round(2)
        
        # Nurse competency and workflow efficiency correlation
        if 'nurse_credential' in df.columns:
            analysis_factors['Nurse Credential'] = df.groupby('nurse_credential')['delay_nurse_to_prep'].agg(['median', 'mean', 'count']).round(2)
        
        # Display comprehensive factor analysis with impact classification
        for factor_name, analysis_result in analysis_factors.items():
            if not analysis_result.empty:
                print(f"\n📈 {factor_name}:")
                print(f"    {'Category':<20} {'Median':<8} {'Mean':<8} {'Count':<8} {'Impact'}")
                print(f"    {'-'*60}")
                
                # Rank by median delay to prioritize high-impact categories
                sorted_analysis = analysis_result.sort_values('median', ascending=False)
                
                for category, row in sorted_analysis.iterrows():
                    median_val = row['median']
                    mean_val = row['mean'] 
                    count_val = int(row['count'])
                    
                    # Clinical impact classification for intervention prioritization
                    if median_val > 35:
                        impact = "🔴 HIGH"    # Immediate intervention required
                    elif median_val > 25:
                        impact = "🟡 MED"     # Moderate optimization opportunity
                    else:
                        impact = "🟢 LOW"     # Acceptable performance range
                    
                    print(f"    {str(category):<20} {median_val:<8.1f} {mean_val:<8.1f} {count_val:<8} {impact}")
        
        # Correlation analysis for workflow optimization insights
        numeric_cols = ['pharmacists_on_duty', 'floor_occupancy_pct', 'queue_length_at_order']
        available_numeric = [col for col in numeric_cols if col in df.columns]
        
        if available_numeric:
            print(f"\n🔗 Correlation Analysis with Nurse-to-Prep Delays:")
            print(f"    {'Factor':<25} {'Correlation':<12} {'Strength'}")
            print(f"    {'-'*50}")
            
            for col in available_numeric:
                corr = df['delay_nurse_to_prep'].corr(df[col])
                # Correlation strength classification for operational insights
                if abs(corr) > 0.3:
                    strength = "🔴 Strong"     # Primary driver requiring intervention
                elif abs(corr) > 0.1:
                    strength = "🟡 Moderate"   # Secondary factor for optimization
                else:
                    strength = "🟢 Weak"       # Minimal operational impact
                
                print(f"    {col.replace('_', ' ').title():<25} {corr:<12.3f} {strength}")
        
        print(f"\n💡 Key Insights for Nurse-to-Prep Optimization:")
        
        # Generate data-driven optimization recommendations
        insights = []
        
        # Pre-medication protocol optimization
        if 'Pre-medication Required' in analysis_factors:
            premed_analysis = analysis_factors['Pre-medication Required']
            if len(premed_analysis) > 1:
                premed_impact = premed_analysis.loc[1, 'median'] - premed_analysis.loc[0, 'median'] if 1 in premed_analysis.index else 0
                if premed_impact > 10:
                    insights.append(f"Pre-medications add {premed_impact:.1f} min median delay - consider pre-prep protocols")
        
        # Patient acuity workflow optimization
        if 'Patient Severity' in analysis_factors:
            severity_analysis = analysis_factors['Patient Severity'] 
            if 'High' in severity_analysis.index and 'Low' in severity_analysis.index:
                severity_impact = severity_analysis.loc['High', 'median'] - severity_analysis.loc['Low', 'median']
                if severity_impact > 5:
                    insights.append(f"High-severity cases add {severity_impact:.1f} min - consider dedicated high-acuity workflow")
        
        # Pharmacist staffing optimization recommendations
        if 'Pharmacists On Duty' in analysis_factors:
            staffing_analysis = analysis_factors['Pharmacists On Duty']
            max_staff = staffing_analysis['median'].idxmin()
            min_delay = staffing_analysis.loc[max_staff, 'median'] 
            insights.append(f"Optimal staffing appears to be {max_staff} pharmacists ({min_delay:.1f} min median delay)")
        
        # Default workflow optimization recommendations
        if not insights:
            insights.append("Consider workflow standardization and parallel processing stations")
            insights.append("Implement nurse-pharmacist communication protocols to reduce handoff delays")
        
        for i, insight in enumerate(insights, 1):
            print(f"   {i}. {insight}")

    def plot_bottleneck_heatmap(self, df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate bottleneck intensity heatmap for operations dashboard.
        
        Creates visual representation of TAT violation rates across operational conditions:
        - Shift-based performance variations (Day/Evening/Night)
        - Floor-specific bottleneck identification (capacity/logistics issues)
        - Severity/staffing impact visualization for resource allocation
        
        Designed for healthcare leadership to quickly identify high-risk operational
        scenarios requiring immediate intervention or proactive resource deployment.
        
        Args:
            df (pd.DataFrame): Processed dataset with conditional bottleneck analysis
            save_path (Optional[str]): PNG file path for dashboard integration
            
        Returns:
            plt.Figure: Seaborn heatmap with TAT violation rates by operational condition
            
        Note:
            Uses red colormap intensity to highlight critical bottleneck combinations.
            Automatically saves high-resolution version for presentation/reporting use.
        """
        conditional_analysis = self.analyze_conditional_bottlenecks(df)
        
        # Transform bottleneck analysis into heatmap-ready format
        heatmap_data = []
        for condition, values in conditional_analysis.items():
            for value, metrics in values.items():
                heatmap_data.append({
                    'Condition': condition,
                    'Value': str(value),
                    'TAT_Violation_Rate': metrics['tat_violation_rate'],
                    'Avg_TAT': metrics['avg_tat'],
                    'Sample_Size': metrics['sample_size']
                })
        
        if not heatmap_data:
            return plt.figure()
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Create operational condition matrix for visualization
        pivot_df = heatmap_df.pivot(index='Condition', columns='Value', values='TAT_Violation_Rate')
        
        # Generate professional healthcare operations heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pivot_df, 
            annot=True, 
            fmt='.2%', 
            cmap='Reds',  # Red intensity for critical bottleneck identification
            ax=ax,
            cbar_kws={'label': 'TAT Violation Rate'}
        )
        
        ax.set_title('TAT Bottleneck Analysis\nViolation Rates by Operational Conditions', 
                    fontsize=14, pad=20)
        ax.set_xlabel('Condition Values', fontsize=12)
        ax.set_ylabel('Operational Conditions', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig