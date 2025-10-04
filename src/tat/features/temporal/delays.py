"""
Step-wise Delay Computation for Pharmacy TAT Workflow Analysis

Production-ready delay engineering system for medication preparation turnaround time
analysis and bottleneck identification. Provides comprehensive step-wise delay
calculation with healthcare-optimized sequential imputation strategies designed
specifically for pharmacy operations and clinical workflow optimization initiatives.

Key Components:
- DelayEngineer: Primary delay computation interface with missing data handling
- Sequential imputation respecting medication preparation workflow chronology
- Healthcare data quality validation and negative delay handling strategies
- Integration with TimeReconstructor and StepDelayVisualizer for comprehensive analysis

Clinical Applications:
- Medication preparation bottleneck identification through step-wise delay analysis
- TAT prediction model feature engineering with robust missing data handling
- Healthcare workflow optimization through delay pattern recognition and analysis
- Pharmacy operations quality monitoring with comprehensive delay tracking

Technical Features:
- Smart imputation respecting chronological order and clinical workflow constraints
- Negative delay detection and handling for healthcare data quality assurance
- Configurable delay pair computation supporting diverse workflow configurations
- Helper column management for clean feature engineering pipeline integration
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Dict
import logging

import numpy as np
import pandas as pd

from tat.config import STEP_COLS, DELAY_PAIRS, HELPER_SUFFIXES

logger = logging.getLogger(__name__)

class DelayEngineer:
    """
    Primary delay computation system for pharmacy TAT workflow analysis and optimization.

    Orchestrates comprehensive step-wise delay calculation from medication preparation
    workflow timestamps with healthcare-optimized imputation strategies. Designed for
    production deployment in pharmacy operations analytics environments with robust
    error handling and clinical workflow validation for bottleneck identification.

    Core Responsibilities:
    - Compute step-to-step processing delays in minutes from workflow timestamp pairs
    - Apply sequential imputation for missing timestamps using clinical workflow patterns
    - Validate chronological sequence integrity for healthcare data quality assurance
    - Provide helper column management for clean feature engineering pipeline integration

    Args:
        pairs: Sequence of (prev_col, next_col, out_col) tuples defining delay calculations.
              If None, uses default medication preparation workflow pairs from configuration.
        drop_suffixes: Column suffixes considered transient for helper column cleanup.
                      Default includes '_mins_unwrapped', '_dt' for clean feature engineering.
        step_cols: Explicit step fragment column names for comprehensive cleanup operations.
                  Defaults to configured workflow step columns for medication preparation.
        impute_missing: Enable sequential imputation for missing timestamp data handling.
                       Recommended for healthcare datasets with typical 5-15% missingness.

    Example:
        # Standard medication preparation delay computation with imputation
        delay_engineer = DelayEngineer(impute_missing=True)
        enhanced_df = delay_engineer.transform(tat_df)
        
        # Access computed delay features for analysis
        delay_columns = [col for col in enhanced_df.columns if col.startswith('delay_')]
        
        # Clean helper columns for downstream modeling
        model_ready_df = delay_engineer.drop_processed_helpers(enhanced_df)

    Note:
        Designed for production deployment in healthcare analytics environments with
        comprehensive error handling and clinical workflow validation. Integrates with
        pharmacy operations monitoring systems and TAT prediction modeling pipelines.
    """

    def __init__(
        self,
        pairs: Optional[Sequence[Tuple[Optional[str], str, str]]] = None,
        drop_suffixes: Optional[Iterable[str]] = ("_mins_unwrapped", "_dt"),
        step_cols: Optional[Iterable[str]] = None,
        impute_missing: bool = True,
    ):
        """
        Initialize delay computation system with healthcare-optimized configuration.

        Sets up comprehensive delay engineering pipeline with appropriate defaults for
        medication preparation workflow analysis and pharmacy operations optimization.
        Configures imputation strategies, helper column management, and clinical
        workflow validation for robust healthcare analytics processing.

        Args:
            pairs: Optional custom delay computation pairs for specialized workflow analysis.
                  Each tuple contains (previous_timestamp_col, next_timestamp_col, output_delay_col).
                  If None, uses default medication preparation workflow sequence pairs.
            drop_suffixes: Column suffixes for transient helper column identification.
                         Default ('_mins_unwrapped', '_dt') supports standard timestamp processing.
            step_cols: Explicit workflow step columns for comprehensive cleanup operations.
                      Defaults to configured medication preparation workflow step columns.
            impute_missing: Enable sequential imputation for missing timestamp handling.
                          Recommended True for healthcare datasets with EHR integration gaps.

        Note:
            Default configuration optimized for standard medication preparation workflows
            with typical healthcare data quality patterns and missing timestamp handling.
            Custom configurations support specialized operational requirements and analysis.
        """
        # Configure delay computation pairs for medication preparation workflow analysis
        self.pairs = list(pairs) if pairs is not None else list(DELAY_PAIRS)
        
        # Set up helper column management for clean feature engineering pipeline
        self.drop_suffixes = tuple(drop_suffixes) if drop_suffixes else HELPER_SUFFIXES
        self.step_cols = list(step_cols) if step_cols else list(STEP_COLS)
        
        # Enable healthcare-optimized imputation for missing timestamp handling
        self.impute_missing = bool(impute_missing)

    @staticmethod
    def _to_num(s: pd.Series) -> pd.Series:
        """
        Convert series to numeric format with robust error handling for healthcare data.
        
        Applies pandas numeric coercion with error tolerance for healthcare datasets
        containing mixed data types, missing values, and EHR integration artifacts.
        Essential for robust processing of timestamp and delay calculations in
        production pharmacy analytics environments with diverse data quality patterns.
        
        Args:
            s: Input series potentially containing mixed types or invalid numeric values
               from healthcare data sources and EHR integration processes
        
        Returns:
            pd.Series: Numeric series with non-numeric values coerced to NaN for
            consistent downstream processing and delay calculation workflows
        """
        return pd.to_numeric(s, errors="coerce")
    
    def _calculate_step_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive delay statistics for medication preparation workflow steps.
        
        Analyzes historical delay patterns to derive empirically-based statistics for
        imputation and validation strategies. Essential for healthcare-optimized missing
        data handling that respects operational context and clinical workflow patterns
        specific to pharmacy operations and medication preparation environments.
        
        Statistical Computation:
        - Percentile-based delay distributions for robust outlier handling
        - Median processing times for typical delay imputation strategies
        - Quartile ranges for delay variability assessment and validation bounds
        - Step-specific statistics supporting operational context-aware imputation
        
        Healthcare Context Integration:
        - Validates positive delay values for clinical workflow sequence integrity
        - Filters invalid or negative delays from EHR integration data quality issues
        - Computes statistics per workflow step for accurate imputation targeting
        - Supports operational benchmarking and quality monitoring initiatives
        
        Args:
            df: TAT dataset containing processed timestamp columns for statistical analysis
               and delay pattern learning from historical medication preparation data
        
        Returns:
            Dict[str, Dict[str, float]]: Nested dictionary mapping delay column names to
            comprehensive statistics including min (5th percentile), q25, median, q75,
            and typical_delay values for healthcare-optimized imputation strategies
            
        Example:
            For medication preparation delay statistics:
            {
                'delay_order_to_validation': {
                    'min': 2.5,         # 5th percentile for minimum bounds
                    'q25': 8.1,         # First quartile for lower range
                    'median': 15.3,     # Typical processing time
                    'q75': 28.7,        # Third quartile for upper range  
                    'typical_delay': 15.3  # Primary imputation value
                }
            }
        
        Note:
            Statistics enable context-aware imputation respecting operational patterns
            while maintaining clinical workflow sequence validation and data quality.
            Used internally for missing timestamp reconstruction and delay validation.
        """
        stats = {}
        
        # Compute statistics for each configured delay pair in medication preparation workflow
        for prev_col, next_col, out_col in self.pairs:
            if next_col not in df.columns:
                continue
                
            # Extract numeric timestamp values with robust error handling
            next_s = self._to_num(df[next_col])
            
            # Compute delays only when both timestamps available for valid analysis
            if prev_col is not None and prev_col in df.columns:
                prev_s = self._to_num(df[prev_col])
                delays = next_s - prev_s
                
                # Filter for valid delays (positive values) respecting workflow chronology
                valid_delays = delays[(delays.notna()) & (delays > 0)]
                
                if not valid_delays.empty:
                    # Compute comprehensive percentile-based statistics for robust imputation
                    stats[out_col] = {
                        'min': valid_delays.quantile(0.05),        # Conservative minimum bound
                        'q25': valid_delays.quantile(0.25),        # First quartile
                        'median': valid_delays.median(),           # Central tendency
                        'q75': valid_delays.quantile(0.75),        # Third quartile
                        'typical_delay': valid_delays.median()     # Primary imputation value
                    }
        
        return stats

    def impute_missing_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply healthcare-optimized sequential imputation for missing workflow timestamps.
        
        Implements sophisticated missing data reconstruction strategy respecting medication
        preparation workflow chronology and operational context patterns. Designed for
        production deployment in pharmacy analytics environments with typical 5-15%
        missing timestamp rates due to EHR integration challenges and manual data entry gaps.
        
        Imputation Strategy:
        - Learn processing time patterns from historical data for empirically-based estimates
        - Respect chronological workflow sequence integrity for clinical interpretation validity
        - Apply operational context awareness (shift, floor patterns) for accurate reconstruction
        - Use forward interpolation with statistical bounds for realistic delay estimation
        
        Healthcare Context Integration:
        - Maintains medication preparation workflow sequence validation throughout process
        - Applies operational factor awareness for context-specific imputation patterns
        - Handles EHR integration artifacts through defensive programming strategies
        - Ensures reconstructed timestamps support downstream clinical decision-making
        
        Technical Implementation:
        - Forward pass processing maintaining strict chronological ordering constraints
        - Statistical bounds validation using learned delay distributions for quality assurance
        - Interpolation between valid timestamps with proportional time distribution
        - Fallback strategies for edge cases and data quality challenges
        
        Args:
            df: Raw TAT dataset containing missing workflow timestamps for reconstruction
               using healthcare-optimized sequential imputation and validation strategies
        
        Returns:
            pd.DataFrame: Enhanced dataset with imputed timestamps maintaining chronological
            sequence integrity and clinical workflow validation for accurate delay analysis
            and bottleneck identification in medication preparation processes
        
        Example:
            For missing timestamp reconstruction in medication preparation workflow:
            Input: Missing nurse_validation_time in 10% of orders
            Output: Imputed values respecting chronological order and operational context
            
        Note:
            Imputation maintains clinical workflow sequence validation while providing
            statistically sound timestamp reconstruction for comprehensive TAT analysis.
            Essential for robust delay computation in healthcare datasets with missing data.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        out = df.copy()
        
        # Learn comprehensive processing time patterns from historical medication preparation data
        processing_times = self._calculate_step_statistics(df)
        typical_delays_by_group = {}
        
        # Compute operational context-specific delay patterns for accurate imputation
        for shift in df['shift'].unique():
            for floor in df['floor'].unique():
                # Filter for specific operational context (shift and floor combination)
                mask = (df['shift'] == shift) & (df['floor'] == floor)
                typical_delays_by_group[(shift, floor)] = self._calculate_step_statistics(df[mask])
        
        # Extract sequence of workflow timestamp columns for chronological processing
        unwrapped_cols = []
        for _, next_col, out_col in self.pairs:
            if next_col in out.columns:
                unwrapped_cols.append((next_col, out_col))
        
        # Process each order individually to maintain workflow sequence integrity
        for idx in out.index:
            row = out.loc[idx]
            last_valid_time = 0.0  # Initialize from order time (workflow anchor point)
            
            # Extract operational context for context-aware imputation
            shift = row['shift']
            floor = row['floor']
            
            # Forward pass through workflow steps maintaining chronological constraints
            for i, (col, delay_col) in enumerate(unwrapped_cols):
                current_time = row[col]
                
                if pd.isna(current_time):
                    # Retrieve delay statistics for current workflow step and operational context
                    group_stats = typical_delays_by_group.get((shift, floor), {}).get(delay_col, {})
                    global_stats = processing_times.get(delay_col, {})
                    
                    # Prioritize operational context-specific statistics when available
                    stats = group_stats if group_stats else global_stats
                    
                    if stats:
                        # Extract statistical bounds for realistic imputation constraints
                        min_delay = stats['min']
                        typical_delay = stats['typical_delay']
                        max_delay = stats['q75']  # Use 75th percentile as upper bound
                        
                        # Identify next valid timestamp for interpolation reference point
                        next_valid_time = None
                        for future_col, _ in unwrapped_cols[i+1:]:
                            if pd.notna(row[future_col]):
                                next_valid_time = row[future_col]
                                break
                        
                        if next_valid_time is not None:
                            # Interpolate between valid timestamps with statistical bounds
                            available_minutes = next_valid_time - last_valid_time  # Both are in minutes (float)
                            steps_between = sum(1 for c, _ in unwrapped_cols[i:] 
                                              if pd.isna(row[c]) and c != col)
                            
                            if steps_between > 0:
                                # Distribute available time proportionally across missing steps
                                proposed_delay_minutes = available_minutes / (steps_between + 1)
                                # Apply statistical bounds for realistic delay constraints
                                proposed_delay_minutes = min(max(proposed_delay_minutes, min_delay), max_delay)
                            else:
                                # Use typical delay when no interpolation needed
                                proposed_delay_minutes = typical_delay
                        else:
                            # No future reference point available - use typical delay pattern
                            proposed_delay_minutes = typical_delay
                        
                        # Ensure chronological sequence integrity in workflow reconstruction (minutes as float)
                        imputed_time = last_valid_time + proposed_delay_minutes
                        out.at[idx, col] = imputed_time
                        last_valid_time = imputed_time
                    else:
                        # Conservative fallback when no statistics available for workflow step
                        out.at[idx, col] = last_valid_time + 5.0  # 5-minute conservative default (float minutes)
                        last_valid_time = out.at[idx, col]
                else:
                    # Validate existing timestamp maintains chronological sequence integrity
                    if current_time < last_valid_time:
                        # Correct chronological violations using statistical minimum delay
                        min_correction = processing_times.get(delay_col, {}).get('min', 1.0)
                        out.at[idx, col] = last_valid_time + min_correction  # Both are float minutes
                    last_valid_time = out.at[idx, col]
        
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute comprehensive delay computation pipeline for medication preparation analysis.
        
        Orchestrates end-to-end delay feature engineering workflow with healthcare-optimized
        missing data handling and clinical workflow validation. Designed for production
        deployment in pharmacy analytics environments supporting TAT prediction modeling,
        bottleneck identification, and comprehensive workflow optimization initiatives.
        
        Processing Pipeline:
        1. Optional sequential imputation for missing timestamps using operational context
        2. Step-wise delay computation from configured workflow timestamp pairs
        3. Chronological sequence validation with negative delay handling
        4. Comprehensive logging for healthcare analytics pipeline monitoring
        
        Healthcare Data Quality:
        - Sequential imputation respecting medication preparation workflow chronology
        - Negative delay detection and correction for integration data quality issues
        - Missing data pattern analysis with detailed logging for quality monitoring
        - Robust error handling for production stability in diverse data quality scenarios
        
        Feature Engineering Output:
        - delay_* columns containing step-wise processing times in minutes
        - Maintains original timestamp columns for comprehensive analysis workflows
        - Clinical workflow sequence validation throughout processing pipeline
        - Integration-ready features for TAT prediction modeling and analysis
        
        Args:
            df: Raw TAT dataset containing medication preparation workflow timestamps
               and operational context variables for comprehensive delay computation
        
        Returns:
            pd.DataFrame: Enhanced dataset with computed delay features suitable for:
            - TAT prediction modeling and bottleneck identification analysis
            - Pharmacy workflow optimization and performance monitoring
            - Clinical decision-making support and operational benchmarking
            - Integration with visualization and statistical analysis workflows
        
        Example:
            For comprehensive medication preparation delay analysis:
            delay_engineer = DelayEngineer(impute_missing=True)
            enhanced_df = delay_engineer.transform(tat_df)
            
            # Access computed delay features
            delay_features = [col for col in enhanced_df.columns if col.startswith('delay_')]
            
            # Validate delay computation results
            print(f"Computed {len(delay_features)} delay features")
            print(f"Missing data reduced from {original_missing}% to {final_missing}%")
        
        Note:
            Production-ready method suitable for automated TAT analytics pipelines and
            healthcare quality monitoring systems. Comprehensive logging enables detailed
            monitoring of data quality improvement and imputation effectiveness.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        out = df.copy()

        # Identify unwrapped timestamp columns for missing data analysis and logging
        unwrapped_cols = [col for col in df.columns if '_mins_unwrapped' in col]
        logger.info("Healthcare data quality assessment - Missing timestamp analysis before imputation:")
        for col in unwrapped_cols:
            missing = df[col].isna().sum()
            logger.info(f"  {col}: {missing:,} missing ({missing/len(df)*100:.1f}%)")

        # Apply healthcare-optimized sequential imputation when enabled
        if self.impute_missing:
            out = self.impute_missing_times(out)

            # Log imputation effectiveness for healthcare analytics pipeline monitoring
            logger.info("Healthcare data quality improvement - Missing timestamp analysis after imputation:")
            for col in unwrapped_cols:
                missing = out[col].isna().sum()
                logger.info(f"  {col}: {missing:,} missing ({missing/len(df)*100:.1f}%)")
        
        # Initialize numeric series cache for efficient delay computation
        num_cache: Dict[str, pd.Series] = {}
        def num(col: str) -> pd.Series:
            """
            Return cached numeric view for column ensuring consistent processing.
            
            Provides efficient numeric coercion with caching for repeated access during
            delay computation workflow. Essential for performance in production healthcare
            analytics environments processing large datasets with comprehensive validation.
            """
            s = num_cache.get(col)
            if s is None:
                # Create numeric series with NaN fallback for missing columns
                s = self._to_num(out[col]) if col in out.columns else pd.Series(np.nan, index=out.index)
                num_cache[col] = s
            return s

        # Compute delay features for each configured workflow step pair
        for prev_col, next_col, out_col in self.pairs:
            # Skip computation if next timestamp column missing from dataset
            if next_col not in out.columns:
                continue

            # Extract numeric timestamp values for delay calculation
            next_s = num(next_col)
            
            # Handle absolute time delays (from workflow anchor point)
            if prev_col is None:
                out[out_col] = next_s
                continue

            # Skip step if previous timestamp column missing from dataset
            if prev_col not in out.columns:
                continue

            # Compute step-wise delay with chronological validation
            prev_s = num(prev_col)
            diff = next_s - prev_s
            
            # Apply positive delay constraint (clamp negatives to 0 for workflow integrity)
            out[out_col] = diff.clip(lower=0)
            
            # Log negative delay detection for healthcare data quality monitoring
            neg_delays = (diff < 0).sum()
            if neg_delays > 0:
                logger.warning(f"Healthcare data quality issue - Found {neg_delays:,} negative delays "
                             f"in {out_col} ({neg_delays/len(df)*100:.2f}%) - corrected to 0 minutes")

        return out

    def fit(self, df: pd.DataFrame) -> "DelayEngineer":
        """
        Scikit-learn compatible fit method for estimator-like interface consistency.
        
        No-operation method maintained for compatibility with sklearn-style pipelines
        and automated machine learning workflows. DelayEngineer computes statistics
        dynamically during transform() to ensure fresh learning from current data
        patterns and operational context for accurate healthcare analytics processing.
        
        Args:
            df: TAT dataset for interface consistency (not used in processing)
        
        Returns:
            DelayEngineer: Self reference for method chaining compatibility
        
        Note:
            Maintained for sklearn pipeline compatibility in healthcare analytics workflows.
            Actual statistical learning occurs during transform() for dynamic adaptation.
        """
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method combining fit and transform for sklearn-style workflow integration.
        
        Provides streamlined interface for delay computation in automated healthcare
        analytics pipelines and machine learning workflows. Equivalent to calling
        fit() followed by transform() while maintaining method chaining compatibility
        for comprehensive feature engineering pipeline construction.
        
        Args:
            df: Raw TAT dataset for comprehensive delay computation and feature engineering
        
        Returns:
            pd.DataFrame: Enhanced dataset with computed delay features ready for analysis
        
        Example:
            For streamlined delay computation in healthcare analytics pipelines:
            delay_engineer = DelayEngineer(impute_missing=True)
            enhanced_df = delay_engineer.fit_transform(tat_df)
        """
        return self.fit(df).transform(df)

    def drop_processed_helpers(self, df: pd.DataFrame, keep: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        Remove transient helper columns for clean feature engineering pipeline output.
        
        Provides comprehensive cleanup of intermediate processing columns generated during
        timestamp reconstruction and delay computation workflows. Essential for preparing
        clean datasets for downstream modeling, analysis, and stakeholder consumption
        while maintaining data integrity and pipeline efficiency in healthcare analytics.
        
        Cleanup Strategy:
        - Remove columns matching configured suffix patterns (default: '_mins_unwrapped', '_dt')
        - Drop explicit step columns used during intermediate processing phases
        - Preserve explicitly specified columns through keep parameter for selective retention
        - Maintain original feature columns and computed delay features for analysis
        
        Args:
            df: Dataset containing computed delay features and transient helper columns
               from delay computation and timestamp reconstruction processing workflows
            keep: Optional iterable of column names to preserve during cleanup operations
                 for selective retention of intermediate columns for debugging or analysis
        
        Returns:
            pd.DataFrame: Clean dataset with helper columns removed while preserving:
            - Original feature columns for comprehensive analysis and validation
            - Computed delay_* features for modeling and bottleneck identification
            - Explicitly preserved columns specified in keep parameter for selective retention
        
        Example:
            For clean model-ready dataset preparation:
            # Compute delays with all helper columns
            enhanced_df = delay_engineer.transform(tat_df)
            
            # Clean for modeling while preserving specific debugging columns
            model_df = delay_engineer.drop_processed_helpers(
                enhanced_df, 
                keep=['nurse_validation_time_mins_unwrapped']  # For quality validation
            )
            
            # Complete cleanup for production deployment
            production_df = delay_engineer.drop_processed_helpers(enhanced_df)
        
        Note:
            Essential for maintaining clean feature engineering pipelines and preparing
            datasets for downstream consumption by modeling, visualization, and analysis
            components in comprehensive healthcare analytics workflows.
        """
        # Create defensive copy to prevent in-place modifications
        out = df.copy()
        keep_set = set(keep or [])
        to_drop: List[str] = []

        # Identify columns matching configured suffix patterns for removal
        for c in out.columns:
            if c in keep_set:
                continue
            # Check against configured transient column suffixes
            if any(c.endswith(suf) for suf in self.drop_suffixes):
                to_drop.append(c)

        # Add explicit step columns configured for removal
        for c in self.step_cols:
            if c in out.columns and c not in keep_set:
                to_drop.append(c)

        # Return original dataset if no cleanup needed
        if not to_drop:
            return out
        
        # Remove duplicates while preserving order for consistent processing
        to_drop = list(dict.fromkeys(to_drop))
        return out.drop(columns=to_drop, errors="ignore")