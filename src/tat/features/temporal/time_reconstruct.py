"""
Timestamp reconstruction for missing or partial time data.

Handles missing timestamps in workflow steps through intelligent
reconstruction and validation.
"""
from __future__ import annotations

from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from tat.config import STEP_COLS, ORDER_TIME_COL


class TimeReconstructor:
    """
    Comprehensive timestamp reconstruction system for pharmacy TAT workflow analysis.

    Orchestrates end-to-end missing timestamp reconstruction from short time fragments
    with healthcare-optimized chronological validation and sequential processing.
    Designed for production deployment in healthcare pharmacy analytics environment
    supporting comprehensive TAT analysis and medication preparation workflow optimization.

    Core Responsibilities:
    - Parse short time fragments (MM:SS, HH:MM:SS) from diverse healthcare data sources
    - Reconstruct complete timestamps anchored to medication order reference points
    - Enforce chronological sequence validation throughout medication preparation workflow
    - Generate processed columns for integration with DelayEngineer and visualization systems

    Args:
        step_cols: Optional workflow step column names for reconstruction. Defaults to
                  configured STEP_COLS covering standard medication preparation sequence.
        order_time_col: Canonical order timestamp column for anchoring reconstruction.
                       Defaults to ORDER_TIME_COL ensuring consistency across analytics.

    Example:
        # Standard timestamp reconstruction for medication preparation workflow
        reconstructor = TimeReconstructor()
        enhanced_df = reconstructor.transform(tat_df)
        
        # Access reconstructed timestamps and offsets for delay computation
        dt_columns = [col for col in enhanced_df.columns if col.endswith('_dt')]
        offset_columns = [col for col in enhanced_df.columns if col.endswith('_mins_unwrapped')]
        
        # TAT-aligned infusion time adjustment for clinical accuracy
        aligned_df = reconstructor.realign_infusion_to_tat(enhanced_df)

    Note:
        Designed for production deployment in healthcare healthcare analytics environment
        with comprehensive error handling and clinical workflow validation. Integrates with
        pharmacy operations monitoring systems and TAT prediction modeling pipelines.
    """

    def __init__(self, step_cols: List[str] = None, order_time_col: str = ORDER_TIME_COL):
        """
        Initialize timestamp reconstruction system with healthcare-optimized configuration.

        Sets up comprehensive timestamp reconstruction pipeline with appropriate defaults
        for healthcare medication preparation workflow analysis and pharmacy operations
        optimization. Configures fragment parsing, chronological validation, and clinical
        workflow sequence enforcement for robust healthcare analytics processing.

        Args:
            step_cols: Optional workflow step column names for timestamp reconstruction.
                      Defaults to configured STEP_COLS covering standard medication
                      preparation sequence from order through patient infusion completion.
            order_time_col: Canonical order timestamp column serving as anchor for
                           reconstruction. Defaults to ORDER_TIME_COL ensuring consistency
                           across healthcare healthcare analytics and TAT analysis workflows.

        Note:
            Default configuration optimized for healthcare standard medication preparation
            workflows with typical healthcare data quality patterns and EHR integration
            requirements. Custom configurations support specialized operational requirements.
        """
        # Configure workflow step columns for medication preparation sequence reconstruction
        self.step_cols = list(step_cols or STEP_COLS)
        self.order_time_col = order_time_col or ORDER_TIME_COL
        self._anchor_dt = f"{self.order_time_col}_dt"

    # --------- Healthcare Data Parsing and Validation Methods ---------

    @staticmethod
    def _parse_fragment_to_seconds(val) -> float:
        """
        Parse healthcare time fragment to seconds with robust error handling.

        Converts short time fragments from diverse healthcare data sources into
        standardized seconds representation supporting healthcare timestamp
        reconstruction workflows. Handles common EHR system formats and manual
        data entry patterns with defensive programming for production stability.

        Supported Healthcare Time Formats:
        - "MM:SS" format: Interpreted as minutes:seconds within current hour
        - "HH:MM:SS" format: Interpreted as hours:minutes:seconds for extended periods
        - Numeric values: Converted via string representation with error tolerance
        - Missing/invalid values: Gracefully handled returning NaN for downstream processing

        Args:
            val: Raw time fragment value from healthcare data sources including EHR
                exports, manual entry systems, and clinical documentation platforms.

        Returns:
            float: Seconds representation of time fragment for timestamp reconstruction,
            or np.nan for invalid/missing values enabling downstream error handling.

        Example:
            For medication preparation workflow fragment parsing:
            fragments = ["15:30", "1:25:45", "invalid", None, 1530]
            parsed = [TimeReconstructor._parse_fragment_to_seconds(f) for f in fragments]
            # Returns: [930.0, 5145.0, nan, nan, nan] (seconds representation)

        Note:
            Defensive parsing strategy prevents processing failures from malformed healthcare
            data while maintaining accurate timestamp reconstruction for valid fragments.
            Essential for robust production deployment in diverse healthcare data environments.
        """
        # Handle missing or null values gracefully for healthcare data quality patterns
        if pd.isna(val):
            return np.nan
        
        # Convert to string representation with error tolerance for diverse input types
        s = str(val).strip()
        parts = s.split(":")
        
        try:
            if len(parts) == 2:
                # MM:SS format common in healthcare workflow timing documentation
                mm, ss = parts
                return float(mm) * 60.0 + float(ss)
            elif len(parts) == 3:
                # HH:MM:SS format for extended medication preparation processes
                hh, mm, ss = parts
                return float(hh) * 3600.0 + float(mm) * 60.0 + float(ss)
        except Exception:
            # Any parsing error treated as missing data for robust error handling
            pass
        
        # Return NaN for malformed fragments enabling downstream processing continuation
        return np.nan

    # --------- Sequential Reconstruction with Clinical Validation ---------

    def _reconstruct_row(self, row: pd.Series) -> Tuple[list, list]:
        """
        Reconstruct complete timestamps for single medication order with chronological validation.

        Implements sophisticated timestamp reconstruction algorithm respecting healthcare
        medication preparation workflow chronology and clinical sequence validation.
        Processes individual medication orders maintaining strict temporal ordering and
        healthcare workflow integrity throughout reconstruction process.

        Reconstruction Algorithm:
        1. Validate anchor timestamp availability for medication order reference point
        2. Parse time fragments using healthcare-optimized parsing strategies
        3. Generate candidate timestamps anchored to order hour for temporal consistency
        4. Apply forward hour wrapping preventing timestamps before medication order
        5. Enforce monotonic sequence validation maintaining clinical workflow integrity

        Healthcare Workflow Validation:
        - Maintains medication preparation sequence chronology for clinical interpretation
        - Prevents timestamp sequence violations compromising workflow analysis accuracy
        - Handles shift transitions and hour boundaries common in healthcare operations
        - Supports diverse operational patterns while preserving clinical sequence validation

        Args:
            row: Individual medication order record containing anchor timestamp and
                workflow step fragments for comprehensive reconstruction and validation.

        Returns:
            Tuple[list, list]: Reconstructed timestamps and minute offsets from anchor
            maintaining chronological order and clinical workflow sequence validation.

        Example:
            For medication preparation workflow reconstruction:
            Input: anchor=2025-01-15 08:00:00, fragments=["15:30", "45:20", "02:10"]
            Output: Chronologically valid timestamps with appropriate hour wrapping

        Note:
            Core reconstruction algorithm ensuring clinical workflow integrity and temporal
            accuracy essential for healthcare pharmacy operations analytics and TAT
            prediction modeling workflows with comprehensive bottleneck identification.
        """
        # Validate anchor timestamp availability for medication order processing
        anchor = row[self._anchor_dt]
        if pd.isna(anchor):
            # Return missing values when anchor unavailable for downstream error handling
            n = len(self.step_cols)
            return [pd.NaT] * n, [np.nan] * n

        # Normalize to hour boundary for consistent candidate timestamp generation
        hour_anchor = anchor.replace(minute=0, second=0, microsecond=0)
        dts: List[pd.Timestamp] = []
        mins: List[float] = []
        prev_dt = anchor  # Initialize with medication order timestamp for sequence validation

        # Process each workflow step maintaining chronological sequence validation
        for col in self.step_cols:
            frag = row.get(col, None)
            if pd.isna(frag):
                # Handle missing fragments gracefully preserving sequence processing
                dts.append(pd.NaT)
                mins.append(np.nan)
                continue

            # Parse time fragment using healthcare-optimized parsing strategy
            secs = self._parse_fragment_to_seconds(frag)
            if np.isnan(secs):
                # Invalid fragments handled gracefully for robust production processing
                dts.append(pd.NaT)
                mins.append(np.nan)
                continue

            # Generate candidate timestamp anchored to medication order hour
            cand = hour_anchor + timedelta(seconds=float(secs))

            # Apply forward hour wrapping preventing timestamps before medication order
            while cand < anchor:
                cand += timedelta(hours=1)
            
            # Enforce monotonic chronology across medication preparation workflow steps
            while cand < prev_dt:
                cand += timedelta(hours=1)

            # Record validated timestamp and offset for delay computation integration
            dts.append(cand)
            mins.append((cand - anchor).total_seconds() / 60.0)
            prev_dt = cand  # Update reference for next step chronological validation

        return dts, mins

    # --------- Production-Ready Public API for Healthcare Analytics ---------

    def fit(self, df: pd.DataFrame) -> "TimeReconstructor":
        """
        Scikit-learn compatible fit method for estimator-like interface consistency.

        No-operation method maintained for compatibility with sklearn-style pipelines
        and automated machine learning workflows in healthcare healthcare analytics
        environment. TimeReconstructor operates statically without requiring training
        or parameter estimation from input data for robust timestamp reconstruction.

        Pipeline Integration:
        - Maintains sklearn transformer interface for automated ML pipeline compatibility
        - Supports integration with healthcare healthcare analytics and TAT prediction workflows
        - Enables seamless inclusion in automated feature engineering and model training pipelines
        - Provides consistent API for diverse healthcare analytics pipeline architectures

        Args:
            df: TAT dataset for interface consistency (not used in reconstruction processing)

        Returns:
            TimeReconstructor: Self reference for method chaining compatibility with
            sklearn pipeline patterns and automated Healthcare healthcare analytics workflows.

        Note:
            Maintained for sklearn pipeline compatibility in healthcare healthcare analytics
            environment. Timestamp reconstruction operates statically without requiring
            data-driven parameter estimation for consistent processing behavior.
        """
        # No-operation method for sklearn transformer interface compatibility
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute comprehensive timestamp reconstruction for healthcare TAT analysis.

        Orchestrates end-to-end timestamp reconstruction workflow generating complete
        temporal sequences from medication order fragments. Designed for production
        deployment in healthcare pharmacy analytics environment supporting comprehensive
        TAT analysis and medication preparation workflow optimization initiatives.

        Processing Workflow:
        1. Parse anchor timestamps with robust error handling for healthcare data quality
        2. Apply row-wise reconstruction maintaining chronological sequence validation
        3. Generate datetime columns for direct temporal analysis and visualization
        4. Create offset columns for DelayEngineer integration and delay computation
        5. Maintain data integrity through non-destructive DataFrame operations

        Generated Columns:
        - <order_time_col>_dt: Parsed anchor timestamp for medication order reference
        - <step>_dt: Reconstructed timestamps for each workflow step with validation
        - <step>_mins_unwrapped: Minute offsets from anchor for delay computation integration

        Args:
            df: Raw TAT dataset containing medication order anchors and workflow step
               fragments for comprehensive timestamp reconstruction and validation processing.

        Returns:
            pd.DataFrame: Enhanced dataset with reconstructed timestamps and offset columns
            suitable for healthcare TAT analysis, delay computation, and pharmacy
            workflow optimization through comprehensive temporal sequence validation.

        Example:
            For Healthcare medication preparation workflow reconstruction:
            reconstructor = TimeReconstructor()
            enhanced_df = reconstructor.transform(tat_df)
            
            # Validate reconstruction results and data quality improvement
            dt_columns = [col for col in enhanced_df.columns if col.endswith('_dt')]
            print(f"Reconstructed {len(dt_columns)} timestamp columns for TAT analysis")

        Note:
            Production-ready method suitable for automated Healthcare healthcare analytics
            pipelines with comprehensive error handling and clinical workflow validation.
            Essential foundation for TAT prediction modeling and pharmacy operations optimization.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        out = df.copy()
        
        try:
            # Try the actual format in the data first
            out[self._anchor_dt] = pd.to_datetime(out[self.order_time_col], 
                                                format='%m/%d/%Y %H:%M', errors="coerce")
        except:
            try:
                # Fallback to standard ISO format
                out[self._anchor_dt] = pd.to_datetime(out[self.order_time_col], 
                                                    format='%Y-%m-%d %H:%M:%S', errors="coerce")
            except:
                # Final fallback to pandas inference
                out[self._anchor_dt] = pd.to_datetime(out[self.order_time_col], errors="coerce")

        # Apply row-wise reconstruction maintaining chronological sequence validation
        # Results is Series of (timestamps_list, offsets_list) tuples for each medication order
        results = out.apply(self._reconstruct_row, axis=1)

        # Generate column names for reconstructed datetime and offset features
        dt_cols = [f"{c}_dt" for c in self.step_cols]
        off_cols = [f"{c}_mins_unwrapped" for c in self.step_cols]

        # Expand reconstruction results into structured DataFrames aligned with original index
        lists_dt = pd.DataFrame(results.map(lambda x: x[0]).to_list(), index=out.index)
        lists_off = pd.DataFrame(results.map(lambda x: x[1]).to_list(), index=out.index)
        lists_dt.columns = dt_cols
        lists_off.columns = off_cols

        # Integrate reconstructed columns into enhanced dataset for comprehensive analysis
        out[dt_cols] = lists_dt
        out[off_cols] = lists_off
        
        return out

    def realign_infusion_to_tat(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust patient infusion timestamps for clinical accuracy and TAT validation.

        Implements sophisticated infusion time realignment algorithm addressing hour-offset
        discrepancies between recorded fragments and expected TAT targets. Essential for
        healthcare clinical accuracy requirements and operational benchmarking ensuring
        realistic temporal sequences supporting comprehensive pharmacy workflow optimization.

        Clinical Problem Addressed:
        - Recorded infusion fragments may fall off-by-hour relative to expected TAT targets
        - EHR timestamp artifacts requiring clinical validation and operational adjustment
        - Workflow analysis accuracy depending on realistic temporal sequence validation
        - TAT prediction modeling requiring clinically consistent infusion timing patterns

        Realignment Algorithm:
        1. Calculate target infusion time based on medication order anchor plus TAT minutes
        2. Generate candidate timestamps through systematic hour shifts (±2 hours range)
        3. Filter candidates maintaining chronological sequence with prior workflow steps
        4. Select candidate minimizing distance to clinical TAT target for accuracy
        5. Update infusion timestamp and offset columns preserving workflow integrity

        Healthcare Validation Features:
        - Chronological sequence enforcement preventing clinical workflow violations
        - Clinical TAT target alignment supporting operational benchmarking accuracy
        - Robust candidate selection maintaining workflow step temporal integrity
        - Integration with DelayEngineer for consistent delay computation workflows

        Args:
            df: Enhanced TAT dataset containing reconstructed timestamps and TAT targets
               for clinical accuracy validation and infusion time realignment processing.

        Returns:
            pd.DataFrame: Dataset with realigned patient infusion timestamps maintaining
            chronological workflow validation and clinical TAT accuracy for healthcare
            comprehensive pharmacy operations optimization and bottleneck identification.

        Example:
            For Healthcare clinical accuracy and TAT validation:
            reconstructor = TimeReconstructor()
            reconstructed_df = reconstructor.transform(tat_df)
            aligned_df = reconstructor.realign_infusion_to_tat(reconstructed_df)
            
            # Validate realignment effectiveness and clinical accuracy improvement
            alignment_quality = validate_tat_alignment(aligned_df)

        Note:
            Advanced clinical validation method supporting healthcare operational accuracy
            requirements and TAT prediction modeling with realistic temporal sequences.
            Essential for comprehensive pharmacy workflow optimization and clinical benchmarking.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        out = df.copy()
        
        # Define target columns for infusion time realignment processing
        inf_dt = "patient_infusion_time_dt"
        inf_off = "patient_infusion_time_mins_unwrapped"
        
        # Validate required columns availability for realignment processing
        if inf_dt not in out or inf_off not in out or "TAT_minutes" not in out:
            # Safe no-op when required columns missing - return original dataset
            return out

        # Identify prior workflow step columns for chronological sequence validation
        prev_cols_dt = [
            "floor_dispatch_time_dt",
            "second_validation_time_dt", 
            "prep_complete_time_dt",
            "nurse_validation_time_dt",
        ]

        def _align_row(r: pd.Series):
            """
            Perform infusion time realignment for individual medication order.
            
            Implements row-level realignment algorithm maintaining clinical workflow
            chronology while optimizing infusion time accuracy relative to TAT targets.
            """
            # Extract required values for realignment processing
            base = r.get(inf_dt, pd.NaT)
            anchor = r.get(self._anchor_dt, pd.NaT)
            tat = r.get("TAT_minutes", np.nan)
            
            # Handle missing required values gracefully preserving original data
            if pd.isna(base) or pd.isna(anchor) or pd.isna(tat):
                return base, r.get(inf_off, np.nan)

            # Calculate clinical target infusion time based on TAT minutes
            target = anchor + pd.to_timedelta(float(tat), unit="m")

            # Determine latest prior workflow step for chronological validation
            prev_dt = anchor  # Initialize with medication order anchor timestamp
            for c in prev_cols_dt:
                v = r.get(c, pd.NaT)
                if pd.notna(v) and v > prev_dt:
                    prev_dt = v

            # Generate candidate timestamps through systematic hour shifts
            cands = []
            for k in (-2, -1, 0, 1, 2):  # ±2 hour range for comprehensive coverage
                cand = base + pd.to_timedelta(60 * k, unit="m")
                if cand >= prev_dt:  # Maintain chronological sequence validation
                    cands.append((abs((cand - target).total_seconds()), cand))
            
            # Fallback to all candidates if none satisfy chronological constraints
            if not cands:
                for k in (-2, -1, 0, 1, 2):
                    cand = base + pd.to_timedelta(60 * k, unit="m")
                    cands.append((abs((cand - target).total_seconds()), cand))

            # Select candidate minimizing distance to clinical TAT target
            _, best = min(cands, key=lambda t: t[0])
            new_off = (best - anchor).total_seconds() / 60.0
            return best, new_off

        # Apply row-wise realignment maintaining clinical workflow validation
        aligned = out.apply(_align_row, axis=1, result_type="expand")
        aligned.columns = [inf_dt, inf_off]
        
        # Update infusion timestamp and offset columns with realigned values
        out[inf_dt] = aligned[inf_dt]
        out[inf_off] = aligned[inf_off]
        
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method combining fit and transform for sklearn-style workflow integration.

        Provides streamlined interface for timestamp reconstruction in automated Healthcare
        healthcare analytics pipelines and machine learning workflows. Equivalent to calling
        fit() followed by transform() while maintaining method chaining compatibility for
        comprehensive feature engineering pipeline construction and TAT analysis workflows.

        Sklearn Integration Benefits:
        - Seamless inclusion in automated ML pipelines for Healthcare TAT prediction modeling
        - Consistent interface with sklearn transformers for healthcare analytics workflows
        - Method chaining support for streamlined feature engineering pipeline development
        - Integration with automated model training and validation in clinical environments

        Args:
            df: Raw TAT dataset for comprehensive timestamp reconstruction using healthcare-
               optimized processing with chronological validation and clinical sequence integrity.

        Returns:
            pd.DataFrame: Enhanced dataset with reconstructed timestamps ready for Healthcare
            TAT analysis, delay computation, and pharmacy workflow optimization initiatives.

        Example:
            For streamlined timestamp reconstruction in Healthcare analytics pipelines:
            reconstructor = TimeReconstructor()
            enhanced_df = reconstructor.fit_transform(tat_df)
            
            # Integration with sklearn pipeline for automated model training
            from sklearn.pipeline import Pipeline
            tat_pipeline = Pipeline([
                ('reconstructor', reconstructor),
                ('delay_engineer', DelayEngineer()),
                ('model', xgb_regressor)
            ])
        """
        # Execute fit and transform in sequence for sklearn-style workflow integration
        return self.fit(df).transform(df)