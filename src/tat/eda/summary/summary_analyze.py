"""
TAT Dataset Analysis Utilities for Pharmacy Workflow Optimization

Core analytical utilities supporting exploratory data analysis of medication 
preparation turnaround times and workflow bottleneck identification. Provides
defensive, production-ready functions for healthcare data preprocessing,
statistical summarization, and visualization support.

Key Functions:
- Timestamp preprocessing for pharmacy workflow steps
- Statistical summarization with clinical context
- Correlation analysis for TAT driver identification  
- Data partitioning for healthcare analytics workflows
- Unicode-aware visualization utilities for stakeholder reporting

"""
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .summary_config import SummaryConfig


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minimal preprocessing for TAT summary analysis compatibility.
    
    Performs minimal preprocessing to ensure consistent datetime handling across
    healthcare analytics workflows while preserving original data characteristics
    for clinical validation and quality assurance workflows.

    Healthcare Data Normalization:
    - Parse all time columns to standardized datetime format for temporal analysis
    - Preserve original data types and encoding for clinical interpretation
    - Maintain data integrity for regulatory compliance and audit trails

    Args:
        df: Raw TAT dataset requiring datetime standardization

    Returns:
        pd.DataFrame: Minimally processed dataset with standardized temporal columns

    Note:
        Gracefully handles malformed timestamps common in healthcare EHR exports.
        Parsing failures result in NaT values for downstream missing data strategies.
    """
    dfc = df.copy()
    
    # Identify complete timestamp columns vs time fragment columns
    # Complete timestamps: doctor_order_time (full datetime format)
    # Time fragments: nurse_validation_time, prep_complete_time, etc. (MM:SS.decimal format)
    complete_timestamp_columns = [col for col in dfc.columns if col == 'doctor_order_time']
    time_fragment_columns = [col for col in dfc.columns 
                           if 'time' in col.lower() and col != 'doctor_order_time']
    
    # Only convert complete timestamp columns to datetime
    # Time fragments are kept as strings for EDA display purposes
    time_columns = complete_timestamp_columns
    for col in time_columns:
        if col in dfc.columns:
            try:
                # Handle different datetime formats in healthcare data
                if col == 'doctor_order_time':
                    # Full datetime format: '4/15/2025 6:27'
                    dfc[col] = pd.to_datetime(dfc[col], format='%m/%d/%Y %H:%M', errors="coerce")
                else:
                    # Other time columns are time fragments (MM:SS.decimal format)
                    # For EDA purposes, leave them as strings - they are not complete timestamps
                    # These need TimeReconstructor processing to become proper datetimes
                    # but for basic EDA summary, we show them as categorical/text data
                    pass  # Keep as original string format for EDA display
            except Exception:
                # Healthcare data often contains irregular timestamp formats
                # Handle gracefully for robust EDA processing
                pass
    return dfc


def supports_unicode(cfg: SummaryConfig) -> bool:
    """
    Determine Unicode support for healthcare analytics visualizations.

    Enables appropriate character encoding for pharmacy team dashboards and
    clinical stakeholder reports. Ensures consistent visualization rendering
    across different healthcare IT environments and deployment platforms.

    Encoding Strategy:
    - cfg.force_ascii=True: ASCII-only output for legacy healthcare systems
    - cfg.force_ascii=False: Unicode visualization for modern analytics platforms  
    - cfg.force_ascii=None: Auto-detection based on deployment environment

    Args:
        cfg: Summary configuration with encoding preferences

    Returns:
        bool: True if Unicode visualizations supported, False for ASCII fallback

    Note:
        Healthcare IT environments often have mixed encoding support.
        Auto-detection provides robust fallback for production deployments.
    """
    if cfg.force_ascii is True:
        return False
    if cfg.force_ascii is False:
        return True
    enc = (getattr(sys.stdout, "encoding", None) or "").lower()
    return ("utf-8" in enc) or ("utf8" in enc)


def sparkbar(arr: Sequence[int], cfg: SummaryConfig) -> str:
    """
    Generate compact distribution visualizations for TAT workflow analysis.

    Creates inline sparkline-style histograms suitable for pharmacy dashboard
    integration and clinical team reports. Optimized for displaying medication
    preparation step delays and bottleneck intensity patterns in minimal space.

    Visualization Modes:
    - Unicode: High-resolution block characters (▁▂▃▄▅▆▇█) for modern displays
    - ASCII: Character-based steps ( .:-=+*#@) for legacy healthcare terminals

    Args:
        arr: Integer counts from histogram binning (e.g., delay distribution)
        cfg: Configuration determining visualization encoding and scaling

    Returns:
        str: Compact sparkline representation scaled relative to maximum value

    Example:
        For TAT delay bins [1,3,8,12,6,2]:
        Unicode: "▁▃▆█▅▂"
        ASCII: " :=@+."

    Note:
        Empty input arrays return empty strings for robust report generation.
        Scaling normalizes against maximum value for consistent visual comparison.
    """
    if not arr:
        return ""
    mx = max(arr) or 1
    if supports_unicode(cfg):
        blocks = "▁▂▃▄▅▆▇█"
        scale = len(blocks) - 1  # 7 (indices 0-7)
        return "".join(blocks[min(scale, int(round((v / mx) * scale)))] for v in arr)
    steps = " .:-=+*#@"
    scale = len(steps) - 1  # 8 (indices 0-8)
    return "".join(steps[min(scale, int(round((v / mx) * scale)))] for v in arr)

def hist_counts(s: pd.Series, bins: int) -> Tuple[List[int], List[str]]:
    """
    Compute histogram statistics for TAT delay distribution analysis.

    Generates binned count distributions suitable for pharmacy workflow bottleneck
    identification and medication preparation timing analysis. Handles healthcare
    data irregularities including missing values and low-cardinality measures.

    Statistical Processing:
    - Numeric coercion with missing value exclusion for robust healthcare data
    - Adaptive binning with duplicate handling for sparse delay measurements
    - Sorted categorical intervals for temporal workflow interpretation

    Args:
        s: Numeric series containing TAT delays or workflow step durations
        bins: Number of histogram bins for distribution analysis

    Returns:
        Tuple containing:
        - counts_list: Integer counts per histogram bin
        - labels_list: String representations of bin intervals
        Returns ([], []) for empty or non-numeric series

    Example:
        For TAT_minutes series, bins=5:
        counts: [120, 350, 200, 80, 50] 
        labels: ['(0, 20]', '(20, 40]', '(40, 60]', '(60, 80]', '(80, 100]']

    Note:
        Pandas cut() with duplicates='drop' handles edge cases in healthcare
        timing data where step durations may have limited unique values.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return [], []
    cats = pd.cut(s, bins=bins, duplicates="drop")
    counts = cats.value_counts().sort_index()
    labels = [str(ix) for ix in counts.index]
    return [int(x) for x in counts.tolist()], labels


def cat_top_inline(s: pd.Series, k: int, cfg: SummaryConfig) -> str:
    """
    Generate inline categorical summaries for pharmacy operational analysis.

    Creates compact representations of categorical distributions for healthcare
    team consumption, emphasizing dominant categories in medication preparation
    workflows (e.g., shift patterns, credential types, department distribution).

    Display Format:
    - Category(percentage): e.g., "Day(45%) • Evening(30%) • Night(25%)"
    - Missing values displayed as "NaN" with computed percentage
    - Unicode/ASCII separator adaptation for healthcare IT compatibility

    Args:
        s: Categorical series (shift, credential, department, etc.)
        k: Maximum number of top categories to display
        cfg: Configuration for Unicode support and formatting preferences

    Returns:
        str: Formatted inline summary with top-k categories and percentages

    Example:
        For nurse_credential series:
        "BSN(42%) • RN(31%) • MSN(18%) • NP(9%)"

    Note:
        Includes missing data as explicit category for healthcare data quality
        assessment. Percentage calculations include NaN counts in denominator.
    """
    vc = s.value_counts(dropna=False)
    if vc.empty:
        return ""
    total = len(s)
    joiner = " • " if supports_unicode(cfg) else " | "
    parts: List[str] = []
    for idx, cnt in vc.head(k).items():
        name = "NaN" if pd.isna(idx) else str(idx)
        frac = cnt / total
        parts.append(f"{name}({frac:.0%})")
    return joiner.join(parts)


def numeric_describe(s: pd.Series, cfg: SummaryConfig) -> Dict[str, float]:
    """
    Generate comprehensive statistical summaries for TAT numeric analysis.

    Computes clinical-relevant descriptive statistics for medication preparation
    timing variables, queue metrics, and operational performance indicators.
    Optimized for healthcare analytics with configurable percentile analysis.

    Statistical Metrics:
    - Central tendency: mean, median (50th percentile)  
    - Variability: standard deviation, min/max ranges
    - Distribution: configurable percentiles for outlier analysis
    - Sample size: count of valid observations for clinical interpretation

    Args:
        s: Numeric series (TAT_minutes, queue_length, occupancy_pct, etc.)
        cfg: Configuration specifying percentiles for healthcare analytics

    Returns:
        Dict containing statistical measures:
        - count, mean, std, min, 50% (median), max
        - p{N}: percentile values (e.g., p10, p90, p95, p99)
        Returns empty dict for non-numeric or empty series

    Example:
        For TAT_minutes analysis:
        {
            'count': 98500.0, 'mean': 42.3, 'std': 18.7,
            'min': 8.2, '50%': 38.1, 'max': 127.4,
            'p10': 21.3, 'p90': 67.8, 'p95': 79.2, 'p99': 98.7
        }

    Note:
        Uses unbiased standard deviation (ddof=1) for population inference.
        Missing values excluded from calculations for robust healthcare analytics.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return {}
    q = s.quantile(cfg.percentiles, interpolation="linear")
    out = {
        "count": float(s.count()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "50%": float(s.median()),
        "max": float(s.max()),
    }
    for p, val in zip(cfg.percentiles, q.values):
        out[f"p{int(round(100 * p))}"] = float(val)
    return out


def example_value(s: pd.Series) -> str:
    """
    Extract representative sample value for healthcare data documentation.

    Provides concrete example for data understanding and validation in clinical
    contexts. Essential for pharmacy team review of data formats, encoding
    schemes, and categorical value interpretation during TAT analysis setup.

    Sampling Strategy:
    - Scans first 50 non-null values for efficiency in large healthcare datasets
    - Returns first valid observation as representative sample
    - Graceful fallback to "NaN" for completely missing columns

    Args:
        s: Any pandas Series from TAT dataset

    Returns:
        str: Representative non-null value or "NaN" if none found

    Example:
        For nurse_credential series: "BSN"
        For TAT_minutes series: "43.2" 
        For empty/all-null series: "NaN"

    Note:
        Limited to 50-value scan prevents performance issues with large
        healthcare datasets while providing sufficient sampling coverage.
    """
    # Handle datetime columns specially to ensure proper formatting
    if pd.api.types.is_datetime64_any_dtype(s):
        for v in s.head(100):  # Check more values for datetime columns
            if pd.notna(v) and v != pd.NaT:
                # Format datetime nicely for display
                return v.strftime('%Y-%m-%d %H:%M:%S') if hasattr(v, 'strftime') else str(v)
    
    # Handle other data types
    for v in s.head(50):
        if pd.notna(v):
            return str(v)
    return "NaN"


def suppress_hist(col_name: str, cfg: SummaryConfig) -> bool:
    """
    Determine histogram suppression for specific TAT analysis columns.

    Controls visualization generation based on clinical relevance and data
    characteristics. Prevents unnecessary histogram generation for identifier
    columns, high-cardinality categoricals, or sensitive healthcare data.

    Suppression Logic:
    - Columns in cfg.no_hist_cols excluded from distribution analysis
    - Typically applied to: patient IDs, physician names, order numbers
    - Preserves computational resources for relevant workflow metrics

    Args:
        col_name: Column name to evaluate for histogram generation
        cfg: Configuration containing histogram exclusion list

    Returns:
        bool: True if histogram should be suppressed, False to generate

    Note:
        Essential for HIPAA compliance and performance optimization in
        large-scale healthcare analytics with sensitive identifier columns.
    """
    return col_name in cfg.no_hist_cols


def missingness(df: pd.DataFrame, cfg: SummaryConfig) -> pd.Series:
    """
    Analyze missing data patterns in TAT dataset for data quality assessment.

    Computes column-wise missingness ratios to identify data collection issues
    in pharmacy workflow systems, incomplete EHR integration, or systematic
    gaps in medication preparation tracking systems.

    Missing Data Analysis:
    - Fraction missing per column (0.0 = complete, 1.0 = entirely missing)
    - Optional sorting by missingness severity for prioritized data quality review
    - Supports pharmacy team identification of critical workflow tracking gaps

    Args:
        df: TAT dataset from pharmacy operations system
        cfg: Configuration controlling missing data analysis sorting

    Returns:
        pd.Series: Column-wise missing data ratios indexed by column name
        Optionally sorted descending by missing fraction for quality review

    Example:
        For TAT dataset analysis:
        prep_complete_time         0.12
        second_validation_time     0.08  
        floor_dispatch_time        0.05
        lab_ALT_U_L               0.03
        TAT_minutes               0.00

    Note:
        Healthcare data commonly has 5-15% missingness in workflow timestamps
        due to system integration challenges and manual data entry gaps.
    """
    s = df.isna().mean()
    return s.sort_values(ascending=False) if cfg.sort_missing else s


def numeric_correlations(
    df: pd.DataFrame,
    cfg: SummaryConfig,
    extra_exclude: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute correlation matrix for TAT driver identification and bottleneck analysis.

    Generates Pearson correlation analysis focused on numeric workflow variables
    while excluding identifier columns and high-cardinality categoricals.
    Essential for understanding relationships between operational factors and
    medication preparation delays.

    Correlation Scope:
    - Includes: TAT metrics, queue lengths, occupancy rates, lab values
    - Excludes: Patient IDs, timestamps, configured exclusion columns
    - Filters: Prefix-based exclusions for systematic column group removal

    Args:
        df: TAT dataset with numeric operational and clinical variables
        cfg: Configuration specifying correlation exclusions and prefixes  
        extra_exclude: Additional columns to exclude from correlation analysis

    Returns:
        pd.DataFrame: Correlation matrix for numeric variables
        Returns empty DataFrame if <2 numeric columns remain after filtering

    Example:
        Key correlations for TAT optimization:
        - queue_length_at_order vs TAT_minutes: 0.34
        - floor_occupancy_pct vs TAT_minutes: 0.28
        - pharmacists_on_duty vs TAT_minutes: -0.19

    Note:
        Correlation analysis guides feature engineering for TAT prediction models
        and identifies operational levers for pharmacy workflow optimization.
    """
    num = df.select_dtypes(include=[np.number]).copy()
    if num.empty or num.shape[1] < 2:
        return pd.DataFrame()

    drop_cols: Set[str] = set(cfg.corr_exclude_columns) | set(extra_exclude or [])
    for c in list(num.columns):
        if c in drop_cols:
            num.drop(columns=[c], inplace=True, errors="ignore")
            continue
        for pref in cfg.corr_exclude_prefixes:
            if c.startswith(pref):
                num.drop(columns=[c], inplace=True, errors="ignore")
                break

    return num.corr() if num.shape[1] > 1 else pd.DataFrame()


def partition_columns(df: pd.DataFrame, cfg: SummaryConfig) -> Tuple[List[str], List[str], List[str]]:
    """
    Classify TAT dataset columns by data type for targeted healthcare analytics.

    Automatically categorizes columns into temporal, categorical, and numeric
    groups based on healthcare data conventions and domain knowledge. Enables
    appropriate statistical analysis and visualization strategies for each
    data type in pharmacy workflow optimization.

    Classification Rules:
    - Temporal: datetime dtypes, timestamp columns, names containing 'time'/_dt suffix
    - Categorical: non-numeric types, healthcare codes, configured categorical overrides  
    - Numeric: continuous measures (TAT, delays, counts, lab values, percentages)

    Args:
        df: TAT dataset from pharmacy operations system
        cfg: Configuration with healthcare-specific column categorization rules

    Returns:
        Tuple containing three lists:
        - time_cols: Timestamp columns for temporal workflow analysis
        - cat_cols: Categorical variables for stratified bottleneck analysis  
        - num_cols: Continuous metrics for statistical modeling and correlation

    Example:
        time_cols: ['doctor_order_time', 'nurse_validation_time', ...]
        cat_cols: ['shift', 'nurse_credential', 'diagnosis_type', ...]  
        num_cols: ['TAT_minutes', 'queue_length_at_order', 'lab_WBC_k_per_uL', ...]

    Note:
        Proper column classification ensures appropriate analysis methods:
        temporal (trend analysis), categorical (group comparisons), 
        numeric (correlation/regression modeling).
    """
    time_cols: List[str] = []
    cat_cols: List[str] = []
    num_cols: List[str] = []
    for col in df.columns:
        s = df[col]
        is_dt = pd.api.types.is_datetime64_any_dtype(s)
        looks_timey = (col in cfg.known_time_cols) or col.endswith("_dt") or ("time" in col.lower())
        if is_dt or looks_timey:
            time_cols.append(col)
            continue
        is_numeric = pd.api.types.is_numeric_dtype(s)
        is_forced_cat = (col in cfg.categorical_overrides) or any(col.startswith(p) for p in cfg.categorical_prefixes)
        if (not is_numeric) or is_forced_cat:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return time_cols, cat_cols, num_cols