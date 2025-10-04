"""
Tabular Artifact Builders for TAT Analysis Summary Generation

Production-ready table construction utilities for pharmacy turnaround time exploratory
data analysis and medication preparation workflow bottleneck identification. Provides
lightweight, testable functions that transform TAT datasets into structured summaries
suitable for healthcare stakeholder consumption and clinical decision-making.

Key Components:
- Workflow timestamp analysis tables for step-by-step delay identification
- Operational factor summaries for staffing and resource optimization insights
- TAT performance metric tables with embedded distribution visualization data
- Correlation analysis builders for bottleneck driver identification
- Missing data assessment tables for healthcare data quality monitoring

"""
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .summary_config import SummaryConfig
from . import summary_analyze as A


def time_table(df: pd.DataFrame, cols: List[str], cfg: SummaryConfig) -> pd.DataFrame:
    """
    Generate workflow timestamp analysis table for medication preparation process evaluation.

    Creates comprehensive summary of sequential workflow steps from physician order
    through patient administration, highlighting data completeness and temporal patterns
    critical for bottleneck identification in pharmacy operations.

    Timestamp Analysis Components:
    - Data completeness assessment for each workflow step
    - Missing data percentage calculation for data quality monitoring
    - Representative samples for clinical team validation and interpretation
    - Dtype information for data pipeline validation and ETL quality assurance

    Args:
        df: TAT dataset containing medication preparation workflow timestamps
        cols: List of validated timestamp column names for sequential analysis
        cfg: Configuration specifying healthcare data quality and analysis parameters

    Returns:
        pd.DataFrame: Structured timestamp analysis with columns:
        - name: Workflow step identifier (doctor_order_time, nurse_validation_time, etc.)
        - type: Data type for pipeline validation and quality assurance
        - n_nonnull: Count of complete timestamp records for workflow step coverage
        - miss%: Missing data percentage for healthcare data quality assessment
        - sample: Representative timestamp value for clinical team validation

    Example:
        For medication preparation workflow analysis:
        name                    type                n_nonnull  miss%   sample
        doctor_order_time       datetime64[ns]      98,500     1.5%    2025-03-15 09:23:41
        nurse_validation_time   datetime64[ns]      89,200     10.8%   2025-03-15 09:31:18
        prep_complete_time      datetime64[ns]      92,100     7.9%    2025-03-15 10:05:22

    Note:
        Healthcare workflow timestamps commonly show 5-15% missingness due to
        EHR integration gaps and manual data entry challenges. Focus on steps
        with >15% missing data for immediate data quality improvement initiatives.
    """
    rows: List[Dict[str, Any]] = []
    miss = A.missingness(df, cfg)
    
    for col in cols:
        if col not in df.columns:
            # Defensive handling for production robustness with healthcare datasets
            continue
            
        s = df[col]
        missing_pct = float(miss.get(col, s.isna().mean())) * 100
        
        rows.append({
            "name": col,
            "type": str(s.dtype),
            "n_nonnull": int(s.notna().sum()),
            "miss%": round(missing_pct, 2),
            "sample": A.example_value(s),
        })
    
    return pd.DataFrame(rows)


def categorical_table(df: pd.DataFrame, cols: List[str], cfg: SummaryConfig) -> pd.DataFrame:
    """
    Build operational factor analysis table for pharmacy workflow optimization insights.

    Analyzes categorical variables affecting medication preparation TAT including staffing
    patterns, clinical contexts, and operational conditions. Essential for identifying
    workflow optimization opportunities and resource allocation strategies.

    Operational Factor Categories:
    - Staffing: shift patterns, nurse/pharmacist credentials, employment experience
    - Clinical Context: diagnosis types, treatment modalities, patient acuity levels
    - Operational: floor assignments, department sources, urgency classifications
    - Infrastructure: queue states, occupancy levels, resource availability patterns

    Args:
        df: TAT dataset with categorical operational and clinical variables
        cols: List of validated categorical column names for operational analysis
        cfg: Configuration with healthcare-specific categorical analysis parameters

    Returns:
        pd.DataFrame: Comprehensive operational factor summary with columns:
        - name: Operational factor identifier for workflow optimization mapping
        - type: Data type for pipeline validation and ETL quality assurance
        - n_nonnull: Complete records count for factor analysis coverage assessment
        - miss%: Missing data percentage for healthcare data quality monitoring
        - nunique: Cardinality for high-dimensional factor identification
        - sample: Representative category value for clinical team interpretation
        - top_values: Dominant categories with percentages for workflow focus areas

    Example:
        For pharmacy staffing and operational context:
        name              nunique  top_values
        shift             3        Day(45%) • Evening(32%) • Night(23%)
        nurse_credential  4        BSN(42%) • RN(31%) • MSN(18%) • NP(9%)
        floor             3        Floor_1(38%) • Floor_2(35%) • Floor_3(27%)
        severity          3        Medium(48%) • High(31%) • Low(21%)

    Note:
        Focus on dominant categories (>30%) for immediate workflow optimization impact.
        High cardinality factors (>50 unique values) may require grouping strategies.
    """
    rows: List[Dict[str, Any]] = []
    miss = A.missingness(df, cfg)
    
    for col in cols:
        if col not in df.columns:
            # Defensive handling for production robustness with healthcare datasets
            continue
            
        s = df[col]
        missing_pct = float(miss.get(col, s.isna().mean())) * 100
        cardinality = int(s.nunique(dropna=True))
        
        # Apply cardinality-based top values display for clinical interpretability
        top_values_display = ""
        if not A.suppress_hist(col, cfg) and cardinality <= cfg.max_categorical_cardinality:
            top_values_display = A.cat_top_inline(s, cfg.cat_top, cfg)
        elif cardinality > cfg.max_categorical_cardinality:
            top_values_display = f"High cardinality ({cardinality} categories)"
        
        rows.append({
            "name": col,
            "type": str(s.dtype),
            "n_nonnull": int(s.notna().sum()),
            "miss%": round(missing_pct, 2),
            "nunique": cardinality,
            "sample": A.example_value(s),
            "top_values": top_values_display,
        })
    
    return pd.DataFrame(rows)


def numeric_table(df: pd.DataFrame, cols: List[str], cfg: SummaryConfig) -> pd.DataFrame:
    """
    Generate TAT  Numerical Features and clinical indicators summary table.

    Constructs comprehensive analysis of continuous variables including turnaround times,
    queue metrics, occupancy rates, laboratory values, and operational performance
    indicators. Integrates distribution visualization data for stakeholder reporting
    and bottleneck identification in medication preparation workflows.

    Performance Metric Categories:
    - TAT Measures: total turnaround time, step-wise delays, violation indicators
    - Queue Metrics: queue lengths, wait times, processing backlogs
    - Occupancy Rates: floor utilization, resource availability, capacity constraints
    - Clinical Indicators: laboratory values affecting preparation complexity
    - Staffing Metrics: pharmacist/nurse counts, experience levels, workload distribution

    Args:
        df: TAT dataset with numeric performance and clinical measurement variables
        cols: List of validated numeric column names for statistical analysis
        cfg: Configuration specifying percentiles and healthcare analytics parameters

    Returns:
        pd.DataFrame: Comprehensive numeric analysis with columns:
        - name: Performance metric identifier for clinical interpretation
        - type: Data type for pipeline validation and quality assurance
        - n_nonnull: Complete measurements count for statistical reliability assessment
        - miss%: Missing data percentage for healthcare data quality monitoring
        - nunique: Unique value count for discrete vs continuous classification
        - sample: Representative numeric value for clinical team validation
        - min/p50/max: Statistical summary for performance benchmarking
        - distribution: Compact sparkline visualization for pattern identification
        - _dist_counts/_dist_labels: Internal histogram data for HTML report integration

    Example:
        For TAT performance analysis:
        name                    n_nonnull   min    p50    max    distribution
        TAT_minutes             98,500      8.2    38.1   127.4  ▁▃▆█▅▂▁
        queue_length_at_order   97,800      0      3      24     ▂█▃▁▁▁
        floor_occupancy_pct     99,100      12.5   67.3   98.7   ▁▂▅█▆▂
        lab_WBC_k_per_uL        89,200      2.1    6.8    18.3   ▂▅█▃▁

    Note:
        Distribution sparklines provide immediate visual assessment of data patterns.
        Focus on metrics with >15% missing data for data quality improvement.
        Statistical summaries enable clinical benchmark comparison and trend analysis.
    """
    rows: List[Dict[str, Any]] = []
    miss = A.missingness(df, cfg)
    
    for col in cols:
        if col not in df.columns:
            # Defensive handling for production robustness with healthcare datasets
            continue
            
        s = df[col]
        desc = A.numeric_describe(s, cfg)
        missing_pct = float(miss.get(col, s.isna().mean())) * 100
        
        # Generate distribution visualization data for TAT pattern analysis
        counts: List[int] = []
        labels: List[str] = []
        distribution_display = ""
        
        if not A.suppress_hist(col, cfg) and not s.empty:
            counts, labels = A.hist_counts(s, cfg.hist_bins)
            if counts:  # Only generate sparkline if data exists
                distribution_display = A.sparkbar(counts, cfg)
        
        rows.append({
            "name": col,
            "type": str(s.dtype),
            "n_nonnull": int(s.notna().sum()),
            "miss%": round(missing_pct, 2),
            "nunique": int(s.nunique(dropna=True)),
            "sample": A.example_value(s),
            "min": desc.get("min", np.nan),
            "p50": desc.get("50%", np.nan),  # Median for clinical interpretation
            "max": desc.get("max", np.nan),
            "distribution": distribution_display,
            "_dist_counts": counts,      # Internal data for HTML histogram rendering
            "_dist_labels": labels,      # Internal data for HTML tooltip generation
        })
    
    # Ensure numeric precision for clinical interpretation and stakeholder reporting
    out = pd.DataFrame(rows)
    for stat_col in ("min", "p50", "max"):
        if stat_col in out.columns:
            out[stat_col] = pd.to_numeric(out[stat_col], errors="coerce").round(3)
    
    return out


def build_artifacts(df: pd.DataFrame, cfg: SummaryConfig) -> Dict[str, Any]:
    """
    Assemble comprehensive TAT analysis artifacts for pharmacy workflow optimization.

    Central orchestration function that processes raw medication preparation datasets
    into structured analysis components suitable for healthcare stakeholder consumption,
    clinical decision-making, and automated TAT monitoring systems.

    Analysis Pipeline:
    1. Healthcare data preprocessing (timestamp parsing, encoding standardization)
    2. Column classification (temporal, categorical, numeric) with clinical context
    3. Workflow timestamp analysis for step-by-step bottleneck identification
    4. Operational factor analysis for staffing and resource optimization insights
    5. Performance metric analysis with distribution visualization integration
    6. Correlation analysis for TAT driver identification and improvement prioritization
    7. Data quality assessment for healthcare workflow integrity monitoring

    Args:
        df: Raw TAT dataset from pharmacy operations system
        cfg: Healthcare analytics configuration with TAT-specific parameters

    Returns:
        Dict containing comprehensive analysis artifacts:
        - time_table: Workflow timestamp analysis for bottleneck identification
        - categorical_table: Operational factor summaries for workflow optimization
        - numeric_table: TAT  Numerical Features with embedded visualizations
        - correlations: Feature correlation matrix for driver identification
        - missing_table: Data quality assessment for workflow integrity monitoring
        - missing_table_console: Truncated missing data summary for terminal display
        - counts: Metadata summary for executive reporting and pipeline monitoring
        - df_processed: Preprocessed dataset for downstream analysis and modeling

    Example Usage:
        artifacts = build_artifacts(tat_df, pharmacy_config)
        
        # Access workflow bottleneck analysis
        timestamp_issues = artifacts['time_table']
        
        # Review operational optimization opportunities  
        staffing_insights = artifacts['categorical_table']
        
        # Examine performance distributions and trends
        tat_metrics = artifacts['numeric_table']
        
        # Identify improvement drivers
        correlation_matrix = artifacts['correlations']

    Note:
        Designed for both interactive analysis and automated pipeline integration.
        All artifacts include clinical interpretation support and healthcare terminology.
        Robust error handling ensures production stability with incomplete datasets.
    """
    # Apply healthcare-specific preprocessing for consistent temporal and encoding handling
    dfc = A.basic_preprocess(df)
    nrows_total, ncols_total = dfc.shape

    # Classify columns with healthcare domain knowledge for appropriate analysis methods
    tcols, ccols, ncols_list = A.partition_columns(dfc, cfg)

    # Generate workflow-specific analysis tables with clinical context
    time_tbl = time_table(dfc, tcols, cfg) if tcols else pd.DataFrame()
    cat_tbl = categorical_table(dfc, ccols, cfg) if ccols else pd.DataFrame()
    num_tbl = numeric_table(dfc, ncols_list, cfg) if ncols_list else pd.DataFrame()

    # Compute TAT driver correlation analysis for bottleneck identification
    corr = A.numeric_correlations(dfc, cfg)

    # Healthcare data quality assessment with clinical significance thresholds
    miss = A.missingness(dfc, cfg)
    miss_nonzero = miss[miss > 0]
    
    # Generate data quality tables for pharmacy team review and improvement planning
    missing_tbl = pd.DataFrame({
        "column": miss_nonzero.index,
        "missing%": (miss_nonzero.values * 100).round(2)
    }).reset_index(drop=True)
    
    # Console-optimized missing data summary for interactive analysis
    missing_tbl_console = missing_tbl.head(cfg.missing_top_n)

    # Executive summary metadata for pharmacy leadership reporting
    counts = {
        "rows": int(nrows_total),
        "cols": int(ncols_total),
        "time": int(len(tcols)),
        "categorical": int(len(ccols)),
        "numeric": int(len(ncols_list)),
        "missing_cols": int((A.missingness(dfc, cfg) > 0).sum()),
    }

    return {
        "time_table": time_tbl,
        "categorical_table": cat_tbl,
        "numeric_table": num_tbl,
        "correlations": corr,
        "missing_table": missing_tbl,
        "missing_table_console": missing_tbl_console,
        "counts": counts,
        "df_processed": dfc,
    }


def correlation_lower(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Generate lower-triangular correlation matrix for clean TAT analysis visualization.

    Transforms full correlation matrix to display format suitable for healthcare
    stakeholder reporting and clinical dashboard integration. Eliminates redundant
    upper-triangle correlations for improved readability in pharmacy workflow analysis.

    Matrix Transformation:
    - Preserves lower triangle for complete pairwise relationship visibility
    - Sets upper triangle to NaN for clean visualization without information redundancy
    - Maintains diagonal (self-correlation = 1.0) for matrix completeness validation
    - Optimizes space utilization in clinical reports and dashboard presentations

    Args:
        corr: Full correlation matrix from TAT driver analysis

    Returns:
        pd.DataFrame: Lower-triangular correlation matrix with upper triangle blanked
        Returns empty DataFrame if input correlation matrix is empty

    Example:
        For TAT correlation analysis display:
        Input (full):                    Output (lower triangular):
                 TAT   queue  occupancy              TAT   queue  occupancy
        TAT      1.00   0.34    0.28      TAT      1.00    NaN     NaN
        queue    0.34   1.00    0.15      queue    0.34   1.00     NaN  
        occupancy 0.28   0.15    1.00     occupancy 0.28   0.15    1.00

    Note:
        Essential for correlation heatmap visualization in healthcare analytics reports.
        Reduces visual complexity while preserving all unique pairwise relationships.
        Supports automated report generation with clean, professional presentation.
    """
    if corr.empty:
        return corr.copy()
    
    # Generate boolean mask for lower triangle including diagonal
    mask = np.tril(np.ones(corr.shape), k=0).astype(bool)
    lower = corr.copy()
    
    # Blank upper triangle for clean visualization
    lower.values[~mask] = np.nan
    
    return lower


def correlation_pairs_table(
    df: pd.DataFrame, cfg: SummaryConfig, min_abs: float = 0.1, top_k: int = 50
) -> pd.DataFrame:
    """
    Generate prioritized correlation pairs table for TAT bottleneck driver identification.

    Extracts actionable correlation insights from pharmacy workflow analysis by identifying
    feature pairs with clinically significant relationships. Essential for prioritizing
    workflow optimization initiatives and resource allocation decisions based on
    quantified impact on medication preparation turnaround times.

    Clinical Significance Criteria:
    - Minimum correlation threshold (default: |r| ≥ 0.10) for operational relevance
    - Sorted by correlation strength for intervention prioritization  
    - Limited to top-K pairs for focused improvement initiatives
    - Excludes self-correlations and duplicate pairs for actionable insights

    Args:
        df: TAT dataset for correlation analysis computation
        cfg: Healthcare analytics configuration with correlation exclusion rules
        min_abs: Minimum absolute correlation for clinical significance (default: 0.1)
        top_k: Maximum correlation pairs for focused workflow optimization (default: 50)

    Returns:
        pd.DataFrame: Prioritized correlation pairs with columns:
        - feature_a: Primary operational or clinical factor
        - feature_b: Secondary factor with significant relationship
        - r: Pearson correlation coefficient with directional information
        - |r|: Absolute correlation strength for intervention prioritization
        
        Returns empty DataFrame when no correlations meet significance threshold

    Example:
        For TAT workflow optimization prioritization:
        feature_a              feature_b                r      |r|
        queue_length_at_order  TAT_minutes            0.342   0.342
        floor_occupancy_pct    TAT_minutes            0.287   0.287
        pharmacists_on_duty    TAT_minutes           -0.194   0.194
        nurse_employment_years prep_complete_delay   -0.156   0.156

    Note:
        Positive correlations indicate factors increasing TAT (intervention targets).
        Negative correlations suggest protective factors (resource optimization opportunities).
        Focus on top 10-15 pairs for maximum workflow improvement impact assessment.
    """
    # Generate correlation matrix with healthcare-specific exclusions
    corr = A.numeric_correlations(df, cfg)
    if corr.empty:
        return pd.DataFrame()
    
    # Extract significant correlation pairs for workflow optimization analysis
    cols = list(corr.columns)
    pairs: List[Tuple[str, str, float, float]] = []
    n = len(cols)
    
    for i in range(n):
        for j in range(i + 1, n):  # Avoid self-correlation and duplicates
            correlation_value = corr.iloc[i, j]
            
            if pd.isna(correlation_value):
                continue
            
            abs_correlation = float(abs(correlation_value))
            
            # Apply clinical significance threshold for actionable insights
            if abs_correlation >= float(min_abs):
                pairs.append((
                    cols[i], 
                    cols[j], 
                    float(correlation_value), 
                    abs_correlation
                ))
    
    # Generate prioritized correlation insights for pharmacy workflow optimization
    correlation_pairs_df = pd.DataFrame(
        pairs, 
        columns=["feature_a", "feature_b", "r", "|r|"]
    )
    
    if correlation_pairs_df.empty:
        return correlation_pairs_df
    
    # Sort by correlation strength for intervention prioritization
    return correlation_pairs_df.sort_values("|r|", ascending=False).head(int(top_k)).reset_index(drop=True)