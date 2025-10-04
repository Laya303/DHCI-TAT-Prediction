"""
Data preparation pipelines for TAT prediction models.

Provides F0 (real-time inference) and diagnostics (analysis) dataset
creation with comprehensive feature engineering and validation.
"""
import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from tat.config import NON_FEATURE_COLS, TARGETS, ORDER_TIME_COL, CATEGORICAL_PREFIX_MAP, REQUIRED_COLUMNS
from tat.features.temporal.time_reconstruct import TimeReconstructor
from tat.features.cleaners import Cleaner
from tat.features.temporal import TemporalEngineer
from tat.features.categoricals import CategoricalEncoder
from tat.features.labs import LabProcessor
from tat.features.temporal.delays import DelayEngineer
from tat.features.operational import OperationalEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_input_data(df: pd.DataFrame) -> None:
    """
    Validate required columns and data ranges for healthcare analytics pipeline.

    Performs comprehensive input validation ensuring data quality and completeness
    for medication preparation TAT analysis. Essential for maintaining healthcare
    data integrity and supporting accurate pharmacy workflow optimization through
    robust data validation and quality assurance processes in clinical environments.

    Healthcare Data Validation:
    - Required column presence: Validates essential healthcare variables for TAT analysis
    - Clinical range validation: Ensures medically realistic values for operational variables
    - Data quality reporting: Provides detailed validation results for healthcare analytics
    - Pipeline safety: Prevents downstream processing errors through comprehensive validation

    Validation Checks:
    - Column completeness: All required healthcare variables present for analysis
    - Operational ranges: Floor occupancy percentages within realistic healthcare bounds
    - Queue validation: Non-negative queue lengths for pharmacy workflow analysis
    - Clinical consistency: Healthcare operational variables within expected ranges

    Args:
        df: Input TAT dataset for validation ensuring healthcare data quality and
           completeness supporting pharmacy workflow optimization requirements.

    Raises:
        ValueError: When required healthcare columns are missing compromising TAT analysis
        capabilities and pharmacy workflow optimization data processing requirements.

    Example:
        For comprehensive healthcare data validation in TAT analysis pipeline:
        validate_input_data(tat_df)
        # Validates 50+ required columns including timestamps, patient data, staffing
    
    """
    logger.info("Validating input data requirements...")
    
    # Validate required healthcare columns for comprehensive TAT analysis
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    logger.info(f"✓ All {len(REQUIRED_COLUMNS)} required columns present")

    # Validate operational ranges with detailed healthcare analytics reporting
    invalid_occupancy = ~df['floor_occupancy_pct'].between(0, 100)
    if invalid_occupancy.any():
        logger.warning(f"floor_occupancy_pct: {invalid_occupancy.sum()} values outside 0-100 range")
        logger.warning(f"  Range: {df['floor_occupancy_pct'].min():.1f} - {df['floor_occupancy_pct'].max():.1f}")
    else:
        logger.info("✓ floor_occupancy_pct values within valid range (0-100)")
    
    # Validate queue length data for pharmacy workflow analysis
    invalid_queue = df['queue_length_at_order'] < 0
    if invalid_queue.any():
        logger.warning(f"queue_length_at_order: {invalid_queue.sum()} negative values found")
        logger.warning(f"  Min value: {df['queue_length_at_order'].min()}")
    else:
        logger.info("✓ queue_length_at_order values non-negative")

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create regression and classification targets for TAT prediction modeling.

    Generates comprehensive target variables supporting both regression and classification
    approaches to medication preparation TAT analysis. Essential for healthcare analytics
    enabling accurate TAT prediction modeling and pharmacy workflow optimization through
    appropriate target variable creation and clinical threshold-based classification.

    Target Variable Creation:
    - TAT_minutes: Continuous regression target for precise TAT prediction modeling
    - TAT_over_60: Binary classification target based on clinical threshold standards
    - Clinical relevance: Targets aligned with healthcare operational excellence requirements
    - Model flexibility: Supports diverse machine learning approaches for TAT analysis

    Healthcare Analytics Integration:
    - Clinical threshold alignment: 60-minute threshold based on healthcare quality standards
    - Operational decision support: Binary classification enabling clinical workflow decisions
    - Performance monitoring: Target statistics supporting healthcare quality assessment
    - Pharmacy optimization: Targets enabling workflow bottleneck identification and improvement

    Args:
        df: Input TAT dataset for target variable creation supporting healthcare analytics
           and TAT prediction modeling requirements in pharmacy workflow optimization.

    Returns:
        pd.DataFrame: Enhanced dataset with target variables suitable for comprehensive
        TAT prediction modeling and pharmacy workflow optimization supporting healthcare
        analytics and clinical decision-making processes in medication preparation environments.

    Example:
        For comprehensive TAT target creation in healthcare analytics workflow:
        target_df = create_target_variables(tat_df)
        
        # Access created targets for model development
        regression_target = target_df['TAT_minutes']
        classification_target = target_df['TAT_over_60']
    """
    logger.info("Creating target variables...")
    df = df.copy()
    
    # Create TAT_minutes if not present in healthcare dataset
    if 'TAT_minutes' not in df.columns:
        logger.warning("TAT_minutes column not found - computing from timestamps")
        order_dt = pd.to_datetime(df['doctor_order_time'])
        infusion_dt = pd.to_datetime(df['patient_infusion_time'])
        df['TAT_minutes'] = (infusion_dt - order_dt).dt.total_seconds() / 60
        logger.info(f"Computed TAT_minutes from timestamps: mean={df['TAT_minutes'].mean():.1f} minutes")
    else:
        # Report TAT statistics for healthcare analytics monitoring
        tat_stats = df['TAT_minutes'].describe()
        logger.info(f"TAT_minutes statistics: mean={tat_stats['mean']:.1f}, "
                   f"median={tat_stats['50%']:.1f}, std={tat_stats['std']:.1f}")
        logger.info(f"TAT_minutes range: {tat_stats['min']:.1f} - {tat_stats['max']:.1f} minutes")

    # Create classification target based on clinical threshold (60 minutes)
    df['TAT_over_60'] = (df['TAT_minutes'] > 60).astype(int)
    over_60_pct = df['TAT_over_60'].mean() * 100
    logger.info(f"✓ Created TAT_over_60: {over_60_pct:.1f}% of orders exceed 60-minute threshold")
    
    return df

def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataset into feature matrix and target variables for machine learning workflows.

    Separates healthcare dataset into features and targets supporting comprehensive TAT
    prediction modeling and pharmacy workflow optimization. Essential for machine learning
    pipeline preparation enabling accurate model training and validation through proper
    data separation and healthcare analytics workflow integration in clinical environments.

    Data Separation Strategy:
    - Feature isolation: Removes target variables from feature matrix for model training
    - Target extraction: Separates regression and classification targets for diverse modeling
    - Pipeline preparation: Enables sklearn-compatible workflows for healthcare analytics
    - Data integrity: Maintains original dataset structure while preparing ML inputs

    Healthcare Analytics Integration:
    - Model training preparation: Features and targets ready for TAT prediction modeling
    - Workflow optimization support: Separated data enabling pharmacy analytics workflows
    - Clinical decision support: Prepared datasets supporting healthcare operational analysis
    - Healthcare ML pipeline: Compatible with automated model training and validation systems

    Args:
        df: Input TAT dataset containing features and targets for machine learning
           preparation supporting healthcare analytics and pharmacy workflow optimization.

    Returns:
        Tuple containing:
        - pd.DataFrame: Feature matrix for TAT prediction model training and validation
        - pd.Series: Regression target (TAT_minutes) for continuous TAT prediction modeling
        - pd.Series: Classification target (TAT_over_60) for binary threshold prediction

    Example:
        For healthcare analytics machine learning pipeline preparation:
        X, y_reg, y_clf = split_features_targets(tat_df)
        
        # Use separated data for model training
        model_reg.fit(X, y_reg)
        model_clf.fit(X, y_clf)
    """
    logger.debug("Splitting features and targets...")
    
    # Extract target variables for TAT prediction modeling
    y_reg = df['TAT_minutes']
    y_clf = df['TAT_over_60']
    
    # Create feature matrix by removing target columns
    X = df.drop(columns=TARGETS)
    
    logger.debug(f"Split complete: {X.shape[1]} features, {len(y_reg)} samples")
    return X, y_reg, y_clf

def scale_features_selectively(df: pd.DataFrame, model_type: str = "mixed") -> Tuple[pd.DataFrame, dict]:
    """
    Apply model-appropriate scaling strategy for healthcare TAT prediction modeling.
    
    Implements healthcare-optimized feature scaling preserving clinical interpretability
    while optimizing model performance for TAT prediction and pharmacy workflow optimization.
    Essential for machine learning model preparation with healthcare domain considerations
    and clinical data characteristics supporting accurate TAT analysis and optimization.

    Healthcare Scaling Considerations:
    - Clinical interpretability preservation: Maintains healthcare operational context meaning
    - Delay column protection: Preserves unscaled delay features for bottleneck analysis
    - Distribution handling: Addresses skewed healthcare data patterns appropriately
    - Model optimization: Scaling strategies tailored for diverse machine learning algorithms

    Scaling Strategies:

    Tree Models ('tree'):
    - Minimal scaling with log-transformation for heavily skewed operational features
    - Preserves all features unscaled including delay columns for interpretability
    - Handles queue length skewness through log1p transformation
    - Maintains clinical context for tree-based model interpretation

    Linear Models ('linear'):
    - RobustScaler for skewed features (labs, operational) excluding delay columns
    - StandardScaler for temporal features with normal distributions
    - Preserves categorical and delay features unscaled for clinical interpretation
    - Balances performance with healthcare domain interpretability requirements

    Mixed Models ('mixed'):
    - RobustScaler for all numerical features excluding delay columns
    - Preserves categorical and delay features for comprehensive model ensemble
    - Optimized for ensemble methods and stacking approaches
    - Maintains clinical interpretability while enabling advanced modeling techniques

    Args:
        df: Feature DataFrame containing healthcare variables for TAT prediction modeling
           supporting pharmacy workflow optimization and clinical analytics requirements.
        model_type: Scaling strategy ('tree', 'linear', 'mixed') optimized for specific
                   machine learning algorithms and healthcare analytics use cases.

    Returns:
        Tuple containing:
        - pd.DataFrame: Scaled feature matrix optimized for specified model type
        - dict: Scaler metadata including scalers, unscaled features, and strategy information

    Example:
        For healthcare analytics model preparation with ensemble approach:
        scaled_df, scaler_info = scale_features_selectively(features_df, model_type="mixed")
        
        # Verify delay columns preserved for bottleneck analysis
        delay_features = [col for col in scaled_df.columns if col.startswith('delay_')]
        print(f"Preserved {len(delay_features)} delay columns for clinical analysis")

    """
    logger.info(f"Applying selective scaling strategy for {model_type} models...")
    
    df_scaled = df.copy()
    scaler_info = {'scalers': {}, 'unscaled_features': [], 'strategy': model_type}
    
    # Identify delay columns to preserve unscaled for bottleneck analysis
    delay_cols = [col for col in df.columns if col.startswith('delay_')]
    
    # Identify feature types (excluding delay columns from scaling)
    lab_cols = [col for col in df.columns if col.startswith('lab_')]
    temporal_cols = [col for col in df.columns if any(x in col.lower() for x in ['hour', 'day', 'month']) 
                     and not col.startswith('delay_')]
    operational_cols = ['floor_occupancy_pct', 'queue_length_at_order', 'pharmacists_on_duty']
    categorical_cols = [col for col in df.columns if df[col].dtype == 'uint8' or 
                       ('_' in col and col not in lab_cols and col not in delay_cols and df[col].nunique() <= 20)]
    
    if model_type == "tree":
        # Tree models: No scaling needed, but log-transform heavy skew
        logger.info("Tree-based strategy: Minimal scaling, log-transform skewed features")
        
        # Log-transform heavily skewed operational features (excluding delays)
        skewed_ops = ['queue_length_at_order']
        for col in skewed_ops:
            if col in df.columns:
                # Add 1 to handle zeros, common in queue data
                df_scaled[f'{col}_log'] = np.log1p(df_scaled[col])
                df_scaled = df_scaled.drop(columns=[col])
                logger.info(f"  Applied log1p transform to {col}")
        
        # All features remain unscaled for tree models (including delays)
        scaler_info['unscaled_features'] = list(df_scaled.columns)
        logger.info(f"  Preserved ALL {len(df_scaled.columns)} features unscaled (including {len(delay_cols)} delay columns)")
        
    elif model_type == "linear":
        # Linear models: Need careful scaling for interpretability
        logger.info("Linear model strategy: RobustScaler for skewed, StandardScaler for normal, preserve delays")
        
        from sklearn.preprocessing import RobustScaler, StandardScaler
        
        # RobustScaler for skewed features (labs, operational) - EXCLUDING delays
        skewed_features = [col for col in (lab_cols + operational_cols) if col not in delay_cols]
        if skewed_features:
            robust_scaler = RobustScaler()
            existing_skewed = [col for col in skewed_features if col in df_scaled.columns]
            if existing_skewed:
                df_scaled[existing_skewed] = robust_scaler.fit_transform(df_scaled[existing_skewed])
                scaler_info['scalers']['robust'] = {'scaler': robust_scaler, 'features': existing_skewed}
                logger.info(f"  Applied RobustScaler to {len(existing_skewed)} skewed features")
        
        # StandardScaler for temporal features (more normally distributed) - EXCLUDING delays
        temporal_numeric = [col for col in temporal_cols if col in df_scaled.columns and 
                           df_scaled[col].dtype in ['float64', 'int64'] and col not in delay_cols]
        if temporal_numeric:
            standard_scaler = StandardScaler()
            df_scaled[temporal_numeric] = standard_scaler.fit_transform(df_scaled[temporal_numeric])
            scaler_info['scalers']['standard'] = {'scaler': standard_scaler, 'features': temporal_numeric}
            logger.info(f"  Applied StandardScaler to {len(temporal_numeric)} temporal features")
            
        # Leave categorical AND delay features unscaled
        unscaled_features = [col for col in categorical_cols if col in df_scaled.columns] + delay_cols
        scaler_info['unscaled_features'] = unscaled_features
        logger.info(f"  Left {len(categorical_cols)} categorical features unscaled")
        logger.info(f"  PRESERVED {len(delay_cols)} delay columns unscaled for bottleneck analysis")
        
    else:  # mixed - for ensemble/stacking
        # Hybrid approach: Robust scaling for interpretability + performance
        logger.info("Mixed strategy: RobustScaler for numerical, preserve categorical AND delay columns")
        
        from sklearn.preprocessing import RobustScaler
        
        # RobustScaler for all numerical features EXCEPT delay columns
        numerical_features = lab_cols + operational_cols + [col for col in temporal_cols 
                           if col in df_scaled.columns and df_scaled[col].dtype in ['float64', 'int64']]
        
        # CRITICAL: Remove delay columns from scaling
        numerical_features = [col for col in numerical_features if col not in delay_cols]
        
        if numerical_features:
            robust_scaler = RobustScaler()
            existing_numerical = [col for col in numerical_features if col in df_scaled.columns]
            df_scaled[existing_numerical] = robust_scaler.fit_transform(df_scaled[existing_numerical])
            scaler_info['scalers']['robust_all'] = {'scaler': robust_scaler, 'features': existing_numerical}
            logger.info(f"  Applied RobustScaler to {len(existing_numerical)} numerical features")
        
        # Keep categorical binary features AND delay columns as-is
        categorical_features = [col for col in df_scaled.columns if col not in numerical_features]
        scaler_info['unscaled_features'] = categorical_features
        logger.info(f"  Preserved {len(categorical_features) - len(delay_cols)} categorical features unscaled")
        logger.info(f"  PRESERVED {len(delay_cols)} delay columns unscaled for bottleneck analysis")
    
    # Verify delay columns are preserved for clinical analysis
    preserved_delays = [col for col in delay_cols if col in df_scaled.columns]
    if len(preserved_delays) != len(delay_cols):
        logger.warning(f"Some delay columns were lost during scaling! Expected {len(delay_cols)}, got {len(preserved_delays)}")
    else:
        logger.info(f"✓ All {len(delay_cols)} delay columns successfully preserved unscaled")
    
    logger.info(f"✓ Selective scaling complete for {model_type} models")
    return df_scaled, scaler_info

def scale_numeric_features(df: pd.DataFrame, strategy: str = "mixed") -> Tuple[pd.DataFrame, dict]:
    """
    Apply healthcare-optimized feature scaling for TAT prediction modeling.

    Wrapper function providing healthcare-aware scaling strategies optimized for diverse
    machine learning algorithms and TAT prediction requirements. Essential for model
    preparation ensuring optimal performance while preserving clinical interpretability
    and healthcare domain knowledge in pharmacy workflow optimization analytics.

    Args:
        df: Feature DataFrame for scaling supporting TAT prediction modeling and
           healthcare analytics requirements in pharmacy workflow optimization.
        strategy: Scaling approach ('mixed', 'tree', 'linear') optimized for specific
                 machine learning models and healthcare analytics use cases.

    Returns:
        Tuple containing scaled DataFrame and scaler metadata for healthcare analytics
        model preparation and pharmacy workflow optimization requirements.
    """
    logger.info("Applying healthcare-optimized feature scaling...")
    
    # Use mixed strategy for model ensemble approaches
    return scale_features_selectively(df, model_type=strategy)

def remove_redundant_features(X: pd.DataFrame, 
                            variance_threshold: float = 0.01,
                            correlation_threshold: float = 0.95,
                            preserve_delay_features: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    Remove redundant features while preserving healthcare analytics capabilities.

    Eliminates low-variance and highly correlated features to improve model performance
    and reduce overfitting while maintaining clinical interpretability and healthcare
    domain knowledge. Essential for healthcare analytics ensuring optimal feature sets
    for TAT prediction modeling and pharmacy workflow optimization in clinical environments.

    Redundancy Removal Strategy:
    - Low variance elimination: Removes features with minimal information content
    - Correlation analysis: Identifies and removes highly correlated feature pairs
    - Clinical preservation: Maintains healthcare domain-relevant features for interpretation
    - Model optimization: Reduces dimensionality while preserving predictive capability

    Feature Quality Assessment:
    - Variance threshold filtering: Eliminates nearly constant features affecting model training
    - Correlation analysis: Reduces multicollinearity improving model stability and performance
    - Information preservation: Retains features with clinical significance and predictive value
    - Computational efficiency: Reduces feature space improving model training and inference speed

    Args:
        X: Feature DataFrame for redundancy analysis supporting TAT prediction modeling
           and healthcare analytics requirements in pharmacy workflow optimization.
        variance_threshold: Minimum variance threshold for feature retention ensuring
                           meaningful information content for healthcare analytics modeling.
        correlation_threshold: Maximum correlation threshold before feature removal preventing
                             multicollinearity issues in TAT prediction modeling workflows.
        preserve_delay_features: If True, preserves delay_* columns regardless of variance
                               for bottleneck analysis and clinical interpretation.

    Returns:
        Tuple containing:
        - pd.DataFrame: Cleaned feature matrix with redundant features removed
        - dict: Removal information including eliminated features and reduction statistics

    Example:
        For healthcare analytics feature optimization in TAT prediction workflow:
        X_clean, removal_info = remove_redundant_features(
            X_features, 
            variance_threshold=0.01,
            correlation_threshold=0.95,
            preserve_delay_features=True
        )
        
        # Review feature reduction results
        print(f"Features: {removal_info['original_features']} → {removal_info['final_features']}")

    """
    logger.info(f"Removing redundant features (variance_threshold={variance_threshold}, "
               f"correlation_threshold={correlation_threshold})...")
    
    removal_info = {
        'low_variance': [],
        'high_correlation': [],
        'original_features': len(X.columns),
        'final_features': 0
    }
    
    X_clean = X.copy()
    
    # Identify delay columns to preserve for bottleneck analysis
    delay_cols = [col for col in X_clean.columns if col.startswith('delay_')]
    if preserve_delay_features and delay_cols:
        logger.info(f"Preserving {len(delay_cols)} delay features for clinical analysis...")
    
    # Step 1: Remove low variance features for healthcare analytics optimization
    logger.info("Step 1: Checking for low variance features...")
    selector = VarianceThreshold(threshold=variance_threshold)
    try:
        # Create subset excluding delay columns if preserving them
        if preserve_delay_features and delay_cols:
            # Apply variance threshold only to non-delay columns
            non_delay_cols = [col for col in X_clean.columns if not col.startswith('delay_')]
            if non_delay_cols:
                X_non_delay = X_clean[non_delay_cols]
                selector.fit(X_non_delay)
                low_var_mask = selector.get_support()
                low_var_cols = X_non_delay.columns[~low_var_mask].tolist()
                # Keep delay columns and non-low-variance features
                keep_cols = delay_cols + X_non_delay.columns[low_var_mask].tolist()
                X_clean = X_clean[keep_cols]
            else:
                # Only delay columns present, keep all
                low_var_cols = []
        else:
            # Apply variance threshold to all columns (original behavior)
            selector.fit(X_clean)
            low_var_mask = selector.get_support()
            low_var_cols = X_clean.columns[~low_var_mask].tolist()
            X_clean = X_clean.loc[:, low_var_mask]
        
        removal_info['low_variance'] = low_var_cols
        
        if low_var_cols:
            logger.info(f"  Removed {len(low_var_cols)} low variance features:")
            for i, col in enumerate(low_var_cols[:5]):  # Show first 5
                logger.info(f"    - {col}")
            if len(low_var_cols) > 5:
                logger.info(f"    ... and {len(low_var_cols) - 5} more")
            if preserve_delay_features and delay_cols:
                preserved_delays = [col for col in delay_cols if col in X_clean.columns]
                logger.info(f"  Preserved {len(preserved_delays)} delay features for analysis")
        else:
            logger.info("  No low variance features found")
    except Exception as e:
        logger.warning(f"Variance threshold filtering failed: {e}")
    
    # Step 2: Remove highly correlated features for model stability
    logger.info("Step 2: Checking for highly correlated features...")
    if len(X_clean.columns) > 1:
        try:
            corr_matrix = X_clean.corr().abs()
            
            # Find pairs of highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > correlation_threshold:
                        col_i, col_j = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col_i, col_j, corr_matrix.iloc[i, j]))
            
            # Remove one feature from each highly correlated pair
            # Keep the first feature, remove the second
            to_remove = set()
            for col_i, col_j, corr_val in high_corr_pairs:
                if col_j not in to_remove:
                    to_remove.add(col_j)
                    removal_info['high_correlation'].append({
                        'removed': col_j,
                        'correlated_with': col_i,
                        'correlation': corr_val
                    })
            
            if to_remove:
                X_clean = X_clean.drop(columns=list(to_remove))
                logger.info(f"  Removed {len(to_remove)} highly correlated features:")
                for item in removal_info['high_correlation'][:5]:  # Show first 5
                    logger.info(f"    - {item['removed']} (r={item['correlation']:.3f} with {item['correlated_with']})")
                if len(removal_info['high_correlation']) > 5:
                    logger.info(f"    ... and {len(removal_info['high_correlation']) - 5} more")
            else:
                logger.info("  No highly correlated features found")
                
        except Exception as e:
            logger.warning(f"Correlation filtering failed: {e}")
    else:
        logger.info("  Skipping correlation check (insufficient features)")
    
    removal_info['final_features'] = len(X_clean.columns)
    
    logger.info(f"✓ Feature reduction complete: {removal_info['original_features']} → "
               f"{removal_info['final_features']} features "
               f"({removal_info['original_features'] - removal_info['final_features']} removed)")
    
    return X_clean, removal_info

def build_base(df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """
    Build foundational features used by both F0 and diagnostics datasets.

    Creates comprehensive base feature set through healthcare-optimized transformations
    supporting both real-time inference and analytical workflows. Essential for TAT
    prediction modeling providing consistent feature engineering foundation across
    different dataset types while maintaining clinical interpretability and data quality.

    Base Feature Engineering Pipeline:
    - Input validation: Ensures healthcare data quality and completeness requirements
    - Identifier removal: Cleans dataset removing non-predictive identifier columns
    - Data cleaning: Applies clinical bounds validation and healthcare data quality enhancement
    - Laboratory processing: Transforms clinical lab values with medical reference validation
    - Operational features: Generates workflow and capacity indicators for pharmacy optimization

    Healthcare Analytics Integration:
    - Clinical validation: Ensures medically realistic data ranges and healthcare consistency
    - Operational context: Preserves pharmacy workflow meaning throughout feature engineering
    - Quality assurance: Comprehensive data cleaning supporting accurate TAT prediction modeling
    - Scalable processing: Efficient pipeline suitable for large healthcare datasets

    Args:
        df: Input TAT dataset containing raw healthcare variables for comprehensive
           feature engineering supporting pharmacy workflow optimization requirements.
        validate: Enable comprehensive input validation ensuring healthcare data quality
                 and completeness for accurate TAT prediction modeling workflows.

    Returns:
        pd.DataFrame: Base feature set with comprehensive healthcare transformations
        suitable for both F0 (real-time) and diagnostics (analytical) dataset creation
        supporting TAT prediction modeling and pharmacy workflow optimization analytics.

    Example:
        For healthcare analytics base feature creation in TAT prediction pipeline:
        base_features = build_base(raw_tat_df, validate=True)
        
        # Base features include cleaned data, lab processing, operational indicators
        print(f"Base features created: {base_features.shape[1]} columns")
    """
    logger.info("Building base features...")
    logger.info(f"Input shape: {df.shape}")
    
    # Validate healthcare data quality and completeness
    if validate:
        validate_input_data(df)
    
    out = df.copy()

    # Drop identifier columns early in the pipeline
    logger.info("Dropping non-feature identifier columns...")
    initial_cols = len(out.columns)
    out = out.drop(columns=NON_FEATURE_COLS, errors='ignore')
    dropped_cols = initial_cols - len(out.columns)
    if dropped_cols > 0:
        logger.info(f"  Dropped {dropped_cols} identifier columns")
    logger.info(f"  Shape after identifier removal: {out.shape}")

    # Apply comprehensive healthcare data cleaning transformations
    logger.info("Applying data cleaning transformations...")
    cleaner = Cleaner()
    out = cleaner.clip_age(out)
    out = cleaner.clip_years(out)
    out = cleaner.coerce_binary(out, ["premed_required", "stat_order"])
    
    # Ensure age bounds are enforced (defensive programming for healthcare data integrity)
    if 'patient_age' in out.columns:
        out['patient_age'] = pd.to_numeric(out['patient_age'], errors='coerce').clip(lower=0, upper=100)
    if 'age' in out.columns:
        out['age'] = pd.to_numeric(out['age'], errors='coerce').clip(lower=0, upper=100)
        
    logger.info(f"  Shape after cleaning: {out.shape}")
    
    # Process laboratory values with clinical validation and enhancement
    logger.info("Processing laboratory values...")
    labs = LabProcessor.default()
    initial_lab_cols = len(out.columns)
    out = labs.transform(out)
    new_lab_features = len(out.columns) - initial_lab_cols
    if new_lab_features > 0:
        logger.info(f"  Added {new_lab_features} lab-derived features")
    logger.info(f"  Shape after lab processing: {out.shape}")

    # Generate operational features for pharmacy workflow optimization
    logger.info("Generating operational features...")
    ops = OperationalEngineer.default()
    initial_ops_cols = len(out.columns)
    out = ops.transform(out)
    new_ops_features = len(out.columns) - initial_ops_cols
    if new_ops_features > 0:
        logger.info(f"  Added {new_ops_features} operational features")
    logger.info(f"  Shape after operational features: {out.shape}")
    
    logger.info(f"✓ Base feature engineering complete. Final shape: {out.shape}")
    return out

def make_f0(df: pd.DataFrame, scaling_strategy: str = "mixed") -> Tuple[pd.DataFrame, pd.Series, pd.Series, Optional[StandardScaler], dict]:
    """
    Build F0 dataset with features safe for real-time TAT prediction inference.

    Creates production-ready feature set excluding any future information ensuring
    safe real-time deployment for medication preparation TAT prediction. Essential
    for healthcare operations supporting live pharmacy workflow optimization while
    preventing data leakage and maintaining clinical decision-making capabilities.

    F0 Dataset Characteristics:
    - Real-time safety: No future information leakage preventing deployment issues
    - Order-time features only: Uses information available at medication order time
    - Production optimization: Efficient feature set for real-time inference requirements
    - Clinical relevance: Maintains healthcare operational context for decision support

    Feature Engineering Pipeline:
    - Base transformations: Comprehensive healthcare feature engineering foundation
    - Categorical encoding: One-hot encoding for machine learning model compatibility
    - Temporal features: Order-time-based temporal patterns without future information
    - Target creation: Regression and classification targets for diverse modeling approaches
    - Timestamp removal: Eliminates time columns preventing production deployment issues

    Real-time Deployment Considerations:
    - Information availability: Only features available at medication order time
    - Computational efficiency: Optimized feature set for production inference speed
    - Data integrity: Prevents future information leakage compromising model validity
    - Operational integration: Compatible with healthcare workflow and decision systems

    Args:
        df: Input TAT dataset for F0 feature creation supporting real-time TAT prediction
           and pharmacy workflow optimization in production healthcare environments.
        scaling_strategy: Feature scaling approach ('mixed', 'tree', 'linear') optimized
                         for specific machine learning algorithms and deployment requirements.

    Returns:
        Tuple containing:
        - pd.DataFrame: F0 feature matrix safe for real-time TAT prediction inference
        - pd.Series: Regression target (TAT_minutes) for continuous prediction modeling
        - pd.Series: Classification target (TAT_over_60) for binary threshold prediction
        - dict: Scaler information for production deployment and feature transformation
        - dict: Feature removal information for model optimization and quality monitoring

    Example:
        For production-ready TAT prediction model deployment:
        X_f0, y_reg, y_clf, scaler_info, removal_info = make_f0(
            tat_df, 
            scaling_strategy="mixed"
        )
        
        # Deploy model with F0 features for real-time inference
        production_model.fit(X_f0, y_reg)
    """
    logger.info("=" * 60)
    logger.info("CREATING F0 DATASET (Real-time prediction features)")
    logger.info("=" * 60)
    
    # Build comprehensive base feature set for healthcare analytics
    out = build_base(df)

    # Apply categorical encodings for machine learning compatibility
    logger.info("Applying categorical encodings...")
    encoder = CategoricalEncoder(one_hot_prefix_map=CATEGORICAL_PREFIX_MAP)
    initial_cat_cols = len(out.columns)
    out = encoder.transform(out)
    new_cat_features = len(out.columns) - initial_cat_cols
    logger.info(f"  Added {new_cat_features} categorical features via one-hot encoding")
    logger.info(f"  Shape after categorical encoding: {out.shape}")

    # Generate temporal features based on order-time only (real-time safe)
    logger.info("Generating temporal features (order-time based only)...")
    te = TemporalEngineer()
    initial_temporal_cols = len(out.columns)
    out = te.transform(out)
    new_temporal_features = len(out.columns) - initial_temporal_cols
    logger.info(f"  Added {new_temporal_features} temporal features")
    logger.info(f"  Shape after temporal features: {out.shape}")
    
    # Create comprehensive target variables for TAT prediction modeling
    out = create_target_variables(out)
    
    # Remove ALL time-related columns for real-time inference safety
    logger.info("Removing timestamp columns for real-time inference safety...")
    time_cols = [col for col in out.columns if any(suffix in col.lower() 
                for suffix in ['_time', '_dt', '_mins_unwrapped'])]
    if time_cols:
        logger.info(f"  Removing {len(time_cols)} timestamp-related columns")
        logger.debug(f"  Timestamp columns: {time_cols}")
    out = out.drop(columns=time_cols, errors='ignore')
    logger.info(f"  Shape after timestamp removal: {out.shape}")

    # Split features and targets for machine learning workflow preparation
    logger.info("Splitting features and targets...")
    X, y_reg, y_clf = split_features_targets(out)
    logger.info(f"  Feature matrix: {X.shape}")
    logger.info(f"  Regression target: {y_reg.shape}")
    logger.info(f"  Classification target: {y_clf.shape}")
    
    # Apply healthcare-optimized feature scaling for model optimization
    X_scaled, scaler_info = scale_numeric_features(X, strategy=scaling_strategy)
    
    # Remove redundant features for model efficiency and performance
    X_final, removal_info = remove_redundant_features(X_scaled)
    
    logger.info(f"✓ F0 dataset creation complete. Final feature matrix: {X_final.shape}")
    return X_final, y_reg, y_clf, scaler_info, removal_info

def make_diagnostics(df: pd.DataFrame, scaling_strategy: str = "mixed") -> Tuple[pd.DataFrame, pd.Series, pd.Series, Optional[StandardScaler], dict]:
    """
    Build diagnostics dataset including step-to-step delays for bottleneck analysis.

    Creates comprehensive analytical dataset including delay features for pharmacy
    workflow bottleneck identification and TAT analysis. Essential for healthcare
    operations research supporting pharmacy workflow optimization through detailed
    step-timing analysis and comprehensive medication preparation process understanding.

    Diagnostics Dataset Characteristics:
    - Delay inclusion: Step-to-step delay features for bottleneck identification analysis
    - Analytical depth: Comprehensive features supporting detailed workflow optimization research
    - Process insight: Medication preparation timing patterns for operational improvement
    - Clinical analysis: Healthcare workflow understanding supporting quality enhancement

    Feature Engineering Pipeline:
    - Base transformations: Comprehensive healthcare feature engineering foundation
    - Time reconstruction: Rebuilds complete timestamps from fragmented time data
    - Delay computation: Calculates step-to-step delays for bottleneck identification
    - Temporal features: Time-based patterns supporting comprehensive workflow analysis
    - Categorical encoding: Machine learning compatible categorical transformations

    Bottleneck Analysis Capabilities:
    - Step-timing analysis: Individual workflow step delay identification and assessment
    - Process optimization: Detailed timing data supporting pharmacy workflow improvement
    - Resource allocation: Delay patterns enabling staffing and capacity optimization
    - Quality monitoring: Workflow timing supporting healthcare operations excellence

    Args:
        df: Input TAT dataset for diagnostics feature creation supporting comprehensive
           workflow analysis and pharmacy operations optimization in healthcare environments.
        scaling_strategy: Feature scaling approach ('mixed', 'tree', 'linear') optimized
                         for analytical modeling and bottleneck identification requirements.

    Returns:
        Tuple containing:
        - pd.DataFrame: Diagnostics feature matrix with delay features for analysis
        - pd.Series: Regression target (TAT_minutes) for continuous workflow analysis
        - pd.Series: Classification target (TAT_over_60) for threshold-based assessment
        - dict: Scaler information for analytical model preparation and optimization
        - dict: Feature removal information for analytical model efficiency and quality

    Example:
        For comprehensive pharmacy workflow bottleneck analysis:
        X_diag, y_reg, y_clf, scaler_info, removal_info = make_diagnostics(
            tat_df,
            scaling_strategy="mixed"  
        )
        
        # Analyze delay features for bottleneck identification
        delay_features = [col for col in X_diag.columns if 'delay_' in col]
        print(f"Available delay features for analysis: {len(delay_features)}")
    """
    logger.info("=" * 60)
    logger.info("CREATING DIAGNOSTICS DATASET (Analysis features with delays)")
    logger.info("=" * 60)
    
    # Build comprehensive base feature set for healthcare analytics
    out = build_base(df)

    # Time reconstruction for comprehensive delay calculation analysis
    logger.info("Reconstructing timestamps from time fragments...")
    tr = TimeReconstructor()
    initial_time_cols = len(out.columns)
    out = tr.transform(out)
    new_time_cols = len(out.columns) - initial_time_cols
    logger.info(f"  Added {new_time_cols} reconstructed timestamp columns")
    logger.info(f"  Shape after time reconstruction: {out.shape}")
    
    # Realign infusion times to TAT targets for accurate analysis
    logger.info("Realigning infusion times to TAT targets...")
    out = tr.realign_infusion_to_tat(out)

    # Calculate step-to-step delays for bottleneck identification analysis
    logger.info("Computing step-to-step delays...")
    de = DelayEngineer()
    initial_delay_cols = len(out.columns)
    out = de.transform(out)
    new_delay_features = len(out.columns) - initial_delay_cols
    logger.info(f"  Added {new_delay_features} delay features")
    logger.info(f"  Shape after delay computation: {out.shape}")

    # Generate temporal features for comprehensive workflow analysis
    logger.info("Generating temporal features...")
    te = TemporalEngineer()
    initial_temporal_cols = len(out.columns)
    out = te.transform(out)
    new_temporal_features = len(out.columns) - initial_temporal_cols
    logger.info(f"  Added {new_temporal_features} temporal features")
    logger.info(f"  Shape after temporal features: {out.shape}")

    # Apply categorical encodings for analytical model compatibility
    logger.info("Applying categorical encodings...")
    encoder = CategoricalEncoder(one_hot_prefix_map=CATEGORICAL_PREFIX_MAP)
    initial_cat_cols = len(out.columns)
    out = encoder.transform(out)
    new_cat_features = len(out.columns) - initial_cat_cols
    logger.info(f"  Added {new_cat_features} categorical features")
    logger.info(f"  Shape after categorical encoding: {out.shape}")
    
    # Create comprehensive target variables for analytical modeling
    out = create_target_variables(out)
    
    # Remove time columns while preserving delay features for analysis
    logger.info("Removing timestamp columns while preserving delay features...")
    time_cols = [col for col in out.columns if any(suffix in col.lower() 
                for suffix in ['_time', '_dt', '_mins_unwrapped'])]
    if time_cols:
        logger.info(f"  Removing {len(time_cols)} timestamp columns")
        logger.debug(f"  Timestamp columns: {time_cols}")
    out = out.drop(columns=time_cols, errors='ignore')
    
    # Count preserved delay features for bottleneck analysis capabilities
    delay_cols = [col for col in out.columns if 'delay_' in col.lower()]
    logger.info(f"  Preserved {len(delay_cols)} delay features for analysis")
    logger.info(f"  Shape after timestamp removal: {out.shape}")
    
    # Split features and targets for analytical workflow preparation
    logger.info("Splitting features and targets...")
    X, y_reg, y_clf = split_features_targets(out)
    logger.info(f"  Feature matrix: {X.shape}")
    logger.info(f"  Regression target: {y_reg.shape}")
    logger.info(f"  Classification target: {y_clf.shape}")
    
    # Apply healthcare-optimized feature scaling for analytical modeling
    X_scaled, scaler_info = scale_numeric_features(X, strategy=scaling_strategy)

    logger.info(f"✓ Diagnostics dataset creation complete. Final feature matrix: {X_scaled.shape}")
    return X_scaled, y_reg, y_clf, scaler_info, scaler_info


class DatasetBuilder:
    """
    Comprehensive dataset creation system for TAT prediction and pharmacy workflow analysis.

    Orchestrates end-to-end dataset preparation from raw healthcare data to machine
    learning-ready features supporting both real-time prediction and analytical research.
    Essential for healthcare analytics providing standardized dataset creation workflows
    for TAT prediction modeling and pharmacy operations optimization in clinical environments.

    Core Responsibilities:
    - F0 dataset creation: Real-time inference features without future information leakage
    - Diagnostics dataset creation: Analytical features including step-to-step delays
    - Feature engineering coordination: Comprehensive transformation pipeline management
    - Data quality assurance: Healthcare-optimized validation and quality enhancement

    Healthcare Analytics Integration:
    - Production deployment: F0 datasets safe for real-time healthcare operations
    - Research analysis: Diagnostics datasets supporting comprehensive workflow optimization
    - Clinical interpretability: Features maintaining healthcare operational context meaning
    - Quality monitoring: Comprehensive validation ensuring clinical data integrity

    Dataset Types:

    F0 (Real-time Inference):
    - Order-time features only preventing future information leakage in production
    - Optimized for real-time TAT prediction supporting healthcare operational decisions
    - Production-safe feature set enabling live pharmacy workflow optimization
    - Efficient computation suitable for healthcare operational time constraints

    Diagnostics (Analytical Research):
    - Comprehensive features including step-to-step delays for bottleneck identification
    - Detailed workflow analysis supporting pharmacy operations research and optimization
    - Process insight capabilities enabling comprehensive medication preparation understanding
    - Quality enhancement research supporting healthcare operations excellence initiatives

    Example:
        For comprehensive healthcare analytics dataset preparation:
        builder = DatasetBuilder()
        
        # Create F0 dataset for production deployment
        f0_data = builder.create_f0(raw_tat_df)
        
        # Create diagnostics dataset for workflow analysis
        diag_data = builder.create_diagnostics(raw_tat_df)

    """
    
    def __init__(self):
        """
        Initialize comprehensive dataset creation system for healthcare analytics.

        Sets up standardized dataset preparation workflows with healthcare-optimized
        configuration supporting TAT prediction modeling and pharmacy operations
        optimization. Provides consistent interface for both production deployment
        and analytical research dataset creation in clinical environments.

        """
        pass
    
    def create_f0(self, df: pd.DataFrame, scaling_strategy: str = "mixed") -> dict:
        """
        Create F0 dataset for real-time TAT prediction deployment.

        Generates production-ready dataset excluding future information ensuring safe
        real-time deployment for medication preparation TAT prediction. Essential for
        healthcare operations supporting live pharmacy workflow optimization while
        maintaining clinical decision-making capabilities and operational efficiency.

        Args:
            df: Raw TAT dataset for F0 creation supporting real-time prediction
               deployment in healthcare operations and pharmacy workflow optimization.
            scaling_strategy: Feature scaling approach optimized for production
                             machine learning models and real-time inference requirements.

        Returns:
            dict: F0 dataset components including features, targets, and metadata
            supporting production deployment and real-time TAT prediction modeling.

        Example:
            For production TAT prediction model deployment:
            builder = DatasetBuilder()
            f0_dataset = builder.create_f0(raw_tat_df, scaling_strategy="mixed")
            
            # Access F0 components for model training and deployment
            X_f0 = f0_dataset['features']
            y_reg = f0_dataset['target_regression']
        """
        X, y_reg, y_clf, scaler_info, removal_info = make_f0(df, scaling_strategy)
        
        return {
            'features': X,
            'target_regression': y_reg,
            'target_classification': y_clf,
            'scaler_info': scaler_info,
            'removal_info': removal_info,
            'dataset_type': 'f0'
        }
    
    def create_diagnostics(self, df: pd.DataFrame, scaling_strategy: str = "mixed") -> dict:
        """
        Create diagnostics dataset for comprehensive workflow analysis.

        Generates analytical dataset including delay features for pharmacy workflow
        bottleneck identification and comprehensive TAT analysis. Essential for
        healthcare operations research supporting pharmacy workflow optimization
        through detailed medication preparation process understanding and improvement.

        Args:
            df: Raw TAT dataset for diagnostics creation supporting comprehensive
               workflow analysis and pharmacy operations optimization research.
            scaling_strategy: Feature scaling approach optimized for analytical
                             modeling and bottleneck identification requirements.

        Returns:
            dict: Diagnostics dataset components including features, targets, and
            delay information supporting comprehensive workflow analysis and research.

        Example:
            For comprehensive pharmacy workflow bottleneck analysis:
            builder = DatasetBuilder()
            diag_dataset = builder.create_diagnostics(raw_tat_df, scaling_strategy="mixed")
            
            # Access diagnostics components for workflow analysis
            X_diag = diag_dataset['features']
            delay_features = [col for col in X_diag.columns if 'delay_' in col]
        """
        X, y_reg, y_clf, scaler_info, removal_info = make_diagnostics(df, scaling_strategy)
        
        return {
            'features': X,
            'target_regression': y_reg,
            'target_classification': y_clf,
            'scaler_info': scaler_info,
            'removal_info': removal_info,
            'dataset_type': 'diagnostics'
        }