"""
Categorical feature encoding for pharmacy workflow analysis.

Provides one-hot and ordinal encoding for categorical variables in 
medication preparation datasets.

Technical Features:
- Robust categorical mapping with missing value handling for healthcare data quality
- Configurable one-hot encoding supporting diverse machine learning model requirements
- Pipeline architecture enabling custom categorical transformations and extensions
- Integration with automated TAT analytics and pharmacy operations monitoring systems
"""

import pandas as pd
from typing import Iterable, Callable, Dict, Optional, List

from tat.config import CATEGORICAL_MAPPINGS

# Type alias for categorical encoder functions in healthcare feature engineering workflows
Encoder = Callable[[pd.DataFrame], pd.DataFrame]

class CategoricalEncoder:
    """
    Comprehensive categorical encoding system for pharmacy TAT workflow analysis.

    Orchestrates end-to-end categorical feature engineering from medication preparation
    operational variables with healthcare-optimized encoding strategies. Designed for
    production deployment in healthcare pharmacy analytics environment supporting
    TAT prediction modeling, workflow optimization, and comprehensive bottleneck analysis.

    Core Responsibilities:
    - Transform clinical variables using healthcare domain-aware ordinal mappings
    - Apply operational context encoding for staffing and resource allocation analysis
    - Generate one-hot encoded features with clinical interpretability preservation
    - Provide extensible pipeline architecture for specialized pharmacy operations requirements

    Feature Engineering Capabilities:
    - Ordinal encoding for clinical variables maintaining healthcare operational hierarchy
    - One-hot encoding with domain-aware prefixes for machine learning model training
    - Custom encoder registration supporting specialized Healthcare requirements
    - Pipeline architecture enabling reproducible categorical feature engineering workflows

    Args:
        one_hot_prefix_map: Custom prefix mapping for one-hot encoded categorical features.
                           Supports domain-aware feature naming for clinical interpretability.
        one_hot_drop_first: Enable reference category dropping for regression modeling.
                           Default False preserves all categories for comprehensive analysis.
        dtype: Data type for generated categorical features supporting model requirements.
              Default int provides efficient encoding for healthcare analytics workflows.

    Example:
        # Standard categorical encoding for Healthcare TAT prediction modeling
        encoder = CategoricalEncoder()
        encoded_df = encoder.transform(tat_df)
        
        # Custom encoding pipeline with specialized healthcare requirements
        custom_encoder = CategoricalEncoder(
            one_hot_prefix_map={'diagnosis_type': 'dx', 'treatment_type': 'tx'},
            one_hot_drop_first=True
        )
        encoded_df = custom_encoder.fit_transform(tat_df)

    Note:
        Designed for production deployment in healthcare healthcare analytics environment
        with comprehensive categorical encoding supporting TAT prediction modeling and
        pharmacy workflow optimization initiatives in clinical operations environments.
    """

    def __init__(
        self,
        one_hot_prefix_map: Optional[Dict[str, str]] = None,
        one_hot_drop_first: bool = False,
        dtype: type = int,
    ):
        """
        Initialize categorical encoding system with healthcare-optimized configuration.

        Sets up comprehensive categorical feature engineering pipeline with appropriate
        defaults for healthcare medication preparation workflow analysis and pharmacy
        operations optimization. Configures ordinal mappings, one-hot encoding strategies,
        and custom encoder integration for robust healthcare analytics processing.

        Args:
            one_hot_prefix_map: Optional custom prefix mapping for one-hot encoded features
                              supporting domain-aware feature naming and clinical interpretability.
                              Defaults to column names preserving healthcare operational context.
            one_hot_drop_first: Enable reference category dropping for regression modeling
                              requirements. Default False preserves all categories for
                              comprehensive Healthcare pharmacy workflow analysis.
            dtype: Data type for generated categorical features supporting downstream
                  machine learning model training and healthcare analytics requirements.
                  Default int provides efficient encoding for TAT prediction workflows.

        Note:
            Default configuration optimized for healthcare standard pharmacy operations
            with healthcare domain knowledge embedded in ordinal mappings and encoding
            strategies. Custom configurations support specialized analytical requirements.
        """
        # Configure one-hot encoding parameters for machine learning model compatibility
        self.one_hot_prefix_map = one_hot_prefix_map or {}
        self.one_hot_drop_first = one_hot_drop_first
        self.dtype = dtype

        # Initialize healthcare domain-aware ordinal mappings for clinical variables
        # These mappings convert clinical categories to ordinal codes preserving operational hierarchy
        self.sex_map = CATEGORICAL_MAPPINGS['sex']
        self.severity_map = CATEGORICAL_MAPPINGS['severity']
        self.nurse_credential_map = CATEGORICAL_MAPPINGS['nurse_credential']
        self.pharmacist_credential_map = CATEGORICAL_MAPPINGS['pharmacist_credential']

        # Initialize custom encoder pipeline for specialized Healthcare requirements
        self._custom_encoders: List[Encoder] = []

    def one_hot(self, df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
        """
        Generate one-hot encoded features with healthcare domain-aware configuration.

        Transforms specified categorical columns into binary indicator features using
        pandas get_dummies with Healthcare-optimized settings. Essential for machine
        learning model training supporting TAT prediction and pharmacy workflow optimization
        while maintaining clinical interpretability and operational context awareness.

        Healthcare Feature Engineering Benefits:
        - Binary indicators supporting diverse machine learning algorithms for TAT prediction
        - Domain-aware prefix mapping enabling clinical interpretability and stakeholder communication
        - Configurable reference category handling for regression modeling requirements
        - Robust missing value handling suitable for healthcare data quality patterns

        Args:
            df: Input TAT dataset containing categorical variables for one-hot encoding
               transformation supporting Healthcare pharmacy workflow analysis requirements.
            cols: Categorical column names for binary indicator feature generation
                 supporting machine learning model training and healthcare analytics workflows.

        Returns:
            pd.DataFrame: Enhanced dataset with one-hot encoded binary features suitable for:
            - TAT prediction modeling and machine learning algorithm training
            - Pharmacy workflow optimization and bottleneck identification analysis
            - Healthcare operations analytics and clinical decision-making support
            - Integration with automated Healthcare analytics and monitoring systems

        Example:
            For Healthcare categorical feature engineering supporting TAT prediction:
            encoder = CategoricalEncoder(
                one_hot_prefix_map={'diagnosis_type': 'dx', 'treatment_type': 'tx'}
            )
            encoded_df = encoder.one_hot(tat_df, ['diagnosis_type', 'treatment_type'])
            
            # Generated features: dx_SolidTumor, dx_Hematologic, tx_Chemotherapy, etc.

        Note:
            Essential method for machine learning model preparation in healthcare healthcare
            analytics environment providing binary categorical features with clinical
            interpretability for TAT prediction modeling and workflow optimization analysis.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Filter for columns actually present in dataset for robust processing
        cols = [c for c in cols if c in df.columns]
        if not cols:
            # Return unchanged dataset when no target columns present
            return df
        
        # Configure domain-aware prefixes for clinical interpretability
        prefix = {c: self.one_hot_prefix_map.get(c, c) for c in cols}
        
        # Generate one-hot encoded features with healthcare-optimized configuration
        return pd.get_dummies(
            df,
            columns=cols,
            prefix=prefix,
            drop_first=self.one_hot_drop_first,
            dtype=self.dtype,
        )

    def encode_sex(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform patient sex variable using healthcare demographic encoding.

        Converts patient sex categories to ordinal codes suitable for healthcare
        healthcare analytics and TAT prediction modeling. Maintains clinical demographic
        context while providing numeric encoding for machine learning algorithms and
        comprehensive pharmacy workflow optimization analysis workflows.

        Healthcare Demographic Encoding:
        - Preserves clinical demographic information for patient population analysis
        - Supports healthcare analytics requiring patient characteristic integration
        - Enables demographic pattern recognition in TAT prediction modeling workflows
        - Maintains operational context for pharmacy workflow optimization initiatives

        Args:
            df: Input TAT dataset containing patient sex variable for healthcare
               demographic encoding supporting Healthcare analytics requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with encoded sex variable suitable for
            Healthcare TAT prediction modeling and pharmacy workflow optimization
            while preserving clinical demographic context and operational meaning.

        Example:
            For Healthcare patient demographic feature engineering:
            encoder = CategoricalEncoder()
            encoded_df = encoder.encode_sex(tat_df)
            # Transforms: 'F' -> 0, 'M' -> 1 maintaining clinical demographic context

        Note:
            Preserves healthcare demographic information while enabling numeric analysis
            essential for healthcare TAT prediction modeling and comprehensive
            pharmacy workflow optimization supporting clinical decision-making processes.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Apply healthcare demographic encoding when sex column present
        if "sex" in df.columns:
            df["sex"] = df["sex"].map(self.sex_map)
        
        return df

    def encode_severity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform clinical severity variable using healthcare priority encoding.

        Converts patient severity categories to ordinal codes preserving clinical priority
        hierarchy essential for healthcare TAT prediction modeling and pharmacy workflow
        optimization. Maintains healthcare operational context supporting treatment priority
        analysis and comprehensive bottleneck identification in medication preparation workflows.

        Clinical Severity Hierarchy:
        - Low severity: Routine medication preparation with standard TAT expectations
        - Medium severity: Elevated priority requiring enhanced workflow monitoring
        - High severity: Critical priority demanding immediate attention and resource allocation

        Args:
            df: Input TAT dataset containing clinical severity variable for healthcare
               priority encoding supporting Healthcare pharmacy analytics requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with encoded severity variable maintaining
            clinical priority hierarchy for Healthcare TAT prediction modeling and
            pharmacy workflow optimization supporting healthcare operational excellence.

        Example:
            For Healthcare clinical priority feature engineering:
            encoder = CategoricalEncoder()
            encoded_df = encoder.encode_severity(tat_df)
            # Transforms: 'Low' -> 1, 'Medium' -> 2, 'High' -> 3 preserving clinical hierarchy

        Note:
            Essential for maintaining clinical priority context in healthcare TAT
            prediction modeling while enabling ordinal analysis supporting pharmacy
            workflow optimization and healthcare quality monitoring initiatives.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Apply clinical priority encoding when severity column present
        if "severity" in df.columns:
            df["severity"] = df["severity"].map(self.severity_map)
        
        return df

    def encode_credentials(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform healthcare credential variables using professional hierarchy encoding.

        Converts nursing and pharmacy credential categories to ordinal codes preserving
        professional hierarchy essential for healthcare staffing analysis and TAT
        prediction modeling. Maintains healthcare operational context supporting resource
        allocation optimization and comprehensive pharmacy workflow bottleneck identification.

        Healthcare Credential Hierarchies:
        
        Nursing Credentials (ascending competency):
        - RN: Registered Nurse (core bedside care and medication administration)
        - BSN: Bachelor of Science in Nursing (enhanced clinical and leadership preparation)
        - MSN: Master of Science in Nursing (advanced clinical specialization and leadership)
        - NP: Nurse Practitioner (advanced practice with assessment and treatment authority)

        Pharmacy Credentials (ascending specialization):
        - RPh: Registered Pharmacist (foundational clinical practice and medication management)
        - PharmD: Doctor of Pharmacy (enhanced clinical training and direct patient care)
        - BCOP: Board Certified Clinical Pharmacist (specialized healthcare expertise)

        Staffing Optimization Applications:
        - Resource allocation analysis considering professional competency levels
        - TAT prediction modeling incorporating staffing expertise and workflow efficiency
        - Pharmacy operations optimization through credential-based performance analysis
        - Healthcare quality monitoring with competency-stratified outcome assessment

        Args:
            df: Input TAT dataset containing healthcare credential variables for professional
               hierarchy encoding supporting Healthcare staffing analytics requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with encoded credential variables maintaining
            professional hierarchy for Healthcare TAT prediction modeling and staffing
            optimization supporting healthcare operational excellence and quality improvement.

        Example:
            For Healthcare staffing optimization feature engineering:
            encoder = CategoricalEncoder()
            encoded_df = encoder.encode_credentials(tat_df)
            # Nursing: 'RN' -> 1, 'BSN' -> 2, 'MSN' -> 3, 'NP' -> 4
            # Pharmacy: 'RPh' -> 1, 'PharmD' -> 2, 'BCOP' -> 3

        Note:
            Critical for maintaining professional hierarchy context in healthcare
            staffing analysis while enabling ordinal competency assessment supporting
            pharmacy workflow optimization and healthcare resource allocation initiatives.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Apply nursing credential hierarchy encoding when column present
        if "nurse_credential" in df.columns:
            df["nurse_credential"] = df["nurse_credential"].map(self.nurse_credential_map)
        
        # Apply pharmacy credential hierarchy encoding when column present
        if "pharmacist_credential" in df.columns:
            df["pharmacist_credential"] = df["pharmacist_credential"].map(self.pharmacist_credential_map)
        
        return df

    def register(self, encoder: Encoder) -> None:
        """
        Register custom categorical encoder for specialized Healthcare requirements.

        Extends categorical encoding pipeline with custom transformation functions supporting
        specialized healthcare analytics requirements and domain-specific categorical
        feature engineering. Essential for implementing Healthcare-specific categorical
        transformations and advanced pharmacy operations optimization analysis workflows.

        Custom Encoder Integration:
        - Executes after standard categorical encoding pipeline for comprehensive processing
        - Supports specialized Healthcare categorical pattern recognition and analysis
        - Enables custom clinical variable transformations for advanced TAT prediction modeling
        - Maintains pipeline architecture for reproducible categorical feature engineering

        Healthcare Applications:
        - Healthcare-specific clinical variable transformations for specialized analysis
        - Custom treatment protocol categorical features for advanced TAT prediction modeling
        - Specialized pharmacy operations categorical patterns for workflow optimization
        - Healthcare regulatory categorical features for compliance monitoring and reporting

        Args:
            encoder: Custom transformation function accepting DataFrame and returning enhanced
                    DataFrame with additional categorical features for Healthcare analytics.

        Example:
            For specialized Healthcare categorical feature engineering:
            def encode_treatment_complexity(df):
                # Custom treatment complexity categorical encoding
                df['treatment_complexity'] = calculate_complexity_score(df)
                return df
            
            encoder = CategoricalEncoder()
            encoder.register(encode_treatment_complexity)
            enhanced_df = encoder.transform(tat_df)

        Note:
            Custom encoders execute after standard categorical pipeline enabling sophisticated
            Healthcare healthcare analytics workflows with specialized categorical feature
            engineering supporting advanced TAT prediction modeling and workflow optimization.
        """
        # Add custom encoder to pipeline for execution after standard categorical processing
        self._custom_encoders.append(encoder)

    def clear_custom(self) -> None:
        """
        Remove all registered custom categorical encoders for pipeline reset.

        Provides categorical encoding pipeline reset capability for iterative feature
        engineering development and testing workflows. Essential for maintaining clean
        pipeline configuration during Healthcare analytics development and ensuring
        reproducible categorical feature extraction in production TAT prediction environments.

        Pipeline Management Benefits:
        - Clears custom categorical processing while preserving standard healthcare encodings
        - Enables iterative development of specialized Healthcare categorical workflows
        - Supports A/B testing of custom categorical features in TAT prediction validation
        - Maintains pipeline architecture integrity for consistent categorical processing

        Development Workflow Support:
        - Facilitates experimentation with custom Healthcare categorical transformations
        - Enables clean pipeline reset between different categorical encoding approaches
        - Supports reproducible categorical feature extraction for model validation and testing
        - Provides flexible configuration management for diverse Healthcare analytics requirements

        Example:
            For iterative Healthcare categorical feature engineering development:
            encoder = CategoricalEncoder()
            encoder.register(custom_treatment_encoder)
            encoder.register(custom_protocol_encoder)
            
            # Reset custom encoders for clean pipeline testing
            encoder.clear_custom()
            baseline_features = encoder.transform(tat_df)

        Note:
            Preserves standard healthcare categorical pipeline while removing custom extensions
            enabling controlled categorical feature engineering experimentation in Healthcare
            analytics development and TAT prediction modeling validation workflows.
        """
        # Clear all custom categorical processing while preserving standard pipeline
        self._custom_encoders.clear()

    def apply(self, df: pd.DataFrame, sequence: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Execute categorical encoding pipeline with configurable processing sequence.

        Orchestrates comprehensive categorical feature engineering workflow with flexible
        processing sequence configuration supporting diverse Healthcare healthcare analytics
        requirements and specialized TAT prediction modeling scenarios. Provides fine-grained
        control over categorical encoding for advanced pharmacy workflow optimization analysis.

        Pipeline Execution Strategy:
        - Executes specified sequence or default categorical encoding steps in configured order
        - Applies custom registered encoders after main processing sequence completion
        - Maintains non-destructive operations for robust Healthcare healthcare analytics
        - Supports iterative categorical feature engineering development and validation workflows

        Advanced Configuration Support:
        - Custom processing sequence for specialized categorical feature requirements
        - Integration with automated Healthcare healthcare analytics and model training pipelines
        - Flexible workflow configuration for diverse clinical operational categorical patterns
        - Comprehensive categorical encoding for complex TAT prediction and workflow scenarios

        Args:
            df: Input TAT dataset containing categorical variables for comprehensive encoding
               transformation supporting Healthcare pharmacy workflow analysis requirements.
            sequence: Optional custom processing sequence replacing default categorical encoding
                     steps for specialized healthcare analytics workflows and requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with comprehensive categorical features from
            configured processing sequence and custom encoders for advanced Healthcare
            healthcare analytics and TAT prediction modeling integration workflows.

        Example:
            For custom Healthcare categorical encoding sequence in TAT prediction workflow:
            custom_sequence = [
                encoder.encode_severity,
                encoder.encode_credentials,
                custom_clinical_encoder
            ]
            
            encoded_df = encoder.apply(tat_df, sequence=custom_sequence)

        Note:
            Advanced method supporting sophisticated categorical feature engineering workflows
            with custom processing sequences for specialized Healthcare healthcare analytics
            requirements and complex TAT prediction modeling in clinical operations environments.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Execute specified sequence or default categorical encoding steps
        if sequence is None:
            # Default healthcare categorical encoding sequence for Healthcare analytics
            sequence = [self.encode_sex, self.encode_severity, self.encode_credentials]

        # Apply each categorical encoding step in configured sequence
        for step in sequence:
            if isinstance(step, str):
                # Execute method by name for string-based step specification
                fn = getattr(self, step)
                df = fn(df)
            elif callable(step):
                # Execute callable step directly for custom function integration
                df = step(df)
            else:
                raise TypeError("Sequence elements must be method names or callables")

        # Apply custom registered encoders after main categorical processing sequence
        for enc in self._custom_encoders:
            df = enc(df)

        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute comprehensive categorical encoding using default pipeline configuration.

        Applies complete categorical feature engineering workflow using standard Healthcare
        configuration suitable for routine medication preparation TAT analysis and pharmacy
        workflow optimization. Provides streamlined interface for categorical encoding in
        healthcare analytics and TAT prediction modeling supporting clinical operations.

        Standard Processing Workflow:
        1. Convert numeric floor variable to categorical for proper one-hot encoding
        2. Apply ordinal encoding for configured clinical and operational variables
        3. Generate one-hot encoded features for remaining categorical variables
        4. Execute custom registered encoders for specialized Healthcare requirements

        Args:
            df: Input TAT dataset containing categorical variables for standard encoding
               transformation supporting Healthcare pharmacy workflow analysis workflows.

        Returns:
            pd.DataFrame: Enhanced dataset with comprehensive categorical features suitable
            for TAT prediction modeling, pharmacy workflow optimization, and Healthcare
            healthcare operations analytics supporting clinical decision-making processes.

        Example:
            For standard Healthcare categorical encoding in TAT prediction workflow:
            encoder = CategoricalEncoder()
            encoded_df = encoder.transform(tat_df)
            
            # Validate categorical encoding results and feature generation
            categorical_cols = [col for col in encoded_df.columns if '_' in col]
            print(f"Generated {len(categorical_cols)} categorical features for TAT analysis")

        Note:
            Primary method for categorical encoding in Healthcare healthcare analytics workflows
            providing comprehensive categorical features for TAT prediction modeling and
            pharmacy operations optimization in clinical environments and quality monitoring.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Convert numeric floor variable to categorical for proper one-hot encoding
        if 'floor' in df.columns:
            df['floor'] = df['floor'].astype('category')

        # Apply ordinal encoding for configured clinical and operational variables
        df = self.apply(df)
        
        # Identify remaining categorical variables for one-hot encoding transformation
        cols_to_onehot = [
            col for col in df.columns 
            if col in self.one_hot_prefix_map and 
            (pd.api.types.is_object_dtype(df[col]) or 
             isinstance(df[col].dtype, pd.CategoricalDtype))
        ]
        
        # Generate one-hot encoded features for machine learning model training
        if cols_to_onehot:
            df = self.one_hot(df, cols_to_onehot)
        
        return df

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        """
        Scikit-learn compatible fit method for estimator-like interface consistency.

        No-operation method maintained for compatibility with sklearn-style pipelines
        and automated machine learning workflows in healthcare healthcare analytics
        environment. CategoricalEncoder transforms categorical variables statically
        without requiring training or parameter estimation from input data patterns.

        Args:
            df: TAT dataset for interface consistency (not used in categorical processing)

        Returns:
            CategoricalEncoder: Self reference for method chaining compatibility with
            sklearn pipeline patterns and automated Healthcare healthcare analytics workflows.

        Note:
            Maintained for sklearn pipeline compatibility in healthcare healthcare analytics
            environment. Categorical encoding operates statically without requiring data-driven
            parameter estimation for consistent Healthcare processing behavior.
        """
        # No-operation method for sklearn transformer interface compatibility
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method combining fit and transform for sklearn-style workflow integration.

        Provides streamlined interface for categorical encoding in automated Healthcare
        healthcare analytics pipelines and machine learning workflows. Equivalent to calling
        fit() followed by transform() while maintaining method chaining compatibility for
        comprehensive feature engineering pipeline construction and TAT analysis workflows.

        Args:
            df: Input TAT dataset for comprehensive categorical encoding using healthcare-
               optimized processing supporting Healthcare pharmacy workflow analysis requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with categorical features ready for Healthcare
            TAT prediction modeling, pharmacy workflow optimization, and healthcare operations
            analytics supporting clinical decision-making and quality monitoring initiatives.

        Example:
            For streamlined Healthcare categorical encoding in automated analytics pipelines:
            encoder = CategoricalEncoder()
            encoded_df = encoder.fit_transform(tat_df)
            
            # Integration with sklearn pipeline for automated Healthcare model training
            from sklearn.pipeline import Pipeline
            tat_pipeline = Pipeline([
                ('categorical', encoder),
                ('model', xgb_regressor)
            ])
        """
        # Execute fit and transform in sequence for sklearn-style workflow integration
        return self.fit(df).transform(df)

    @classmethod
    def default(cls) -> "CategoricalEncoder":
        """
        Create CategoricalEncoder instance with Healthcare healthcare-optimized configuration.

        Factory method providing pre-configured categorical encoding system optimized for
        standard Healthcare medication preparation TAT analysis and pharmacy workflow
        optimization. Eliminates configuration overhead while ensuring appropriate defaults
        for healthcare analytics and clinical operations environments supporting quality monitoring.

        Returns:
            CategoricalEncoder: Configured instance with Healthcare healthcare-optimized defaults
            for standard medication preparation TAT analysis and pharmacy workflow optimization
            supporting clinical decision-making and operational excellence initiatives.

        Example:
            For rapid Healthcare categorical encoding in healthcare analytics workflows:
            # Quick setup for standard TAT analysis
            encoder = CategoricalEncoder.default()
            categorical_features = encoder.transform(tat_df)
            
            # Equivalent to standard initialization with defaults
            standard_encoder = CategoricalEncoder()
            same_features = standard_encoder.transform(tat_df)

        Note:
            Recommended approach for standard Healthcare healthcare analytics workflows requiring
            comprehensive categorical encoding without custom configuration needs supporting
            TAT prediction modeling and pharmacy operations optimization in clinical environments.
        """
        # Create instance with Healthcare healthcare-optimized default configuration
        return cls()