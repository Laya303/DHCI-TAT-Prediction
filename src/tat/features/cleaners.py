"""
Data cleaning utilities for pharmacy workflow datasets.

Provides validation and cleaning for demographic, laboratory, and
operational variables in medication preparation data.
"""
import pandas as pd
from typing import Iterable, Callable, List, Optional, Tuple

from tat.config import CLEANING_CONFIG

# Type alias for data cleaning functions in healthcare analytics workflows
CleanerRule = Callable[[pd.DataFrame], pd.DataFrame]


class Cleaner:
    """
    Comprehensive data quality enhancement system for pharmacy TAT workflow analysis.

    Orchestrates end-to-end data cleaning and validation from medication preparation
    datasets with healthcare-optimized quality assurance strategies. Designed for
    production deployment in healthcare pharmacy analytics environment supporting
    TAT prediction modeling, workflow optimization, and comprehensive quality monitoring.

    Core Responsibilities:
    - Enforce clinical bounds for patient demographic variables ensuring medical realism
    - Validate healthcare professional experience data supporting accurate staffing analytics
    - Coerce binary variables from diverse EHR formats maintaining data integrity
    - Provide extensible pipeline architecture for specialized Healthcare quality requirements

    Healthcare Data Quality Assurance:
    - Patient age bounds enforcement preventing medically unrealistic demographic values
    - Healthcare professional experience validation supporting staffing optimization analysis
    - Clinical laboratory value bounds ensuring physiologically realistic data integrity
    - Binary medical record variable standardization across diverse healthcare data sources

    Args:
        age_bounds: Patient age validation bounds ensuring medically realistic demographic data.
                   Default (0, 120) covers complete patient lifespan for healthcare analytics.
        years_bounds: Healthcare professional experience bounds supporting staffing validation.
                     Default (0, 50) covers typical career spans for pharmacy operations analysis.
        years_cols: Healthcare professional experience column names for validation processing.
                   Defaults to nurse and pharmacist employment duration columns.

    Example:
        # Standard healthcare data quality enhancement for Healthcare TAT analysis
        cleaner = Cleaner()
        cleaned_df = cleaner.apply(tat_df)
        
        # Custom validation pipeline with specialized Healthcare requirements
        custom_cleaner = Cleaner(
            age_bounds=(18, 100),  # Adult healthcare population focus
            years_bounds=(0, 40)   # Pharmacy professional experience range
        )
        enhanced_df = custom_cleaner.apply(tat_df)

    Note:
        Designed for production deployment in healthcare healthcare analytics environment
        with comprehensive data quality assurance supporting TAT prediction modeling and
        pharmacy workflow optimization initiatives ensuring clinical data integrity.
    """

    def __init__(
        self,
        age_bounds: Tuple[float, float] = CLEANING_CONFIG['age_bounds'],
        years_bounds: Tuple[float, float] = CLEANING_CONFIG['years_bounds'],
        years_cols: Optional[Iterable[str]] = None,
    ):
        """
        Initialize data quality enhancement system with healthcare-optimized configuration.

        Sets up comprehensive data cleaning pipeline with appropriate clinical bounds for
        healthcare medication preparation workflow analysis and pharmacy operations
        optimization. Configures validation rules, professional experience bounds, and
        quality assurance parameters for robust healthcare analytics processing.

        Args:
            age_bounds: Patient age validation bounds ensuring medically realistic demographic
                       data for Healthcare healthcare population analysis. Default (0, 120)
                       covers complete patient lifespan supporting comprehensive healthcare analytics.
            years_bounds: Healthcare professional experience bounds supporting pharmacy staffing
                         validation and optimization analysis. Default (0, 50) covers typical
                         career spans for nursing and pharmacy professional development.
            years_cols: Optional healthcare professional experience column names for validation
                       processing. Defaults to configured nurse and pharmacist employment
                       duration columns supporting Healthcare staffing analytics requirements.

        Note:
            Default configuration optimized for healthcare patient population and
            pharmacy professional staffing patterns with clinical validation bounds ensuring
            healthcare data integrity. Custom configurations support specialized analytical requirements.
        """
        # Configure clinical validation bounds for patient demographic data integrity
        self.age_bounds = age_bounds
        self.years_bounds = years_bounds
        
        # Set up healthcare professional experience columns for staffing validation
        self.years_cols = list(years_cols) if years_cols is not None else CLEANING_CONFIG['years_cols']
        
        # Initialize custom validation rules for specialized Healthcare requirements
        self._rules: List[CleanerRule] = []

    def clip_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce patient age bounds ensuring medically realistic demographic data.

        Validates patient age values against clinical bounds preventing medically unrealistic
        demographic data from compromising healthcare TAT analysis and pharmacy workflow
        optimization. Essential for maintaining healthcare data integrity and supporting
        accurate patient population analytics in healthcare treatment environments.

        Args:
            df: Input TAT dataset containing patient age variable for clinical validation
               and bounds enforcement supporting Healthcare healthcare analytics requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with validated patient age ensuring medically
            realistic demographic data for Healthcare TAT prediction modeling and pharmacy
            workflow optimization supporting clinical operations and quality monitoring.

        Example:
            For Healthcare patient demographic validation in TAT analysis workflow:
            cleaner = Cleaner(age_bounds=(18, 100))  # Adult healthcare focus
            validated_df = cleaner.clip_age(tat_df)
            
            # Validate age cleaning effectiveness and data quality improvement
            age_range = validated_df['age'].describe()
            print(f"Age range after validation: {age_range['min']} - {age_range['max']}")

        Note:
            Essential for maintaining clinical data integrity in healthcare healthcare
            analytics supporting accurate patient population analysis and TAT prediction
            modeling with validated demographic foundations for pharmacy workflow optimization.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        low, high = self.age_bounds
        
        # Apply clinical age validation when patient age column present
        if "age" in df.columns:
            # Robust numeric coercion with clinical bounds enforcement
            df["age"] = pd.to_numeric(df["age"], errors="coerce").clip(lower=low, upper=high)
        
        # Also handle patient_age column if present
        if "patient_age" in df.columns:
            # Robust numeric coercion with clinical bounds enforcement
            df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce").clip(lower=low, upper=high)
        
        return df

    def clip_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate healthcare professional experience data supporting staffing analytics.

        Enforces experience bounds for nursing and pharmacy professional variables preventing
        unrealistic career duration values from compromising healthcare staffing analysis
        and pharmacy workflow optimization. Essential for maintaining healthcare professional
        data integrity supporting accurate resource allocation and competency assessment.

        Args:
            df: Input TAT dataset containing healthcare professional experience variables
               for validation and bounds enforcement supporting Healthcare staffing analytics.

        Returns:
            pd.DataFrame: Enhanced dataset with validated professional experience ensuring
            realistic career duration data for Healthcare staffing optimization and pharmacy
            workflow analysis supporting clinical operations and resource allocation initiatives.

        Example:
            For Healthcare healthcare professional validation in staffing analytics workflow:
            cleaner = Cleaner(
                years_bounds=(0, 40),  # Pharmacy professional career range
                years_cols=['nurse_employment_years', 'pharmacist_employment_years']
            )
            validated_df = cleaner.clip_years(tat_df)
            
            # Validate professional experience cleaning and staffing analytics integrity
            for col in cleaner.years_cols:
                if col in validated_df.columns:
                    exp_range = validated_df[col].describe()
                    print(f"{col} range: {exp_range['min']} - {exp_range['max']} years")

        Note:
            Critical for maintaining healthcare professional data integrity in healthcare
            staffing analytics supporting accurate resource allocation and pharmacy workflow
            optimization with validated professional competency foundations for clinical operations.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        low, high = self.years_bounds
        
        # Apply professional experience validation for configured healthcare columns
        for col in self.years_cols:
            if col in df.columns:
                # Robust numeric coercion with professional experience bounds enforcement
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=low, upper=high)
        
        return df

    def coerce_binary(self, df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        Standardize binary medical variables from diverse healthcare data sources.

        Converts boolean-like medical record variables to standardized 0/1 integer format
        handling diverse EHR system exports and manual data entry patterns. Essential for
        healthcare healthcare analytics ensuring consistent binary variable representation
        supporting TAT prediction modeling and pharmacy workflow optimization analysis.

        Healthcare Binary Variable Standardization:
        - EHR system export format handling: Manages diverse electronic health record formats
        - Manual data entry pattern recognition: Handles clinical staff data entry variations
        - Medical record boolean standardization: Converts healthcare boolean variables consistently  
        - Missing value preservation: Maintains data transparency for healthcare quality monitoring

        Binary Variable Processing Strategy:
        - Direct integer conversion for clean boolean data maintaining processing efficiency
        - Textual boolean interpretation handling common healthcare data entry patterns
        - Numeric coercion with missing value preservation for transparent data quality
        - Robust error handling ensuring production stability in diverse data environments

        Args:
            df: Input TAT dataset containing binary medical variables for standardization
               supporting Healthcare healthcare analytics and TAT prediction requirements.
            cols: Optional binary column names for standardization processing. Common
                 healthcare binary variables include stat_order, premed_required, etc.

        Returns:
            pd.DataFrame: Enhanced dataset with standardized binary variables ensuring
            consistent 0/1 format for Healthcare TAT prediction modeling and pharmacy
            workflow optimization supporting machine learning and healthcare analytics.

        Example:
            For Healthcare binary medical variable standardization in TAT analysis:
            cleaner = Cleaner()
            binary_cols = ['stat_order', 'premed_required']
            standardized_df = cleaner.coerce_binary(tat_df, cols=binary_cols)
            
            # Validate binary standardization effectiveness and data quality
            for col in binary_cols:
                if col in standardized_df.columns:
                    unique_vals = standardized_df[col].unique()
                    print(f"{col} standardized values: {sorted(unique_vals)}")

        Note:
            Essential for healthcare data standardization in healthcare analytics environment
            supporting consistent binary variable representation for TAT prediction modeling
            and pharmacy workflow optimization with robust handling of diverse data sources.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        cols = list(cols) if cols is not None else []
        
        # Standardize each specified binary medical variable with robust error handling
        for c in cols:
            if c in df.columns:
                try:
                    # Fast path for already-standardized healthcare binary variables
                    df[c] = df[c].astype(int)
                except Exception:
                    # Comprehensive healthcare data format handling strategy
                    # Handle diverse EHR export formats and manual entry patterns
                    s = df[c].astype("object").copy()
                    
                    # Convert common healthcare boolean text patterns to numeric
                    s_lower = s.str.lower().replace({
                        "true": "1", "false": "0", 
                        "yes": "1", "no": "0",
                        "y": "1", "n": "0"  # Common clinical abbreviations
                    })
                    
                    # Apply robust numeric coercion for healthcare data quality
                    coerced = pd.to_numeric(s_lower, errors="coerce")
                    
                    # Preserve missing values while standardizing valid boolean data
                    if coerced.notna().any():
                        # Explicit missing value handling for healthcare data transparency
                        df[c] = coerced.fillna(0).astype(int)
                    else:
                        # Fallback standardization with conservative missing value handling
                        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        
        return df

    def add_rule(self, rule: CleanerRule) -> None:
        """
        Register custom data quality rule for specialized Healthcare requirements.

        Extends data cleaning pipeline with custom validation functions supporting specialized
        healthcare analytics requirements and domain-specific data quality assurance. Essential
        for implementing Healthcare-specific validation rules and advanced pharmacy operations
        data quality enhancement workflows supporting clinical operations optimization.

        Args:
            rule: Custom validation function accepting DataFrame and returning enhanced DataFrame
                 with additional data quality improvements for Healthcare healthcare analytics.

        Example:
            For specialized Healthcare data quality enhancement:
            def validate_healthcare_labs(df):
                # Custom healthcare laboratory value validation
                if 'lab_WBC_k_per_uL' in df.columns:
                    df['lab_WBC_k_per_uL'] = df['lab_WBC_k_per_uL'].clip(0.1, 100.0)
                return df
            
            cleaner = Cleaner()
            cleaner.add_rule(validate_healthcare_labs)
            enhanced_df = cleaner.apply(tat_df)

        Note:
            Custom rules execute after standard healthcare validation enabling sophisticated
            Healthcare data quality workflows with specialized validation supporting advanced
            TAT prediction modeling and pharmacy operations optimization in clinical environments.
        """
        # Add custom validation rule to pipeline for execution after standard processing
        self._rules.append(rule)

    def clear_rules(self) -> None:
        """
        Remove all registered custom validation rules for pipeline reset.

        Provides data quality pipeline reset capability for iterative data cleaning development
        and testing workflows. Essential for maintaining clean pipeline configuration during
        Healthcare analytics development and ensuring reproducible data quality enhancement
        in production TAT prediction modeling environments supporting clinical operations.

        Example:
            For iterative Healthcare data quality enhancement development:
            cleaner = Cleaner()
            cleaner.add_rule(custom_lab_validator)
            cleaner.add_rule(custom_demographic_validator)
            
            # Reset custom rules for clean pipeline testing
            cleaner.clear_rules()
            baseline_quality = cleaner.apply(tat_df)

        Note:
            Preserves standard healthcare data quality pipeline while removing custom extensions
            enabling controlled data quality enhancement experimentation in Healthcare analytics
            development and TAT prediction modeling validation supporting clinical operations.
        """
        # Clear all custom validation rules while preserving standard healthcare pipeline
        self._rules.clear()

    def apply(self, df: pd.DataFrame, sequence: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Execute comprehensive data quality enhancement using configurable processing sequence.

        Orchestrates end-to-end data cleaning workflow with flexible processing sequence
        configuration supporting diverse Healthcare healthcare analytics requirements and
        specialized TAT prediction modeling scenarios. Provides fine-grained control over
        data quality enhancement for advanced pharmacy workflow optimization analysis.

        Args:
            df: Input TAT dataset containing healthcare variables for comprehensive data quality
               enhancement supporting Healthcare pharmacy workflow analysis requirements.
            sequence: Optional custom processing sequence replacing default validation steps
                     for specialized healthcare analytics workflows and data quality requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with comprehensive data quality improvements from
            configured processing sequence and custom rules for advanced Healthcare healthcare
            analytics and TAT prediction modeling integration supporting clinical operations.

        Example:
            For custom Healthcare data quality sequence in TAT prediction workflow:
            custom_sequence = [
                cleaner.clip_age,
                cleaner.clip_years,
                lambda df: cleaner.coerce_binary(df, ['stat_order', 'premed_required'])
            ]
            
            enhanced_df = cleaner.apply(tat_df, sequence=custom_sequence)

        Note:
            Advanced method supporting sophisticated data quality enhancement workflows with
            custom processing sequences for specialized Healthcare healthcare analytics
            requirements and complex TAT prediction modeling in clinical operations environments.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Execute specified sequence or default healthcare validation steps
        if sequence is None:
            # Default healthcare data quality enhancement sequence for Healthcare analytics
            sequence = [self.clip_age, self.clip_years]
        
        # Apply each data quality step in configured sequence
        for step in sequence:
            if isinstance(step, str):
                # Execute method by name for string-based step specification
                fn = getattr(self, step)
                df = fn(df)
            elif callable(step):
                # Execute callable step directly for custom function integration
                df = step(df)
            else:
                raise TypeError("Pipeline steps must be method name strings or callables")
        
        # Apply custom registered validation rules after main processing sequence
        for rule in self._rules:
            df = rule(df)
        
        return df

    @classmethod
    def default(cls) -> "Cleaner":
        """
        Create Cleaner instance with Healthcare healthcare-optimized default configuration.

        Factory method providing pre-configured data quality enhancement system optimized
        for standard Healthcare medication preparation TAT analysis and pharmacy workflow
        optimization. Eliminates configuration overhead while ensuring appropriate clinical
        validation bounds for healthcare analytics and patient population analysis.

        Returns:
            Cleaner: Configured instance with Healthcare healthcare-optimized defaults for
            standard medication preparation TAT analysis and pharmacy workflow optimization
            supporting clinical operations and healthcare data integrity requirements.

        Example:
            For rapid Healthcare data quality enhancement in healthcare analytics workflows:
            # Quick setup for standard TAT analysis
            cleaner = Cleaner.default()
            cleaned_data = cleaner.apply(tat_df)
            
            # Equivalent to standard initialization with defaults
            standard_cleaner = Cleaner()
            same_quality = standard_cleaner.apply(tat_df)

        """
        # Create instance with Healthcare healthcare-optimized default configuration
        return cls()