"""
Laboratory feature engineering for medication preparation analysis.

Processes clinical lab values with validation, abnormality detection,
and derived feature creation for TAT prediction.
- Production-ready: Scalable processing suitable for 100k+ patient order datasets in analytics

Technical Features:
- Clinical bounds enforcement preventing medically unrealistic laboratory values
- Reference range validation supporting accurate clinical decision-making processes
- Laboratory abnormality detection enabling patient safety and treatment readiness assessment
- Extensible pipeline architecture supporting custom Healthcare clinical requirements

Laboratory Test Coverage:
- Complete Blood Count (CBC): WBC, Hemoglobin, Platelets for hematologic assessment
- Renal Function: Creatinine for medication dosing and nephrotoxicity monitoring
- Hepatic Function: ALT for liver function assessment and drug metabolism evaluation
"""
import pandas as pd
import numpy as np
from typing import Callable, Iterable, Dict, List, Optional, Any
from tat.config import LAB_COLS, LAB_CLIPS, LAB_NORMALS

# Type alias for laboratory processing functions in clinical feature engineering workflows
Step = Callable[[pd.DataFrame], pd.DataFrame]

class LabProcessor:
    """
    Comprehensive clinical laboratory feature engineering system for pharmacy TAT workflow analysis.

    Orchestrates end-to-end laboratory data processing from patient datasets
    with healthcare-optimized clinical validation and feature extraction. Designed for
    production deployment in healthcare pharmacy analytics environment supporting
    TAT prediction modeling, patient safety monitoring, and comprehensive clinical analytics.

    Core Responsibilities:
    - Validate laboratory values against medical reference ranges ensuring clinical realism
    - Generate clinical abnormality flags supporting patient safety and treatment readiness
    - Create derived laboratory features for comprehensive patient analytics
    - Provide extensible pipeline architecture for specialized Healthcare clinical requirements

    Laboratory Test Coverage and Reference Ranges:
    - WBC (White Blood Cell Count): 4.0-11.0 (×10³/μL) - immune system assessment
    - HGB (Hemoglobin): 12.0-16.0 g/dL - oxygen carrying capacity and anemia detection
    - Platelets: 150-400 (×10³/μL) - bleeding risk assessment for chemotherapy safety
    - Creatinine: 0.6-1.3 mg/dL - renal function monitoring for drug dosing adjustment
    - ALT (Alanine Aminotransferase): 7-56 U/L - hepatic function and drug toxicity monitoring

    Args:
        cols: Laboratory column names for processing. Defaults to configured LAB_COLS
             covering standard healthcare laboratory monitoring requirements.
        clips: Laboratory value clipping ranges ensuring medically realistic bounds.
              Defaults to LAB_CLIPS with clinical validation for patient safety.
        normals: Medical reference ranges for abnormality detection. Defaults to
                LAB_NORMALS aligned with patient monitoring standards.
        missing_strategy: Missing value handling approach ('mean', 'median', 'flag')
                         supporting clinical data integrity and transparency requirements.

    Example:
        # Standard laboratory feature engineering for Healthcare TAT prediction modeling
        lab_processor = LabProcessor()
        lab_features = lab_processor.transform(tat_df)
        
        # Custom laboratory processing with healthcare-specific configuration
        healthcare_processor = LabProcessor(
            missing_strategy='flag',  # Preserve missing lab transparency
            clips=custom_healthcare_clips
        )
        enhanced_features = healthcare_processor.fit_transform(tat_df)

 """

    def __init__(
        self,
        cols: Optional[Iterable[str]] = None,
        clips: Optional[Dict[str, tuple]] = None,
        normals: Optional[Dict[str, tuple]] = None,
        missing_strategy: str = 'median'  # 'mean', 'median', or 'flag'
    ):
        """
        Initialize clinical laboratory processing system with healthcare-optimized configuration.

        Sets up comprehensive laboratory feature engineering pipeline with appropriate clinical
        validation bounds for healthcare patient population analysis and pharmacy
        operations optimization. Configures medical reference ranges, missing value handling,
        and clinical abnormality detection for robust healthcare analytics processing.

        Args:
            cols: Laboratory column names for clinical processing. Defaults to configured
                 LAB_COLS covering standard healthcare laboratory monitoring including CBC,
                 renal function, and hepatic assessment for comprehensive patient analytics.
            clips: Laboratory value clipping ranges ensuring medically realistic bounds for
                  patient safety. Defaults to LAB_CLIPS with clinical validation preventing
                  physiologically impossible values in Healthcare analytics workflows.
            normals: Medical reference ranges for clinical abnormality detection. Defaults
                    to LAB_NORMALS aligned with patient monitoring standards and
                    clinical decision-making requirements for treatment readiness assessment.
            missing_strategy: Missing laboratory value handling approach supporting clinical
                             data integrity. Options: 'mean', 'median' (robust imputation),
                             'flag' (preserve transparency) for healthcare analytics requirements.

    """
        # Configure laboratory columns for patient clinical monitoring
        self.cols = list(cols) if cols is not None else list(LAB_COLS)
        self.clips = clips or dict(LAB_CLIPS)
        self.normals = normals or dict(LAB_NORMALS)
        self.missing_strategy = missing_strategy
        
        # Initialize custom clinical processing steps for specialized Healthcare requirements
        self._custom_steps: List[Step] = []
        
        # Configure default clinical laboratory processing pipeline for healthcare analytics
        self._pipeline: List[Step] = [
            self.clip_labs,
            self.handle_missing_values,
            self.add_lab_flags,
            self.add_derived_features
        ]
        
        # Cache for clinical imputation values supporting consistent laboratory processing
        self._imputation_values: Dict[str, float] = {}

    def clip_labs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce medical bounds for laboratory values ensuring clinical realism.

        Validates laboratory test results against medically sensible ranges preventing
        physiologically impossible values from compromising healthcare TAT analysis
        and pharmacy workflow optimization. Essential for maintaining clinical data
        integrity and supporting accurate patient safety assessment in healthcare environments.

        Laboratory Bounds Enforcement:
        - WBC: Prevents impossible white blood cell counts ensuring hematologic assessment validity
        - Hemoglobin: Enforces physiologic bounds for accurate anemia and transfusion assessment
        - Platelets: Validates platelet counts for reliable bleeding risk and safety evaluation
        - Creatinine: Ensures realistic renal function values for accurate drug dosing assessment
        - ALT: Enforces hepatic enzyme bounds for reliable liver function and toxicity monitoring

        Args:
            df: Input TAT dataset containing laboratory values for clinical validation
               and bounds enforcement supporting Healthcare healthcare analytics requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with validated laboratory values ensuring medically
            realistic clinical data for Healthcare TAT prediction modeling and pharmacy
            workflow optimization supporting patient safety and clinical decision-making.

        Example:
            For Healthcare clinical laboratory validation in TAT analysis workflow:
            processor = LabProcessor(clips={
                'lab_WBC_k_per_uL': (0.1, 50.0),  # Healthcare-specific WBC range
                'lab_Platelets_k_per_uL': (5.0, 1000.0)  # Chemotherapy impact range
            })
            validated_df = processor.clip_labs(tat_df)
   """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Apply clinical laboratory bounds validation for configured test columns
        for col in self.cols:
            if col in df and col in self.clips:
                lo, hi = self.clips[col]
                # Robust numeric coercion with clinical bounds enforcement for patient safety
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lo, upper=hi)
        
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process missing laboratory values with clinical context preservation.

        Implements healthcare-optimized missing value handling strategies preserving clinical
        data transparency while supporting robust TAT prediction modeling. Essential for
        healthcare laboratory analytics maintaining clinical interpretation capabilities
        and healthcare data integrity throughout pharmacy workflow optimization analysis.

        Clinical Missing Value Strategies:
        
        Flag Strategy ('flag'):
        - Preserves missing laboratory transparency for clinical interpretation
        - Adds binary indicators for missing laboratory tests supporting clinical decision-making
        - Maintains healthcare data integrity without imputation assumptions
        - Supports clinical workflows requiring explicit missing data awareness

        Imputation Strategies ('mean', 'median'):
        - Robust statistical imputation supporting machine learning model training requirements
        - Median imputation preferred for clinical data reducing outlier sensitivity
        - Consistent imputation values cached for production deployment stability
        - Maintains analytical capabilities while addressing missing laboratory patterns

        Args:
            df: Input TAT dataset containing laboratory variables with potential missing values
               for clinical context-aware processing supporting Healthcare analytics requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with processed missing laboratory values maintaining
            clinical data integrity for Healthcare TAT prediction modeling and pharmacy
            workflow optimization supporting healthcare analytics and clinical decision-making.

        Example:
            For Healthcare clinical laboratory missing value processing:
            # Preserve clinical transparency
            flag_processor = LabProcessor(missing_strategy='flag')
            flagged_df = flag_processor.handle_missing_values(tat_df)
            
            # Robust imputation for modeling
            impute_processor = LabProcessor(missing_strategy='median')
            imputed_df = impute_processor.handle_missing_values(tat_df)

       """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Flag strategy: Preserve missing laboratory transparency for clinical interpretation
        if self.missing_strategy == 'flag':
            for col in self.cols:
                if col in df:
                    # Add missing indicators maintaining clinical data transparency
                    df[f"{col}_missing"] = df[col].isna().astype(int)
            return df
            
        # Compute clinical imputation values for consistent laboratory processing
        if not self._imputation_values:
            for col in self.cols:
                if col in df:
                    series = pd.to_numeric(df[col], errors='coerce')
                    # Handle empty series to avoid numpy warnings
                    if series.dropna().empty:
                        self._imputation_values[col] = np.nan  # Use NaN for completely missing lab values
                    elif self.missing_strategy == 'mean':
                        self._imputation_values[col] = series.mean()
                    else:  # median - preferred for clinical data robustness
                        self._imputation_values[col] = series.median()
        
        # Apply clinical imputation maintaining healthcare data integrity
        for col in self.cols:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
                    self._imputation_values.get(col, np.nan)
                )
        
        return df

    def add_lab_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate clinical abnormality flags for laboratory values outside reference ranges.

        Creates binary indicators for laboratory test results falling outside medical reference
        ranges essential for healthcare patient safety monitoring and TAT bottleneck
        identification. Supports clinical decision-making through automated abnormality
        detection and comprehensive patient assessment in pharmacy workflow analysis.

        Args:
            df: Input TAT dataset containing validated laboratory values for clinical abnormality
               detection supporting Healthcare patient safety and pharmacy analytics requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with laboratory abnormality flags supporting Dana
            Farber clinical decision-making, patient safety monitoring, and TAT bottleneck
            identification through comprehensive healthcare laboratory assessment capabilities.

        Example:
            For Healthcare clinical laboratory abnormality detection in TAT analysis:
            processor = LabProcessor()
            flagged_df = processor.add_lab_flags(tat_df)
            
            # Generated flags include: lab_WBC_low, lab_WBC_high, lab_HGB_low, etc.
            abnormal_flags = [col for col in flagged_df.columns if col.endswith(('_low', '_high'))]
            print(f"Generated {len(abnormal_flags)} laboratory abnormality flags")

     """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Generate clinical abnormality flags for each configured laboratory test
        for col, (lo_norm, hi_norm) in self.normals.items():
            if col not in df:
                continue
            
            # Extract laboratory base name for flag generation
            base = col.rsplit("_", 1)[0] if "_" in col else col
            series = pd.to_numeric(df[col], errors="coerce")
            
            # Generate low abnormality flag for below-normal laboratory results
            df[f"{base}_low"] = (series < lo_norm).fillna(False).astype(int)
            
            # Generate high abnormality flag for above-normal laboratory results  
            df[f"{base}_high"] = (series > hi_norm).fillna(False).astype(int)
        
        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create clinically relevant derived laboratory features for comprehensive patient assessment.

        Generates advanced laboratory features combining multiple test results supporting
        healthcare patient analytics and TAT prediction modeling. Essential for
        comprehensive patient acuity assessment and clinical decision-making through
        sophisticated laboratory pattern recognition and healthcare workflow optimization.

        Clinical Derived Features:

        Laboratory Abnormality Summary:
        - lab_abnormal_count: Total number of abnormal laboratory results per patient
        - has_abnormal_labs: Binary indicator for any laboratory abnormalities present
        - Clinical significance: Overall patient laboratory status for treatment readiness

        Complete Blood Count (CBC) Panel Assessment:
        - cbc_abnormal_count: Hematologic abnormalities affecting chemotherapy safety
        - has_abnormal_cbc: Binary CBC panel abnormality indicator for treatment decisions
        - Clinical significance: Blood count status critical for healthcare treatment safety
    
        Args:
            df: Input TAT dataset containing laboratory abnormality flags for derived feature
               generation supporting Healthcare comprehensive patient analytics requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with derived laboratory features supporting Dana
            Farber TAT prediction modeling, patient acuity assessment, and pharmacy workflow
            optimization through comprehensive clinical laboratory pattern recognition capabilities.

        Example:
            For Healthcare comprehensive laboratory feature engineering:
            processor = LabProcessor()
            enhanced_df = processor.add_derived_features(tat_df)
            
            # Access derived clinical features
            derived_features = ['lab_abnormal_count', 'has_abnormal_labs', 
                              'cbc_abnormal_count', 'has_abnormal_cbc']
            print("Generated derived laboratory features for clinical assessment")

     """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Identify laboratory abnormality flag columns for derived feature generation
        flag_cols = [c for c in df.columns if c.endswith('_low') or c.endswith('_high')]
        
        # Generate comprehensive laboratory abnormality summary features
        if flag_cols:
            # Total abnormal laboratory count for patient acuity assessment
            df['lab_abnormal_count'] = df[flag_cols].sum(axis=1)
            
            # Binary indicator for any laboratory abnormalities present
            df['has_abnormal_labs'] = (df['lab_abnormal_count'] > 0).astype(int)
        
        # Generate Complete Blood Count (CBC) panel specific assessment features
        cbc_flags = [c for c in flag_cols if any(lab in c for lab in ['WBC', 'HGB', 'Platelets'])]
        if cbc_flags:
            # Hematologic abnormality count for chemotherapy safety assessment
            df['cbc_abnormal_count'] = df[cbc_flags].sum(axis=1)
            
            # Binary CBC panel abnormality indicator for treatment readiness
            df['has_abnormal_cbc'] = (df['cbc_abnormal_count'] > 0).astype(int)
        
        return df

    def register(self, step: Step) -> "LabProcessor":
        """
        Register custom laboratory processing step for specialized Healthcare requirements.

        Extends laboratory feature engineering pipeline with custom clinical processing functions
        supporting specialized healthcare analytics requirements and domain-specific laboratory
        pattern analysis. Essential for implementing Healthcare-specific laboratory transformations
        and advanced patient assessment workflows supporting clinical operations optimization.

        Args:
            step: Custom laboratory processing function accepting DataFrame and returning enhanced
                 DataFrame with additional clinical features for Healthcare healthcare analytics.

        Returns:
            LabProcessor: Self reference enabling method chaining for laboratory pipeline construction
            and streamlined clinical feature engineering workflow configuration.

        Example:
            For specialized Healthcare laboratory feature engineering:
            def add_healthcare_risk_scores(df):
                # Custom healthcare laboratory risk assessment
                if 'lab_WBC_k_per_uL' in df.columns and 'lab_Platelets_k_per_uL' in df.columns:
                    df['infection_risk_score'] = calculate_infection_risk(df)
                    df['bleeding_risk_score'] = calculate_bleeding_risk(df)
                return df
            
            processor = LabProcessor()
            processor.register(add_healthcare_risk_scores)
            enhanced_df = processor.transform(tat_df)

        Note:
            Custom steps execute after standard laboratory processing enabling sophisticated Dana
            Farber clinical analytics workflows with specialized laboratory pattern recognition
            supporting advanced TAT prediction modeling and patient assessment requirements.
        """
        # Add custom laboratory processing step to pipeline for execution after standard processing
        self._custom_steps.append(step)
        return self

    def clear_custom(self) -> "LabProcessor":
        """
        Remove all registered custom laboratory processing steps for pipeline reset.

        Provides laboratory processing pipeline reset capability for iterative clinical feature
        engineering development and testing workflows. Essential for maintaining clean pipeline
        configuration during Healthcare analytics development and ensuring reproducible
        laboratory feature extraction in production TAT prediction modeling environments.


        Returns:
            LabProcessor: Self reference enabling method chaining for streamlined laboratory
            pipeline configuration and clinical feature engineering workflow management.

        Example:
            For iterative Healthcare laboratory feature engineering development:
            processor = LabProcessor()
            processor.register(custom_healthcare_scorer)
            processor.register(custom_safety_assessor)
            
            # Reset custom steps for clean pipeline testing
            processor.clear_custom()
            baseline_features = processor.transform(tat_df)

        Note:
            Preserves standard clinical laboratory pipeline while removing custom extensions
            enabling controlled laboratory feature engineering experimentation in Healthcare
            analytics development and TAT prediction modeling validation supporting clinical operations.
        """
        # Clear all custom laboratory processing steps while preserving standard clinical pipeline
        self._custom_steps.clear()
        return self

    def apply(self, df: pd.DataFrame, sequence: Optional[Iterable[Step]] = None) -> pd.DataFrame:
        """
        Execute comprehensive laboratory processing using configurable clinical sequence.

        Orchestrates end-to-end laboratory feature engineering workflow with flexible processing
        sequence configuration supporting diverse Healthcare healthcare analytics requirements
        and specialized TAT prediction modeling scenarios. Provides fine-grained control over
        laboratory processing for advanced patient assessment and clinical analytics.

        Args:
            df: Input TAT dataset containing laboratory variables for comprehensive processing
               supporting Healthcare patient analytics and clinical assessment requirements.
            sequence: Optional custom processing sequence replacing default laboratory processing
                     steps for specialized healthcare analytics workflows and clinical requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with comprehensive laboratory features from configured
            processing sequence and custom steps for advanced Healthcare healthcare analytics
            and TAT prediction modeling integration supporting clinical operations optimization.

        Example:
            For custom Healthcare laboratory processing sequence in TAT prediction workflow:
            custom_sequence = [
                processor.clip_labs,
                processor.add_lab_flags,
                custom_healthcare_assessment
            ]
            
            enhanced_df = processor.apply(tat_df, sequence=custom_sequence)

        Note:
            Advanced method supporting sophisticated laboratory feature engineering workflows with
            custom processing sequences for specialized Healthcare healthcare analytics requirements
            and complex TAT prediction modeling in healthcare clinical operations environments.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Execute specified sequence or default laboratory processing steps in configured order
        seq = list(sequence) if sequence is not None else list(self._pipeline)
        for step in seq:
            df = step(df)
            
        # Apply custom registered laboratory processing steps after main sequence completion
        for step in self._custom_steps:
            df = step(df)
            
        return df

    def fit(self, df: pd.DataFrame) -> "LabProcessor":
        """
        Scikit-learn compatible fit method for estimator-like interface consistency.

        No-operation method maintained for compatibility with sklearn-style pipelines and
        automated machine learning workflows in healthcare healthcare analytics environment.
        LabProcessor processes laboratory data statically without requiring training or
        parameter estimation from input data patterns for robust clinical processing.

        Args:
            df: TAT dataset for interface consistency (not used in laboratory processing)

        Returns:
            LabProcessor: Self reference for method chaining compatibility with sklearn pipeline
            patterns and automated Healthcare healthcare analytics workflows.

        Note:
            Maintained for sklearn pipeline compatibility in healthcare healthcare analytics
            environment. Laboratory processing operates statically without requiring data-driven
            parameter estimation for consistent clinical processing behavior.
        """
        # No-operation method for sklearn transformer interface compatibility
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute comprehensive laboratory feature engineering using default clinical pipeline.

        Applies complete laboratory processing workflow using standard Healthcare configuration
        suitable for routine patient TAT analysis and pharmacy workflow optimization.
        Provides streamlined interface for laboratory feature engineering in healthcare analytics
        and TAT prediction modeling supporting clinical operations and patient safety monitoring.

        Standard Processing Workflow:
        1. Clinical laboratory bounds validation ensuring medically realistic values
        2. Missing value handling with clinical context preservation and transparency
        3. Medical reference range abnormality detection for patient safety assessment
        4. Derived clinical features for comprehensive patient analytics
        5. Custom registered processing for specialized Healthcare requirements

        Args:
            df: Input TAT dataset containing laboratory variables for standard clinical processing
               supporting Healthcare patient analytics and pharmacy workflow requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with comprehensive laboratory features suitable for
            TAT prediction modeling, pharmacy workflow optimization, and Healthcare healthcare
            operations analytics supporting clinical decision-making and patient safety monitoring.

        Example:
            For standard Healthcare laboratory feature engineering in TAT prediction workflow:
            processor = LabProcessor()
            lab_features = processor.transform(tat_df)
            
            # Validate laboratory feature engineering results and clinical assessment capabilities
            lab_cols = [col for col in lab_features.columns if 'lab_' in col]
            print(f"Generated {len(lab_cols)} laboratory features for clinical analysis")

        Note:
            Primary method for laboratory feature engineering in Healthcare healthcare analytics
            workflows providing comprehensive clinical laboratory features for TAT prediction
            modeling and pharmacy operations optimization in healthcare clinical environments.
        """
        # Execute default laboratory processing pipeline for comprehensive clinical analysis
        return self.apply(df)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience method combining fit and transform for sklearn-style workflow integration.

        Provides streamlined interface for laboratory feature engineering in automated Dana
        Farber healthcare analytics pipelines and machine learning workflows. Equivalent to
        calling fit() followed by transform() while maintaining method chaining compatibility
        for comprehensive feature engineering pipeline construction and clinical analytics.

        Args:
            df: Input TAT dataset for comprehensive laboratory feature engineering using healthcare-
               optimized processing supporting Healthcare patient analytics requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with laboratory features ready for Healthcare TAT
            prediction modeling, pharmacy workflow optimization, and healthcare operations
            analytics supporting clinical decision-making and patient safety monitoring initiatives.

        Example:
            For streamlined Healthcare laboratory processing in automated analytics pipelines:
            processor = LabProcessor()
            lab_features = processor.fit_transform(tat_df)
            
            # Integration with sklearn pipeline for automated model training
            from sklearn.pipeline import Pipeline
            tat_pipeline = Pipeline([
                ('laboratory', processor),
                ('model', xgb_regressor)
            ])
        """
        # Execute fit and transform in sequence for sklearn-style workflow integration
        return self.fit(df).transform(df)

    @classmethod
    def default(cls) -> "LabProcessor":
        """
        Create LabProcessor instance with Healthcare healthcare-optimized default configuration.

        Factory method providing pre-configured laboratory processing system optimized for
        standard Healthcare patient TAT analysis and pharmacy workflow optimization.
        Eliminates configuration overhead while ensuring appropriate clinical validation bounds
        for healthcare analytics and patient safety monitoring in clinical operations environments.

        Returns:
            LabProcessor: Configured instance with Healthcare healthcare-optimized defaults for
            standard patient TAT analysis and pharmacy workflow optimization supporting
            clinical operations and patient safety monitoring requirements.

        Example:
            For rapid Healthcare laboratory processing in healthcare analytics workflows:
            # Quick setup for standard clinical analysis
            processor = LabProcessor.default()
            lab_features = processor.transform(tat_df)
            
            # Equivalent to standard initialization with optimized defaults
            standard_processor = LabProcessor(
                missing_strategy='median',  # Robust clinical imputation
                clips=LAB_CLIPS,           # Medical validation bounds
                normals=LAB_NORMALS        # Clinical reference ranges
            )
    """
        # Create instance with Healthcare healthcare-optimized default configuration
        return cls(
            missing_strategy='median',  # Robust clinical imputation reducing outlier sensitivity
            clips=LAB_CLIPS,           # Medical validation bounds ensuring patient safety
            normals=LAB_NORMALS        # Clinical reference ranges for patient monitoring
        )