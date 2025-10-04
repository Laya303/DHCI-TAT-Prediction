"""
Operational feature engineering for pharmacy workflow optimization.

Creates features related to staffing, workload, and operational context
for medication preparation TAT analysis.

Technical Features:
- Healthcare professional experience indicators supporting staffing optimization analysis
- Operational capacity and load features enabling pharmacy workflow bottleneck identification
- Patient complexity assessment supporting clinical prioritization and treatment readiness
- Interaction features capturing operational synergies and healthcare workflow dependencies
"""
from typing import Optional
import pandas as pd


class OperationalEngineer:
    """
    Comprehensive operational feature engineering system for pharmacy TAT workflow analysis.

    Orchestrates end-to-end operational feature generation from medication preparation
    datasets with healthcare-optimized operational pattern recognition. Designed for
    production deployment in healthcare pharmacy analytics environment supporting
    TAT prediction modeling, staffing optimization, and comprehensive operational excellence.

    Core Responsibilities:
    - Generate healthcare professional experience indicators for staffing optimization analysis
    - Create operational load and capacity features for pharmacy workflow bottleneck identification
    - Develop patient complexity assessments supporting clinical prioritization and resource allocation
    - Provide interaction features capturing operational synergies and healthcare workflow dependencies

    Operational Feature Categories:

    Experience Features:
    - Healthcare professional experience thresholds for competency-based staffing analysis
    - Nursing and pharmacy professional development indicators for workflow optimization
    - Clinical expertise assessment supporting patient safety and operational efficiency

    Load Features:
    - Floor occupancy indicators for capacity planning and resource allocation optimization
    - Queue management features supporting workflow bottleneck identification and mitigation
    - Operational stress indicators enabling proactive pharmacy operations management

    Complexity Features:
    - Patient case complexity assessment for clinical prioritization and resource allocation
    - Treatment complexity indicators supporting medication preparation workflow optimization
    - Clinical acuity features enabling appropriate staffing and resource deployment

    Interaction Features:
    - Professional experience and operational load synergies for comprehensive workflow analysis
    - Patient complexity and operational capacity interactions for advanced TAT prediction modeling
    - Healthcare workflow dependency patterns supporting operational optimization initiatives

    Args:
        experience_threshold: Healthcare professional experience threshold (years) for competency
                            classification supporting Healthcare staffing optimization analysis.
                            Default 5 years balances clinical expertise with operational efficiency.
        high_occupancy_threshold: Floor occupancy percentage threshold for capacity stress
                                identification supporting pharmacy operations bottleneck analysis.
                                Default 80% reflects healthcare operational capacity planning standards.

    Example:
        # Standard operational feature engineering for Healthcare TAT prediction modeling
        engineer = OperationalEngineer()
        operational_features = engineer.transform(tat_df)
        
        # Custom operational processing with specialized Healthcare requirements
        custom_engineer = OperationalEngineer(
            experience_threshold=7,     # Senior professional competency focus
            high_occupancy_threshold=85 # High-capacity operational environment
        )
        enhanced_features = custom_engineer.transform(tat_df)

      """
    
    def __init__(self, 
                 experience_threshold: int = 5,
                 high_occupancy_threshold: float = 80.0):
        """
        Initialize operational feature engineering system with healthcare-optimized configuration.

        Sets up comprehensive operational feature generation pipeline with appropriate thresholds
        for healthcare medication preparation workflow analysis and pharmacy operations
        optimization. Configures professional experience assessment, capacity planning parameters,
        and operational complexity evaluation for robust healthcare analytics processing.

        Args:
            experience_threshold: Healthcare professional experience threshold (years) defining
                                competency classification boundaries for Healthcare staffing
                                optimization analysis. Default 5 years balances clinical expertise
                                development with operational efficiency and patient safety requirements.
            high_occupancy_threshold: Floor occupancy percentage threshold defining capacity stress
                                    indicators for pharmacy operations bottleneck identification.
                                    Default 80% reflects healthcare operational capacity planning
                                    standards and workflow optimization best practices.

         """
        # Configure healthcare professional experience thresholds for competency assessment
        self.experience_threshold = experience_threshold
        self.high_occupancy_threshold = high_occupancy_threshold

    def add_experience_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate healthcare professional experience indicators for staffing optimization.

        Creates binary competency indicators based on professional experience thresholds
        supporting healthcare staffing optimization and pharmacy workflow analysis.
        Essential for resource allocation planning and professional development initiatives
        through experience-based feature engineering and competency assessment capabilities.

        Healthcare Professional Experience Assessment:
        - Nursing competency indicators: Senior nursing experience for patient safety and workflow efficiency
        - Pharmacy competency indicators: Advanced pharmacy experience for medication preparation optimization
        - Clinical expertise classification: Experience-based competency for staffing and resource allocation
        - Professional development tracking: Experience thresholds supporting career progression analytics

        Staffing Optimization Applications:
        - Resource allocation planning: Experience-based staffing for optimal patient care delivery
        - Professional development: Competency assessment supporting career advancement and training
        - Patient safety enhancement: Senior professional deployment for complex medication preparation
        - Workflow optimization: Experience-matched staffing for efficient pharmacy operations delivery

        Args:
            df: Input TAT dataset containing healthcare professional employment duration variables
               for experience-based competency assessment supporting Healthcare staffing analytics.

        Returns:
            pd.DataFrame: Enhanced dataset with professional experience indicators supporting
            Healthcare staffing optimization, resource allocation planning, and pharmacy
            workflow analysis through healthcare competency assessment and development analytics.

        Example:
            For Healthcare healthcare professional competency assessment in TAT analysis:
            engineer = OperationalEngineer(experience_threshold=7)  # Senior competency focus
            experienced_df = engineer.add_experience_features(tat_df)
            
            # Generated features support staffing optimization analysis
            experience_features = ['high_experience_nurse', 'high_experience_pharmacist']
            competency_analysis = experienced_df[experience_features].value_counts()

        Note:
            Critical for healthcare staffing optimization supporting healthcare professional
            development and pharmacy workflow efficiency through experience-based competency
            assessment enabling resource allocation planning and operational excellence initiatives.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Generate nursing competency indicator for patient safety and workflow optimization
        df['high_experience_nurse'] = (df['nurse_employment_years'] > 
                                     self.experience_threshold).astype(int)
        
        # Generate pharmacy competency indicator for medication preparation optimization
        df['high_experience_pharmacist'] = (df['pharmacist_employment_years'] > 
                                          self.experience_threshold).astype(int)
        
        return df
    
    def add_load_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate operational load and capacity indicators for pharmacy workflow optimization.

        Creates capacity stress and queue management features supporting healthcare pharmacy
        operations bottleneck identification and workflow optimization. Essential for operational
        capacity planning and resource allocation through load-based feature engineering and
        pharmacy workflow efficiency enhancement supporting clinical operations excellence.

        Operational Load Assessment:
        - Floor occupancy indicators: Capacity stress identification for resource allocation planning
        - Queue management features: Workflow bottleneck detection for pharmacy operations optimization
        - Operational stress classification: Load-based indicators supporting proactive management
        - Capacity planning support: Load features enabling efficient resource deployment strategies

        Args:
            df: Input TAT dataset containing operational load variables for capacity assessment
               and workflow optimization supporting Healthcare pharmacy operations analytics.

        Returns:
            pd.DataFrame: Enhanced dataset with operational load indicators supporting Dana
            Farber pharmacy workflow optimization, capacity planning, and bottleneck identification
            through operational stress assessment and resource allocation planning analytics.

        Example:
            For Healthcare operational capacity assessment in pharmacy workflow analysis:
            engineer = OperationalEngineer(high_occupancy_threshold=85)  # High-capacity focus
            load_df = engineer.add_load_features(tat_df)
            
            # Generated features support operational optimization analysis
            load_features = ['high_occupancy', 'long_queue']
            capacity_analysis = load_df[load_features].value_counts()

        Note:
            Essential for healthcare pharmacy operations optimization supporting workflow
            efficiency and resource allocation through operational load assessment enabling
            bottleneck identification and capacity planning for clinical operations excellence.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Generate high occupancy indicator for capacity stress identification
        df['high_occupancy'] = (df['floor_occupancy_pct'] > 
                               self.high_occupancy_threshold).astype(int)
        
        # Generate long queue indicator for workflow bottleneck detection
        df['long_queue'] = (df['queue_length_at_order'] > 
                           df['queue_length_at_order'].median()).astype(int)
        
        return df
    
    def add_complexity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate patient case complexity indicators for clinical prioritization and resource allocation.

        Creates comprehensive patient complexity assessment features supporting healthcare
        clinical decision-making and pharmacy workflow optimization. Essential for treatment
        prioritization and resource allocation through complexity-based feature engineering
        and patient acuity assessment supporting clinical operations and medication preparation.

        Patient Complexity Assessment:
        - Clinical severity indicators: High-severity cases requiring enhanced resource allocation
        - Treatment complexity factors: Premedication requirements and STAT orders for prioritization
        - Patient acuity classification: Comprehensive complexity assessment for workflow optimization
        - Clinical decision support: Complexity features enabling appropriate resource deployment

        Clinical Applications:
        - Treatment prioritization: Complexity-based patient scheduling for optimal care delivery
        - Resource allocation: Acuity-matched staffing for complex medication preparation requirements
        - Workflow optimization: Complexity indicators supporting efficient pharmacy operations planning
        - Patient safety enhancement: Complex case identification for enhanced clinical attention

        Complexity Factors Integration:
        - High severity: Critical patient conditions requiring immediate attention and resources
        - Premedication requirements: Additional preparation complexity affecting TAT and workflow
        - STAT orders: Urgent medication preparation requiring priority resource allocation
        - Comprehensive assessment: Multi-factor complexity evaluation for clinical decision-making

        Args:
            df: Input TAT dataset containing patient clinical variables for complexity assessment
               and prioritization supporting Healthcare clinical decision-making and workflow optimization.

        Returns:
            pd.DataFrame: Enhanced dataset with patient complexity indicators supporting Dana
            Farber clinical prioritization, resource allocation, and pharmacy workflow optimization
            through comprehensive patient acuity assessment and treatment complexity analytics.

        Example:
            For Healthcare patient complexity assessment in clinical workflow analysis:
            engineer = OperationalEngineer()
            complex_df = engineer.add_complexity_features(tat_df)
            
            # Generated features support clinical prioritization analysis
            complexity_distribution = complex_df['complex_case'].value_counts()
            print(f"Complex cases: {complexity_distribution[1]} / {len(complex_df)}")

        Note:
            Critical for healthcare clinical decision-making supporting patient prioritization
            and pharmacy workflow optimization through comprehensive complexity assessment enabling
            appropriate resource allocation and treatment planning for clinical operations excellence.
        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Generate comprehensive patient complexity indicator combining multiple clinical factors
        df['complex_case'] = ((df['severity'] == 'High') | 
                            (df['premed_required'] == 1) | 
                            (df['stat_order'] == 1)).astype(int)
        
        return df

    def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate operational interaction features capturing healthcare workflow synergies.

        Creates sophisticated interaction terms combining operational variables supporting
        healthcare advanced TAT prediction modeling and pharmacy workflow optimization.
        Essential for capturing complex operational dependencies and healthcare workflow
        synergies through interaction-based feature engineering and operational pattern recognition.

        Operational Interaction Categories:

        Professional Experience × Operational Load:
        - Nurse experience × occupancy: Senior nursing performance under operational stress
        - Pharmacist experience × queue: Advanced pharmacy efficiency with workflow pressure
        - Experience-load synergies: Professional competency interaction with operational demands

        Clinical Complexity × Operational Capacity:
        - Severity × occupancy: Patient acuity interaction with operational capacity stress
        - Readiness × queue: Patient preparation status interaction with workflow bottlenecks
        - Complexity-capacity dependencies: Clinical demands interaction with operational resources

        Healthcare Workflow Applications:
        - Advanced TAT prediction: Interaction features for sophisticated modeling and forecasting
        - Operational optimization: Synergy patterns supporting workflow efficiency enhancement
        - Resource allocation: Interaction-based staffing for optimal operational performance
        - Bottleneck identification: Complex dependency patterns for workflow optimization analysis

        Args:
            df: Input TAT dataset containing operational variables for interaction feature generation
               supporting Healthcare advanced analytics and pharmacy workflow optimization requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with operational interaction features supporting Dana
            Farber advanced TAT prediction modeling, workflow optimization, and operational
            excellence through sophisticated operational pattern recognition and dependency analysis.

        Example:
            For Healthcare advanced operational analytics in TAT prediction modeling:
            engineer = OperationalEngineer()
            interaction_df = engineer.add_interaction_features(tat_df)
            
            # Generated interaction features for advanced modeling
            interaction_features = [col for col in interaction_df.columns if '_x_' in col]
            print(f"Generated {len(interaction_features)} interaction features for modeling")

        """
        # Create defensive copy to prevent in-place modifications of caller's dataset
        df = df.copy()
        
        # Generate professional experience × operational load interaction features
        df['nurse_exp_x_occupancy'] = (df['nurse_employment_years'] * 
                                      df['floor_occupancy_pct'])
        df['pharmacist_exp_x_queue'] = (df['pharmacist_employment_years'] * 
                                       df['queue_length_at_order'])
        
        # Generate clinical complexity × operational capacity interaction features
        df['severity_x_occupancy'] = (pd.Categorical(df['severity']).codes * 
                                    df['floor_occupancy_pct'])
        df['readiness_x_queue'] = (df['patient_readiness_score'] * 
                                  df['queue_length_at_order'])
        
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute comprehensive operational feature engineering using integrated pipeline.

        Orchestrates complete operational feature generation workflow applying all operational
        transformation steps for healthcare TAT prediction modeling and pharmacy workflow
        optimization. Provides streamlined interface for operational feature engineering in
        healthcare analytics supporting clinical operations excellence and efficiency initiatives.

        Args:
            df: Input TAT dataset containing operational variables for comprehensive feature
               engineering supporting Healthcare pharmacy workflow optimization requirements.

        Returns:
            pd.DataFrame: Enhanced dataset with comprehensive operational features suitable for
            TAT prediction modeling, pharmacy workflow optimization, and Healthcare healthcare
            operations analytics supporting clinical decision-making and operational excellence.

        Example:
            For comprehensive Healthcare operational feature engineering in TAT prediction workflow:
            engineer = OperationalEngineer()
            operational_features = engineer.transform(tat_df)
            
            # Validate operational feature engineering results and comprehensive analysis capabilities
            operational_cols = [col for col in operational_features.columns 
                              if any(term in col for term in ['experience', 'occupancy', 'queue', 'complex'])]
            print(f"Generated {len(operational_cols)} operational features for workflow analysis")

        """
        # Execute comprehensive operational feature engineering pipeline
        df = self.add_experience_features(df)
        df = self.add_load_features(df)
        df = self.add_complexity_features(df)
        df = self.add_interaction_features(df)
        return df

    @classmethod
    def default(cls) -> "OperationalEngineer":
        """
        Create OperationalEngineer instance with Healthcare healthcare-optimized configuration.

        Factory method providing pre-configured operational feature engineering system optimized
        for standard Healthcare medication preparation TAT analysis and pharmacy workflow
        optimization. Eliminates configuration overhead while ensuring appropriate operational
        thresholds for healthcare analytics and clinical operations environments.

        Returns:
            OperationalEngineer: Configured instance with Healthcare healthcare-optimized defaults
            for standard medication preparation TAT analysis and pharmacy workflow optimization
            supporting clinical operations excellence and healthcare professional development.

        Example:
            For rapid Healthcare operational feature engineering in healthcare analytics workflows:
            # Quick setup for standard operational analysis
            engineer = OperationalEngineer.default()
            operational_features = engineer.transform(tat_df)
            
            # Equivalent to standard initialization with optimized defaults
            standard_engineer = OperationalEngineer(
                experience_threshold=5,      # Balanced professional competency assessment
                high_occupancy_threshold=80.0 # Healthcare capacity planning standard
            )

       """
        # Create instance with Healthcare healthcare-optimized default configuration
        return cls(
            experience_threshold=5,         # Balanced professional competency for staffing optimization
            high_occupancy_threshold=80.0   # Healthcare operational capacity planning standard
        )