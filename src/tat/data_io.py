"""
Data I/O utilities for TAT analysis.

Provides DataIO class for loading, validating, and processing healthcare datasets
with automated quality checks and error handling.
"""
import os
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging
from pathlib import Path

# Configure logging for healthcare data I/O operations
logger = logging.getLogger(__name__)

class DataIO:
    """
    Healthcare Analytics Data I/O Manager for TAT Prediction System
    
    Comprehensive data loading and validation utilities optimized for medication
    preparation TAT analysis supporting healthcare pharmacy workflow optimization.
    Provides healthcare-specific data handling with clinical validation, quality
    assurance, and production-ready error handling for robust analytics pipelines.

    Args:
        sort_column: Primary sorting column for consistent data organization,
                    defaults to "order_id" for medication order chronological analysis.
        sort_ascending: Sorting direction for temporal analysis, defaults to True
                       supporting chronological workflow sequence analysis.
        validate_healthcare_data: Enable healthcare-specific validation checks
                                 ensuring clinical data integrity and compliance.
    
    Example:
        For comprehensive healthcare data loading:
        ```python
        # Initialize with healthcare validation
        io = DataIO(validate_healthcare_data=True)
        
        # Load medication preparation dataset with validation
        df = io.read_csv("Healthcare_TAT_Dataset.csv")
        
        # Validate TAT threshold compliance
        validation_report = io.validate_tat_data(df)
        ```
    """

    def __init__(
        self, 
        sort_column: Optional[str] = "order_id", 
        sort_ascending: bool = True,
        validate_healthcare_data: bool = True
    ):
        self.sort_column = sort_column
        self.sort_ascending = sort_ascending
        self.validate_healthcare_data = validate_healthcare_data
        
        # Healthcare data validation requirements
        self.required_healthcare_columns = {
            'doctor_order_time',           # Temporal anchor for TAT calculations
            'nurse_validation_time',       # First workflow step validation
            'prep_complete_time',          # Pharmacy preparation completion
            'second_validation_time',      # Secondary quality validation
            'floor_dispatch_time',         # Medication dispatch workflow
            'patient_infusion_time',       # Final administration completion
            'TAT_minutes',                 # Primary continuous target for prediction
            'age',                         # Patient demographic for complexity
            'severity',                    # Clinical severity affecting workflow
            'floor',                       # Treatment location for logistics
            'shift'                        # Temporal context for staffing analysis
        }
        
        logger.info(f"DataIO initialized for healthcare analytics with validation: {validate_healthcare_data}")

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Load healthcare dataset with comprehensive validation and clinical quality checks.
        
        Provides healthcare-optimized CSV loading with automated data validation,
        clinical quality assessment, and TAT analysis preparation. Essential for
        medication preparation workflow analysis supporting healthcare pharmacy
        operations optimization and 60-minute threshold compliance requirements.
        
        Healthcare Data Processing:
        - Automated clinical data validation ensuring medication order integrity
        - TAT threshold analysis supporting 60-minute quality compliance assessment
        - Healthcare workflow organization through temporal sorting and sequence validation
        - Missing data assessment with clinical context and workflow impact analysis
        - Production error handling supporting robust healthcare analytics pipeline operation
        
        Args:
            path: Healthcare dataset file path supporting medication preparation analysis
            **kwargs: Additional pandas read_csv parameters for healthcare data customization
        
        Returns:
            pd.DataFrame: Validated healthcare dataset ready for TAT analysis and
            bottleneck identification supporting pharmacy workflow optimization.
        
        Raises:
            FileNotFoundError: Healthcare dataset not accessible for clinical analysis
            ValueError: Data validation failure compromising healthcare analytics integrity
            HealthcareDataError: Clinical data quality issues affecting TAT analysis
        
        Example:
            For medication preparation dataset loading:
            ```python
            io = DataIO(validate_healthcare_data=True)
            df = io.read_csv("Healthcare_TAT_Dataset.csv")
            
            # Automatic validation provides clinical insights
            print(f"Orders loaded: {len(df):,}")
            print(f"TAT >60min: {(df['TAT_minutes'] > 60).mean():.1%}")
            ```

        """
        logger.info(f"Loading healthcare dataset: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Healthcare dataset not found: {path}")

        try:
            # Load healthcare dataset with pandas optimization
            df = pd.read_csv(path, **kwargs)
            logger.info(f"✅ Dataset loaded: {df.shape[0]:,} medication orders, {df.shape[1]} variables")
            
        except Exception as e:
            logger.error(f"❌ Failed to load healthcare dataset {path}: {e}")
            raise ValueError(f"Healthcare data loading failed for {path}: {e}") from e

        # Apply healthcare-specific validation if enabled
        if self.validate_healthcare_data:
            try:
                validation_results = self._validate_healthcare_dataset(df, path)
                logger.info("✅ Healthcare data validation completed successfully")
                
                # Log critical validation insights for clinical stakeholders
                if 'tat_analysis' in validation_results:
                    tat_stats = validation_results['tat_analysis']
                    logger.info(f"   • TAT threshold (60min) violations: {tat_stats['violation_rate']:.1%}")
                    logger.info(f"   • Average TAT: {tat_stats['mean_tat']:.1f} minutes")
                
            except Exception as e:
                logger.warning(f"⚠️  Healthcare validation warning: {e}")
                # Continue processing with warning - don't fail pipeline

        # Apply healthcare workflow sorting for temporal analysis
        if self.sort_column and self.sort_column in df.columns:
            df = df.sort_values(
                by=self.sort_column, 
                ascending=self.sort_ascending
            ).reset_index(drop=True)
            logger.info(f"   • Data sorted by {self.sort_column} for temporal workflow analysis")
        
        return df

    def write_csv(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """
        Save healthcare analytics results with production-ready format and documentation.
        
        Provides healthcare-optimized data export supporting clinical documentation
        requirements and MLOps pipeline integration. Essential for pharmacy workflow
        optimization results, TAT analysis artifacts, and bottleneck identification
        reporting supporting healthcare healthcare operations excellence initiatives.
        
        Healthcare Output Features:
        - Production-ready formatting supporting healthcare IT infrastructure integration
        - Clinical documentation compliance ensuring audit trail and quality assurance
        - MLOps artifact management enabling automated pipeline monitoring and deployment
        - Healthcare stakeholder communication through organized data export formatting
        - Quality validation ensuring output data integrity for clinical decision-making
        
        Args:
            df: Healthcare analytics DataFrame containing TAT analysis results,
               bottleneck identification, or model predictions for clinical application.
            path: Output file path supporting healthcare documentation and MLOps integration
            **kwargs: Additional pandas to_csv parameters for healthcare output customization
        
        Raises:
            IOError: Healthcare data export failure affecting clinical documentation
            ValueError: Invalid data format compromising healthcare analytics integrity
        
        Example:
            For TAT analysis results export:
            ```python
            # Save bottleneck analysis results
            io.write_csv(bottleneck_results, "reports/bottleneck_analysis.csv")
            
            # Export model predictions for clinical review
            io.write_csv(predictions_df, "outputs/tat_predictions.csv")
            ```
        """
        logger.info(f"Saving healthcare analytics results: {path}")
        
        # Ensure output directory exists for healthcare documentation
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        try:
            # Export with healthcare-optimized formatting  
            # Handle potential index parameter conflict
            if 'index' in kwargs:
                df.to_csv(path, **kwargs)
            else:
                df.to_csv(path, index=False, **kwargs)
            logger.info(f"✅ Healthcare data exported: {len(df):,} records saved to {path}")
            
        except Exception as e:
            logger.error(f"❌ Healthcare data export failed for {path}: {e}")
            raise IOError(f"Healthcare analytics export failed for {path}: {e}") from e

    def _validate_healthcare_dataset(self, df: pd.DataFrame, source_path: str) -> Dict:
        """
        Comprehensive healthcare data validation for clinical quality assurance.
        
        Performs detailed validation of medication preparation dataset ensuring
        clinical data integrity, TAT analysis capability, and workflow optimization
        readiness. Essential for healthcare analytics quality assurance supporting
        healthcare pharmacy operations excellence and patient care improvement.
        
        Validation Components:
        - Clinical column presence validation ensuring complete workflow analysis capability
        - TAT threshold compliance assessment supporting 60-minute quality requirements
        - Temporal data integrity validation ensuring accurate bottleneck identification
        - Missing data pattern analysis with clinical workflow impact assessment
        - Healthcare data distribution validation supporting robust predictive modeling
        
        Args:
            df: Healthcare dataset for comprehensive clinical validation
            source_path: Dataset source for validation reporting and audit trail
        
        Returns:
            Dict: Comprehensive validation results including clinical insights,
            data quality assessment, and healthcare analytics readiness evaluation.
        
        Raises:
            HealthcareDataError: Critical validation failure affecting clinical analysis

        """
        validation_results = {
            'source_path': source_path,
            'validation_timestamp': pd.Timestamp.now(),
            'dataset_shape': df.shape,
            'validation_status': 'PASSED'
        }
        
        # Validate essential healthcare columns for workflow analysis
        missing_required = self.required_healthcare_columns - set(df.columns)
        if missing_required:
            logger.warning(f"Missing required healthcare columns: {missing_required}")
            validation_results['missing_columns'] = list(missing_required)
            validation_results['validation_status'] = 'WARNING'
        
        # TAT threshold compliance analysis for clinical quality assessment
        if 'TAT_minutes' in df.columns:
            tat_series = pd.to_numeric(df['TAT_minutes'], errors='coerce')
            validation_results['tat_analysis'] = {
                'mean_tat': tat_series.mean(),
                'median_tat': tat_series.median(),
                'violation_rate': (tat_series > 60).mean(),
                'missing_tat_pct': tat_series.isna().mean()
            }
            
            # Clinical quality assessment
            if validation_results['tat_analysis']['violation_rate'] > 0.5:
                logger.warning("⚠️  High TAT threshold violation rate detected (>50%)")
                validation_results['quality_alerts'] = ['high_tat_violations']
        
        # Temporal data integrity validation for workflow analysis
        timestamp_cols = [col for col in df.columns if 'time' in col.lower()]
        if timestamp_cols:
            validation_results['temporal_validation'] = {}
            for col in timestamp_cols:
                missing_pct = df[col].isna().mean()
                validation_results['temporal_validation'][col] = {
                    'missing_percentage': missing_pct
                }
                
                if missing_pct > 0.2:  # >20% missing timestamps
                    logger.warning(f"⚠️  High missing data in {col}: {missing_pct:.1%}")
                    validation_results['validation_status'] = 'WARNING'
        
        # Healthcare data distribution validation
        if 'age' in df.columns:
            age_series = pd.to_numeric(df['age'], errors='coerce')
            age_range = (age_series.min(), age_series.max())
            if age_range[0] < 0 or age_range[1] > 120:
                logger.warning(f"⚠️  Unusual age range detected: {age_range}")
                validation_results['data_quality_alerts'] = ['unusual_age_range']
        
        return validation_results

    def validate_tat_data(self, df: pd.DataFrame) -> Dict:
        """
        Focused TAT data validation for medication preparation workflow analysis.
        
        Provides specialized validation focusing on TAT prediction readiness and
        bottleneck analysis capability. Essential for healthcare pharmacy
        workflow optimization ensuring dataset suitability for comprehensive
        TAT analysis and 60-minute threshold compliance assessment.
        
        TAT Validation Focus:
        - TAT calculation validation ensuring accurate workflow timing analysis
        - Step-to-step delay validation supporting bottleneck identification accuracy
        - Clinical threshold compliance assessment for quality monitoring requirements
        - Workflow sequence integrity validation ensuring temporal analysis capability
        - Predictive modeling readiness assessment for production deployment preparation
        
        Args:
            df: Healthcare dataset for focused TAT validation and analysis readiness
        
        Returns:
            Dict: Comprehensive TAT validation results including clinical insights
            and workflow optimization readiness assessment for pharmacy operations.
        
        Example:
            For medication preparation TAT assessment:
            ```python
            io = DataIO()
            df = io.read_csv("dataset.csv")
            
            # Focused TAT validation for workflow optimization
            tat_validation = io.validate_tat_data(df)
            print(f"TAT compliance: {tat_validation['threshold_compliance']:.1%}")
            ```
        """
        logger.info("Performing focused TAT data validation for workflow optimization...")
        
        validation_report = {
            'validation_type': 'TAT_FOCUSED',
            'timestamp': pd.Timestamp.now(),
            'validation_status': 'PASSED'
        }
        
        # TAT calculation and distribution analysis
        if 'TAT_minutes' in df.columns:
            tat_data = pd.to_numeric(df['TAT_minutes'], errors='coerce')
            
            validation_report['tat_statistics'] = {
                'count': len(tat_data.dropna()),
                'mean': tat_data.mean(),
                'median': tat_data.median(),
                'std': tat_data.std(),
                'min': tat_data.min(),
                'max': tat_data.max(),
                'q25': tat_data.quantile(0.25),
                'q75': tat_data.quantile(0.75)
            }
            
            # Clinical threshold compliance assessment
            validation_report['threshold_compliance'] = (tat_data <= 60).mean()
            validation_report['critical_delays'] = (tat_data > 90).mean()
            validation_report['missing_tat_rate'] = tat_data.isna().mean()
            
            # Clinical quality insights
            if validation_report['threshold_compliance'] < 0.5:
                logger.warning("🚨 Critical: <50% TAT threshold compliance - urgent workflow review needed")
                validation_report['clinical_priority'] = 'URGENT'
            elif validation_report['threshold_compliance'] < 0.8:
                logger.warning("⚠️  TAT compliance below target - workflow optimization recommended")
                validation_report['clinical_priority'] = 'HIGH'
            else:
                validation_report['clinical_priority'] = 'STANDARD'
        
        # Workflow sequence validation for bottleneck analysis
        workflow_steps = [
            'doctor_order_time', 'nurse_validation_time', 'prep_complete_time',
            'second_validation_time', 'floor_dispatch_time', 'patient_infusion_time'
        ]
        
        present_steps = [col for col in workflow_steps if col in df.columns]
        validation_report['workflow_completeness'] = len(present_steps) / len(workflow_steps)
        
        if validation_report['workflow_completeness'] < 0.8:
            logger.warning("⚠️  Incomplete workflow steps may limit bottleneck analysis")
            validation_report['validation_status'] = 'WARNING'
        
        logger.info(f"✅ TAT validation complete - Priority: {validation_report.get('clinical_priority', 'N/A')}")
        return validation_report

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive healthcare dataset summary for clinical stakeholders.
        
        Provides detailed dataset overview optimized for healthcare analytics and
        clinical decision-making. Essential for healthcare pharmacy workflow
        optimization supporting medication preparation efficiency assessment and
        bottleneck identification through comprehensive data understanding.
        
        Summary Components:
        - Healthcare dataset overview with clinical context and operational insights
        - TAT distribution analysis supporting 60-minute threshold compliance assessment
        - Missing data patterns with clinical workflow impact evaluation
        - Categorical distribution analysis for demographic and operational understanding
        - Temporal coverage assessment supporting comprehensive workflow analysis
        
        Args:
            df: Healthcare dataset for comprehensive summary generation
        
        Returns:
            Dict: Detailed dataset summary with clinical insights supporting
            pharmacy workflow optimization and healthcare analytics decision-making.
        
        Example:
            For comprehensive dataset assessment:
            ```python
            summary = io.get_data_summary(df)
            print(f"Orders: {summary['total_orders']:,}")
            print(f"TAT compliance: {summary['tat_compliance']:.1%}")
            ```
        """
        logger.info("Generating comprehensive healthcare dataset summary...")
        
        summary = {
            'dataset_overview': {
                'total_orders': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
                'date_range': self._get_temporal_coverage(df)
            }
        }
        
        # TAT analysis for clinical quality assessment
        if 'TAT_minutes' in df.columns:
            tat_data = pd.to_numeric(df['TAT_minutes'], errors='coerce')
            summary['tat_analysis'] = {
                'mean_tat': tat_data.mean(),
                'median_tat': tat_data.median(),
                'tat_compliance': (tat_data <= 60).mean(),
                'critical_delays_pct': (tat_data > 90).mean(),
                'missing_tat_pct': tat_data.isna().mean()
            }
        
        # Clinical and operational categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        summary['categorical_distribution'] = {}
        
        for col in categorical_cols[:10]:  # Top 10 categorical columns
            value_counts = df[col].value_counts()
            summary['categorical_distribution'][col] = {
                'unique_values': len(value_counts),
                'top_categories': value_counts.head(5).to_dict(),
                'missing_pct': df[col].isna().mean()
            }
        
        # Missing data pattern analysis for clinical workflow impact
        missing_summary = df.isnull().sum()
        summary['missing_data_analysis'] = {
            'columns_with_missing': (missing_summary > 0).sum(),
            'highest_missing_columns': missing_summary.nlargest(5).to_dict(),
            'overall_missing_rate': df.isnull().mean().mean()
        }
        
        logger.info("✅ Healthcare dataset summary generated successfully")
        return summary
    
    def _get_temporal_coverage(self, df: pd.DataFrame) -> Optional[Dict]:
        """Extract temporal coverage from healthcare dataset for workflow analysis."""
        if 'doctor_order_time' in df.columns:
            try:
                order_times = pd.to_datetime(df['doctor_order_time'], errors='coerce').dropna()
                if len(order_times) > 0:
                    return {
                        'start_date': order_times.min().strftime('%Y-%m-%d'),
                        'end_date': order_times.max().strftime('%Y-%m-%d'),
                        'total_days': (order_times.max() - order_times.min()).days
                    }
            except Exception:
                pass
        return None


class HealthcareDataError(Exception):
    """Custom exception for healthcare data validation failures."""
    pass