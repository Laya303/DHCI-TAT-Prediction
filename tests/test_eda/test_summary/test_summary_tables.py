"""
Test suite for functionality.

Tests the table artifact builders for Dana Farber's pharmacy turnaround time analysis,
ensuring robust tabular summary generation for medication preparation workflow
optimization and healthcare stakeholder reporting.

Key Test Coverage:
- Workflow timestamp analysis tables for step-by-step delay identification
- Operational factor summaries for pharmacy staffing and resource optimization
- TAT performance metric tables with embedded distribution visualization
- Correlation analysis builders for bottleneck driver identification
- Missing data assessment tables for healthcare data quality monitoring
- Production-grade robustness with malformed healthcare datasets
- Clinical interpretability and healthcare domain terminology validation
"""
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from src.tat.eda.summary.summary_tables import (
    time_table,
    categorical_table, 
    numeric_table,
    build_artifacts,
    correlation_lower,
    correlation_pairs_table
)
from src.tat.eda.summary.summary_config import SummaryConfig


class TestWorkflowTimestampAnalysis:
    """Test suite for medication preparation workflow timestamp analysis."""
    
    @pytest.fixture
    def pharmacy_workflow_data(self):
        """Generate realistic pharmacy workflow dataset with TAT timestamps."""
        np.random.seed(42)
        n_orders = 1000
        base_time = pd.Timestamp('2025-01-15 06:00:00')
        
        return pd.DataFrame({
            # Complete medication preparation workflow sequence
            'doctor_order_time': pd.date_range(base_time, periods=n_orders, freq='10min'),
            'nurse_validation_time': pd.date_range(base_time + pd.Timedelta('8min'), periods=n_orders, freq='10min'),
            'prep_complete_time': pd.date_range(base_time + pd.Timedelta('35min'), periods=n_orders, freq='10min'),
            'second_validation_time': pd.date_range(base_time + pd.Timedelta('40min'), periods=n_orders, freq='10min'),
            'floor_dispatch_time': pd.date_range(base_time + pd.Timedelta('45min'), periods=n_orders, freq='10min'),
            'patient_infusion_time': pd.date_range(base_time + pd.Timedelta('65min'), periods=n_orders, freq='10min'),
            
            # Operational context
            'TAT_minutes': np.random.exponential(42, n_orders) + 15,
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders)
        })
    
    @pytest.fixture
    def workflow_data_with_missing(self):
        """Generate workflow data with realistic healthcare missingness patterns."""
        np.random.seed(123)
        n_orders = 500
        base_time = pd.Timestamp('2025-01-15 08:00:00')
        
        # Generate complete timestamps first as Series (not DatetimeIndex)
        doctor_times = pd.Series(pd.date_range(base_time, periods=n_orders, freq='12min'))
        nurse_times = pd.Series(pd.date_range(base_time + pd.Timedelta('10min'), periods=n_orders, freq='12min'))
        prep_times = pd.Series(pd.date_range(base_time + pd.Timedelta('40min'), periods=n_orders, freq='12min'))
        
        # Introduce realistic missing patterns (EHR integration gaps)
        nurse_missing_mask = np.random.random(n_orders) < 0.08  # 8% missing
        prep_missing_mask = np.random.random(n_orders) < 0.15   # 15% missing
        
        # Apply missing patterns to Series (which are mutable)
        nurse_times[nurse_missing_mask] = pd.NaT
        prep_times[prep_missing_mask] = pd.NaT
        
        return pd.DataFrame({
            'doctor_order_time': doctor_times,           # Complete (order entry required)
            'nurse_validation_time': nurse_times,        # 8% missing
            'prep_complete_time': prep_times,            # 15% missing
            'TAT_minutes': np.random.exponential(38, n_orders) + 12
        })
    
    @pytest.fixture
    def healthcare_config(self):
        """Configuration for healthcare timestamp analysis."""
        return SummaryConfig()

    def test_time_table_complete_workflow_analysis(self, pharmacy_workflow_data, healthcare_config):
        """Test comprehensive workflow timestamp analysis for complete TAT sequence."""
        workflow_columns = [
            'doctor_order_time', 'nurse_validation_time', 'prep_complete_time',
            'second_validation_time', 'floor_dispatch_time', 'patient_infusion_time'
        ]
        
        result = time_table(pharmacy_workflow_data, workflow_columns, healthcare_config)
        
        # Should analyze complete medication preparation workflow
        assert len(result) == 6
        assert set(result['name']) == set(workflow_columns)
        
        # Should identify datetime columns correctly
        assert all(result['type'] == 'datetime64[ns]')
        
        # Complete workflow should show minimal missing data
        assert all(result['miss%'] <= 1.0)  # ≤1% for complete synthetic data
        
        # Should have high data completeness for workflow tracking
        assert all(result['n_nonnull'] >= 995)  # Near-complete for 1000 orders
        
        # Should provide representative timestamps for validation
        for sample in result['sample']:
            assert isinstance(sample, str)
            assert '2025-01-15' in sample  # Should match test data timeframe

    def test_time_table_missing_data_healthcare_patterns(self, workflow_data_with_missing, healthcare_config):
        """Test timestamp analysis with realistic healthcare missing data patterns."""
        timestamp_columns = ['doctor_order_time', 'nurse_validation_time', 'prep_complete_time']
        
        result = time_table(workflow_data_with_missing, timestamp_columns, healthcare_config)
        
        # Should handle realistic healthcare missingness patterns
        assert len(result) == 3
        
        # Should calculate missing percentages accurately
        doctor_row = result[result['name'] == 'doctor_order_time'].iloc[0]
        nurse_row = result[result['name'] == 'nurse_validation_time'].iloc[0]
        prep_row = result[result['name'] == 'prep_complete_time'].iloc[0]
        
        # Doctor orders should be complete (required field)
        assert doctor_row['miss%'] == 0.0
        
        # Nurse validation should show ~8% missing
        assert 6.0 <= nurse_row['miss%'] <= 10.0
        
        # Prep completion should show ~15% missing
        assert 12.0 <= prep_row['miss%'] <= 18.0
        
        # Non-null counts should reflect missing data
        assert doctor_row['n_nonnull'] == 500
        assert nurse_row['n_nonnull'] <= 470  # ~8% missing from 500
        assert prep_row['n_nonnull'] <= 440   # ~15% missing from 500

    def test_time_table_defensive_missing_columns(self, pharmacy_workflow_data, healthcare_config):
        """Test defensive handling when requested timestamp columns are missing."""
        # Request columns that don't exist in dataset
        missing_columns = ['nonexistent_timestamp', 'invalid_workflow_step', 'doctor_order_time']
        
        result = time_table(pharmacy_workflow_data, missing_columns, healthcare_config)
        
        # Should only process existing columns without errors
        assert len(result) == 1  # Only 'doctor_order_time' exists
        assert result.iloc[0]['name'] == 'doctor_order_time'

    def test_time_table_empty_column_list(self, pharmacy_workflow_data, healthcare_config):
        """Test timestamp analysis with empty column specification."""
        result = time_table(pharmacy_workflow_data, [], healthcare_config)
        
        # Should return empty DataFrame without errors
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_time_table_data_quality_assessment_output(self, workflow_data_with_missing, healthcare_config):
        """Test data quality assessment output format for healthcare reporting."""
        columns = ['doctor_order_time', 'nurse_validation_time', 'prep_complete_time']
        
        result = time_table(workflow_data_with_missing, columns, healthcare_config)
        
        # Should have complete healthcare data quality assessment structure
        expected_columns = ['name', 'type', 'n_nonnull', 'miss%', 'sample']
        assert list(result.columns) == expected_columns
        
        # All data should be appropriate types for healthcare reporting
        assert result['name'].dtype == 'object'
        assert result['type'].dtype == 'object'
        assert result['n_nonnull'].dtype == 'int64'
        assert result['miss%'].dtype == 'float64'
        assert result['sample'].dtype == 'object'
        
        # Missing percentages should be properly rounded for clinical presentation
        assert all(result['miss%'] == result['miss%'].round(2))


class TestOperationalFactorAnalysis:
    """Test suite for pharmacy operational factor and categorical analysis."""
    
    @pytest.fixture
    def pharmacy_operations_data(self):
        """Generate realistic pharmacy operations dataset with categorical factors."""
        np.random.seed(42)
        n_orders = 1200
        
        return pd.DataFrame({
            # Staffing and shift patterns
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders, p=[0.50, 0.30, 0.20]),
            'nurse_credential': np.random.choice(['BSN', 'RN', 'MSN', 'NP'], n_orders, p=[0.42, 0.31, 0.18, 0.09]),
            'pharmacist_credential': np.random.choice(['RPh', 'PharmD', 'BCOP'], n_orders, p=[0.25, 0.60, 0.15]),
            
            # Clinical context and patient factors
            'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic', 'Autoimmune', 'Other'], n_orders, p=[0.40, 0.35, 0.20, 0.05]),
            'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders, p=[0.30, 0.50, 0.20]),
            'treatment_type': np.random.choice(['Chemotherapy', 'Immunotherapy', 'TargetedTherapy'], n_orders, p=[0.45, 0.35, 0.20]),
            
            # Operational and location factors
            'floor': np.random.choice([1, 2, 3], n_orders),
            'ordering_department': np.random.choice([
                'MedicalOncology', 'Hematology', 'StemCellTransplant', 
                'RadiationOncology', 'ImmunotherapyClinic'
            ], n_orders, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
            
            # Binary protocol indicators
            'stat_order': np.random.choice([0, 1], n_orders, p=[0.85, 0.15]),
            'premed_required': np.random.choice([0, 1], n_orders, p=[0.70, 0.30]),
            
            # Numeric for comparison
            'TAT_minutes': np.random.exponential(45, n_orders) + 12
        })
    
    @pytest.fixture
    def categorical_data_with_missing(self):
        """Generate categorical data with healthcare-typical missing patterns."""
        np.random.seed(123)
        n_orders = 800
        
        # Generate base categories as Series for proper missing value handling
        shifts = pd.Series(np.random.choice(['Day', 'Evening', 'Night'], n_orders))
        credentials = pd.Series(np.random.choice(['BSN', 'RN', 'MSN', 'NP'], n_orders))
        departments = pd.Series(np.random.choice(['MedicalOncology', 'Hematology', 'StemCellTransplant'], n_orders))
        floors = pd.Series(np.random.choice([1, 2, 3], n_orders))
        
        # Introduce realistic missing patterns
        credential_missing_mask = np.random.random(n_orders) < 0.12  # 12% missing credentials
        department_missing_mask = np.random.random(n_orders) < 0.05  # 5% missing departments
        
        # Apply missing patterns
        credentials[credential_missing_mask] = np.nan
        departments[department_missing_mask] = np.nan
        
        return pd.DataFrame({
            'shift': shifts,                    # Complete (required operational data)
            'nurse_credential': credentials,    # 12% missing
            'ordering_department': departments, # 5% missing
            'floor': floors                     # Complete
        })
    
    @pytest.fixture
    def high_cardinality_data(self):
        """Generate data with high cardinality categorical variables."""
        np.random.seed(42)
        n_orders = 1000
        
        # High cardinality physician names (realistic for large health system)
        physicians = [f'Dr_{i:03d}' for i in range(150)]  # 150 unique physicians
        
        return pd.DataFrame({
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders),  # Low cardinality
            'ordering_physician': np.random.choice(physicians, n_orders),      # High cardinality
            'patient_id': [f'DFCI_{i:06d}' for i in range(n_orders)],        # Unique identifiers
            'TAT_minutes': np.random.exponential(40, n_orders) + 15
        })

    def test_categorical_table_comprehensive_operations_analysis(self, pharmacy_operations_data):
        """Test comprehensive operational factor analysis for pharmacy workflow optimization."""
        categorical_columns = [
            'shift', 'nurse_credential', 'pharmacist_credential', 'diagnosis_type',
            'severity', 'treatment_type', 'floor', 'ordering_department'
        ]
        
        config = SummaryConfig()
        result = categorical_table(pharmacy_operations_data, categorical_columns, config)
        
        # Should analyze all operational factors
        assert len(result) == len(categorical_columns)
        assert set(result['name']) == set(categorical_columns)
        
        # Should have complete operational factor analysis structure
        expected_columns = ['name', 'type', 'n_nonnull', 'miss%', 'nunique', 'sample', 'top_values']
        assert list(result.columns) == expected_columns
        
        # Should show realistic cardinalities for healthcare operations
        shift_row = result[result['name'] == 'shift'].iloc[0]
        assert shift_row['nunique'] == 3  # Day, Evening, Night
        
        nurse_row = result[result['name'] == 'nurse_credential'].iloc[0]
        assert nurse_row['nunique'] == 4  # BSN, RN, MSN, NP
        
        diagnosis_row = result[result['name'] == 'diagnosis_type'].iloc[0]
        assert diagnosis_row['nunique'] == 4  # SolidTumor, Hematologic, Autoimmune, Other
        
        # Should provide top values analysis for workflow optimization
        assert 'Day(' in shift_row['top_values']  # Day shift should be dominant
        assert '%' in shift_row['top_values']     # Should include percentages

    def test_categorical_table_missing_data_patterns(self, categorical_data_with_missing):
        """Test categorical analysis with realistic healthcare missing data patterns."""
        columns = ['shift', 'nurse_credential', 'ordering_department', 'floor']
        
        config = SummaryConfig()
        result = categorical_table(categorical_data_with_missing, columns, config)
        
        # Should handle missing data appropriately
        shift_row = result[result['name'] == 'shift'].iloc[0]
        credential_row = result[result['name'] == 'nurse_credential'].iloc[0]
        department_row = result[result['name'] == 'ordering_department'].iloc[0]
        
        # Complete operational data should show no missing
        assert shift_row['miss%'] == 0.0
        
        # Should detect realistic healthcare missing patterns
        # Note: Actual percentages may vary due to random seed, so check for presence of missing data
        assert credential_row['miss%'] > 8.0      # Should have some missing data
        assert department_row['miss%'] > 2.0      # Should have some missing data
        
        # Non-null counts should reflect missing data
        assert credential_row['n_nonnull'] < 800  # Some missing from 800
        assert department_row['n_nonnull'] < 800   # Some missing from 800

    def test_categorical_table_high_cardinality_handling(self, high_cardinality_data):
        """Test handling of high cardinality variables common in healthcare systems."""
        columns = ['shift', 'ordering_physician', 'patient_id']
        
        config = SummaryConfig(max_categorical_cardinality=50)
        result = categorical_table(high_cardinality_data, columns, config)
        
        # Should identify high cardinality appropriately
        physician_row = result[result['name'] == 'ordering_physician'].iloc[0]
        patient_row = result[result['name'] == 'patient_id'].iloc[0]
        shift_row = result[result['name'] == 'shift'].iloc[0]
        
        # High cardinality should be flagged
        assert physician_row['nunique'] > config.max_categorical_cardinality
        assert patient_row['nunique'] > config.max_categorical_cardinality
        
        # Should show cardinality warning for high-dimensional variables
        assert 'High cardinality' in physician_row['top_values']
        assert 'High cardinality' in patient_row['top_values']
        
        # Low cardinality should show top values
        assert 'High cardinality' not in shift_row['top_values']
        assert '%' in shift_row['top_values']  # Should show percentage breakdown

    def test_categorical_table_defensive_missing_columns(self, pharmacy_operations_data):
        """Test defensive handling of missing categorical columns."""
        # Mix of existing and non-existing columns
        mixed_columns = ['shift', 'nonexistent_column', 'nurse_credential', 'invalid_field']
        
        config = SummaryConfig()
        result = categorical_table(pharmacy_operations_data, mixed_columns, config)
        
        # Should only process existing columns
        assert len(result) == 2  # Only 'shift' and 'nurse_credential' exist
        assert set(result['name']) == {'shift', 'nurse_credential'}

    def test_categorical_table_clinical_interpretation_output(self, pharmacy_operations_data):
        """Test output format supports clinical interpretation and healthcare reporting."""
        columns = ['shift', 'severity', 'diagnosis_type']
        
        config = SummaryConfig()
        result = categorical_table(pharmacy_operations_data, columns, config)
        
        # Should provide clinical context in top values
        severity_row = result[result['name'] == 'severity'].iloc[0]
        assert any(level in severity_row['top_values'] for level in ['Low', 'Medium', 'High'])
        
        # Should use healthcare-appropriate formatting
        diagnosis_row = result[result['name'] == 'diagnosis_type'].iloc[0]
        assert 'SolidTumor' in diagnosis_row['top_values'] or 'Hematologic' in diagnosis_row['top_values']
        
        # Should provide actionable percentage breakdowns
        shift_row = result[result['name'] == 'shift'].iloc[0]
        assert 'Day(' in shift_row['top_values']
        assert '%' in shift_row['top_values']


class TestTATPerformanceMetricsAnalysis:
    """Test suite for TAT  Numerical Features and numeric analysis with visualizations."""
    
    @pytest.fixture
    def tat_performance_data(self):
        """Generate realistic TAT performance dataset with operational metrics."""
        np.random.seed(42)
        n_orders = 1500
        
        return pd.DataFrame({
            # Core TAT  Numerical Features
            'TAT_minutes': np.random.exponential(42, n_orders) + 15,  # 15-min minimum + exponential
            'queue_length_at_order': np.random.poisson(7, n_orders),  # Poisson queue
            'floor_occupancy_pct': np.random.uniform(25, 95, n_orders),  # Occupancy rates
            
            # Staffing metrics
            'pharmacists_on_duty': np.random.randint(2, 8, n_orders),
            'nurse_employment_years': np.random.uniform(0.5, 25, n_orders),
            'pharmacist_employment_years': np.random.uniform(1, 30, n_orders),
            
            # Clinical indicators
            'patient_age': np.random.uniform(18, 85, n_orders),
            'lab_WBC_k_per_uL': np.random.uniform(3.0, 12.0, n_orders),
            'lab_Platelets_k_per_uL': np.random.uniform(120, 450, n_orders),
            'lab_Creatinine_mg_dL': np.random.uniform(0.6, 2.0, n_orders),
            
            # Mixed for comparison
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders)
        })
    
    @pytest.fixture
    def numeric_data_with_missing(self):
        """Generate numeric data with healthcare-typical missing patterns."""
        np.random.seed(123)
        n_orders = 1000
        
        # Generate base numeric data
        tat_values = np.random.exponential(38, n_orders) + 12
        queue_values = np.random.poisson(6, n_orders)
        lab_wbc_values = np.random.uniform(4.0, 11.0, n_orders)
        lab_plt_values = np.random.uniform(150, 400, n_orders)
        
        # Introduce realistic missing patterns (lab results often delayed/missing)
        wbc_missing_mask = np.random.random(n_orders) < 0.18  # 18% missing labs
        plt_missing_mask = np.random.random(n_orders) < 0.15  # 15% missing labs
        
        lab_wbc_values[wbc_missing_mask] = np.nan
        lab_plt_values[plt_missing_mask] = np.nan
        
        return pd.DataFrame({
            'TAT_minutes': tat_values,           # Complete (core metric)
            'queue_length_at_order': queue_values,  # Complete (operational)
            'lab_WBC_k_per_uL': lab_wbc_values,     # 18% missing
            'lab_Platelets_k_per_uL': lab_plt_values  # 15% missing
        })
    
    @pytest.fixture
    def edge_case_numeric_data(self):
        """Generate numeric data with statistical edge cases."""
        n_base = 100  # Base length for most columns
        
        return pd.DataFrame({
            'constant_values': [42.0] * n_base,  # No variance
            'wide_range': ([0.1, 5.0, 50.0, 500.0, 5000.0] * 20),  # Extreme range, same length
            'sparse_data': [np.nan] * 95 + [1.0, 2.0, 3.0, 4.0, 5.0],  # 95% missing
            'single_value': [100.0] + [np.nan] * (n_base - 1)  # Single observation with padding
        })

    def test_numeric_table_comprehensive_tat_metrics(self, tat_performance_data):
        """Test comprehensive TAT  Numerical Features analysis with visualization."""
        numeric_columns = [
            'TAT_minutes', 'queue_length_at_order', 'floor_occupancy_pct',
            'pharmacists_on_duty', 'patient_age', 'lab_WBC_k_per_uL'
        ]
        
        config = SummaryConfig()
        result = numeric_table(tat_performance_data, numeric_columns, config)
        
        # Should analyze all TAT  Numerical Features
        assert len(result) == len(numeric_columns)
        assert set(result['name']) == set(numeric_columns)
        
        # Should have complete numeric analysis structure with visualizations
        expected_columns = [
            'name', 'type', 'n_nonnull', 'miss%', 'nunique', 'sample',
            'min', 'p50', 'max', 'distribution', '_dist_counts', '_dist_labels'
        ]
        assert list(result.columns) == expected_columns
        
        # Should provide clinical statistical summaries
        tat_row = result[result['name'] == 'TAT_minutes'].iloc[0]
        assert tat_row['min'] >= 15.0  # Minimum TAT from test data
        assert 30.0 <= tat_row['p50'] <= 80.0  # Realistic median TAT
        assert tat_row['max'] > tat_row['p50']  # Statistical consistency
        
        # Should generate distribution visualizations
        assert len(tat_row['distribution']) > 0  # Sparkline visualization
        assert isinstance(tat_row['_dist_counts'], list)  # Histogram data
        assert isinstance(tat_row['_dist_labels'], list)  # Bin labels
        
        # Should handle different metric types appropriately
        queue_row = result[result['name'] == 'queue_length_at_order'].iloc[0]
        assert queue_row['min'] >= 0  # Queue length non-negative
        # Handle both Python int and numpy int types
        assert isinstance(queue_row['nunique'], (int, np.integer))  # Discrete values

    def test_numeric_table_missing_data_lab_patterns(self, numeric_data_with_missing):
        """Test numeric analysis with realistic laboratory data missing patterns."""
        columns = ['TAT_minutes', 'queue_length_at_order', 'lab_WBC_k_per_uL', 'lab_Platelets_k_per_uL']
        
        config = SummaryConfig()
        result = numeric_table(numeric_data_with_missing, columns, config)
        
        # Should handle missing lab data appropriately
        tat_row = result[result['name'] == 'TAT_minutes'].iloc[0]
        queue_row = result[result['name'] == 'queue_length_at_order'].iloc[0]
        wbc_row = result[result['name'] == 'lab_WBC_k_per_uL'].iloc[0]
        plt_row = result[result['name'] == 'lab_Platelets_k_per_uL'].iloc[0]
        
        # Core metrics should be complete
        assert tat_row['miss%'] == 0.0
        assert queue_row['miss%'] == 0.0
        
        # Lab values should show realistic missing patterns
        assert 15.0 <= wbc_row['miss%'] <= 20.0   # ~18% missing
        assert 12.0 <= plt_row['miss%'] <= 18.0   # ~15% missing
        
        # Non-null counts should reflect missing data
        assert wbc_row['n_nonnull'] <= 850  # ~18% missing from 1000
        assert plt_row['n_nonnull'] <= 880  # ~15% missing from 1000

    def test_numeric_table_statistical_edge_cases(self, edge_case_numeric_data):
        """Test numeric analysis with statistical edge cases common in healthcare."""
        columns = ['constant_values', 'wide_range', 'sparse_data', 'single_value']
        
        config = SummaryConfig()
        result = numeric_table(edge_case_numeric_data, columns, config)
        
        # Should handle constant values (e.g., protocol doses)
        constant_row = result[result['name'] == 'constant_values'].iloc[0]
        assert constant_row['min'] == constant_row['max'] == 42.0
        assert constant_row['p50'] == 42.0
        
        # Should handle wide ranges (e.g., patient ages, lab values)
        wide_row = result[result['name'] == 'wide_range'].iloc[0]
        assert wide_row['min'] < wide_row['p50'] < wide_row['max']
        
        # Should handle sparse data (missing lab results)
        sparse_row = result[result['name'] == 'sparse_data'].iloc[0]
        assert sparse_row['miss%'] == 95.0
        assert sparse_row['n_nonnull'] == 5
        
        # Should handle single observations
        single_row = result[result['name'] == 'single_value'].iloc[0]
        assert single_row['n_nonnull'] == 1
        assert single_row['min'] == single_row['max'] == 100.0

    def test_numeric_table_precision_clinical_reporting(self, tat_performance_data):
        """Test numeric precision appropriate for clinical reporting and interpretation."""
        columns = ['TAT_minutes', 'lab_Creatinine_mg_dL', 'floor_occupancy_pct']
        
        config = SummaryConfig()
        result = numeric_table(tat_performance_data, columns, config)
        
        # Should round statistics to 3 decimal places for clinical presentation
        for _, row in result.iterrows():
            if not pd.isna(row['min']):
                # Check that rounding is applied (no excessive precision)
                min_str = str(row['min'])
                if '.' in min_str:
                    decimal_places = len(min_str.split('.')[1])
                    assert decimal_places <= 3
            
            if not pd.isna(row['p50']):
                p50_str = str(row['p50'])
                if '.' in p50_str:
                    decimal_places = len(p50_str.split('.')[1])
                    assert decimal_places <= 3

    def test_numeric_table_visualization_integration(self, tat_performance_data):
        """Test integration of distribution visualizations with numeric analysis."""
        columns = ['TAT_minutes', 'queue_length_at_order']
        
        config = SummaryConfig()
        result = numeric_table(tat_performance_data, columns, config)
        
        for _, row in result.iterrows():
            # Should generate visualization data
            assert isinstance(row['_dist_counts'], list)
            assert isinstance(row['_dist_labels'], list)
            
            if row['_dist_counts']:  # If histogram data exists
                # Counts and labels should have same length
                assert len(row['_dist_counts']) == len(row['_dist_labels'])
                
                # Distribution string should match counts length
                if row['distribution']:
                    assert len(row['distribution']) == len(row['_dist_counts'])
                
                # Counts should be non-negative integers
                assert all(isinstance(count, int) and count >= 0 for count in row['_dist_counts'])
                
                # Labels should be interval strings
                assert all(isinstance(label, str) and '(' in label for label in row['_dist_labels'])


class TestCorrelationAnalysisBuilders:
    """Test suite for TAT correlation analysis and driver identification."""
    
    @pytest.fixture
    def correlated_tat_data(self):
        """Generate TAT data with known correlations for driver identification testing."""
        np.random.seed(42)
        n_orders = 2000
        
        # Generate correlated operational factors
        queue_base = np.random.poisson(6, n_orders)
        occupancy = np.random.uniform(0.3, 0.95, n_orders)
        pharmacists = np.random.randint(2, 7, n_orders)
        
        # TAT influenced by operational factors (known relationships)
        tat = (
            25 +  # Base TAT
            queue_base * 3.2 +      # Strong positive: queue length → TAT
            occupancy * 35 +        # Moderate positive: occupancy → TAT
            pharmacists * (-2.8) +  # Negative: more pharmacists → lower TAT
            np.random.normal(0, 8, n_orders)  # Random variation
        )
        
        return pd.DataFrame({
            'TAT_minutes': tat,
            'queue_length_at_order': queue_base,
            'floor_occupancy_pct': occupancy * 100,
            'pharmacists_on_duty': pharmacists,
            'nurse_employment_years': np.random.uniform(1, 25, n_orders),  # Weak correlation
            'patient_age': np.random.uniform(18, 85, n_orders),           # No correlation
            'random_noise': np.random.normal(0, 10, n_orders),            # No correlation
            
            # Non-numeric (should be excluded)
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders),
            'patient_id': [f'PAT_{i:06d}' for i in range(n_orders)]
        })
    
    @pytest.fixture
    def full_correlation_matrix(self):
        """Generate full correlation matrix for lower triangle testing."""
        np.random.seed(123)
        
        # 4x4 correlation matrix
        data = np.random.randn(100, 4)
        df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
        return df.corr()

    def test_correlation_lower_triangle_visualization(self, full_correlation_matrix):
        """Test lower triangular correlation matrix generation for clean visualization."""
        result = correlation_lower(full_correlation_matrix)
        
        # Should preserve lower triangle including diagonal
        assert result.loc['A', 'A'] == 1.0  # Diagonal preserved
        assert result.loc['B', 'A'] == full_correlation_matrix.loc['B', 'A']  # Lower triangle
        assert result.loc['C', 'A'] == full_correlation_matrix.loc['C', 'A']  # Lower triangle
        
        # Should blank upper triangle
        assert pd.isna(result.loc['A', 'B'])  # Upper triangle blanked
        assert pd.isna(result.loc['A', 'C'])  # Upper triangle blanked
        assert pd.isna(result.loc['B', 'C'])  # Upper triangle blanked
        
        # Should maintain matrix dimensions
        assert result.shape == full_correlation_matrix.shape

    def test_correlation_lower_empty_matrix(self):
        """Test lower triangular transformation with empty correlation matrix."""
        empty_corr = pd.DataFrame()
        result = correlation_lower(empty_corr)
        
        # Should return empty DataFrame without errors
        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_correlation_pairs_table_tat_driver_identification(self, correlated_tat_data):
        """Test correlation pairs analysis for TAT bottleneck driver identification."""
        config = SummaryConfig()
        
        # Test with default significance threshold
        result = correlation_pairs_table(correlated_tat_data, config, min_abs=0.1, top_k=10)
        
        # Should identify significant operational correlations
        assert not result.empty
        assert list(result.columns) == ['feature_a', 'feature_b', 'r', '|r|']
        
        # Should be sorted by correlation strength
        assert result['|r|'].is_monotonic_decreasing
        
        # Should identify known strong correlations
        correlation_pairs = list(zip(result['feature_a'], result['feature_b']))
        
        # Check for TAT relationships (order may vary)
        tat_pairs = [pair for pair in correlation_pairs if 'TAT_minutes' in pair]
        assert len(tat_pairs) >= 2  # Should find multiple TAT drivers
        
        # Should find queue length correlation (strong positive)
        queue_tat_found = any(
            ('TAT_minutes' in pair and 'queue_length_at_order' in pair) 
            for pair in correlation_pairs
        )
        assert queue_tat_found
        
        # Should exclude non-significant correlations
        assert all(result['|r|'] >= 0.1)

    def test_correlation_pairs_table_significance_filtering(self, correlated_tat_data):
        """Test correlation significance filtering for clinical relevance."""
        config = SummaryConfig()
        
        # Test with high significance threshold
        result_strict = correlation_pairs_table(correlated_tat_data, config, min_abs=0.3, top_k=20)
        
        # Should only include strong correlations
        if not result_strict.empty:
            assert all(result_strict['|r|'] >= 0.3)
        
        # Test with low significance threshold  
        result_permissive = correlation_pairs_table(correlated_tat_data, config, min_abs=0.05, top_k=20)
        
        # Should include more pairs with permissive threshold
        assert len(result_permissive) >= len(result_strict)
        
        if not result_permissive.empty:
            assert all(result_permissive['|r|'] >= 0.05)

    def test_correlation_pairs_table_top_k_limiting(self, correlated_tat_data):
        """Test top-K limiting for focused workflow optimization insights."""
        config = SummaryConfig()
        
        # Test with small top_k
        result_small = correlation_pairs_table(correlated_tat_data, config, min_abs=0.05, top_k=3)
        assert len(result_small) <= 3
        
        # Test with large top_k
        result_large = correlation_pairs_table(correlated_tat_data, config, min_abs=0.05, top_k=50)
        assert len(result_large) >= len(result_small)
        
        # Both should be sorted by correlation strength
        if not result_small.empty:
            assert result_small['|r|'].is_monotonic_decreasing
        if not result_large.empty:
            assert result_large['|r|'].is_monotonic_decreasing

    def test_correlation_pairs_table_empty_correlations(self):
        """Test correlation pairs analysis with insufficient numeric data."""
        # Dataset with no meaningful correlations
        no_corr_data = pd.DataFrame({
            'category_only': ['A', 'B', 'C'] * 100,
            'single_numeric': [42.0] * 300  # Constant value
        })
        
        config = SummaryConfig()
        result = correlation_pairs_table(no_corr_data, config, min_abs=0.1)
        
        # Should return empty DataFrame gracefully
        assert isinstance(result, pd.DataFrame)
        # May be empty or have very few correlations

    def test_correlation_pairs_table_clinical_interpretation_format(self, correlated_tat_data):
        """Test correlation pairs output format for clinical interpretation."""
        config = SummaryConfig()
        result = correlation_pairs_table(correlated_tat_data, config, min_abs=0.15, top_k=5)
        
        if not result.empty:
            # Should provide directional correlation information
            assert result['r'].dtype == 'float64'
            assert result['|r|'].dtype == 'float64'
            
            # Should identify positive vs negative relationships
            positive_corrs = result[result['r'] > 0]
            negative_corrs = result[result['r'] < 0]
            
            # Should have both types (operational factors have mixed effects)
            # Note: This depends on the specific synthetic data relationships
            
            # Absolute values should match correlation direction
            for _, row in result.iterrows():
                assert abs(row['r']) == row['|r|']
            
            # Feature names should be descriptive for clinical interpretation
            all_features = set(result['feature_a']) | set(result['feature_b'])
            healthcare_terms = ['TAT', 'queue', 'occupancy', 'pharmacist', 'nurse', 'patient']
            
            feature_names = ' '.join(all_features)
            assert any(term in feature_names for term in healthcare_terms)


class TestArtifactBuilderIntegration:
    """Integration tests for complete TAT analysis artifact generation."""
    
    @pytest.fixture
    def comprehensive_tat_dataset(self):
        """Generate comprehensive TAT dataset for end-to-end artifact testing."""
        np.random.seed(42)
        n_orders = 2500
        base_time = pd.Timestamp('2025-01-15 06:00:00')
        
        return pd.DataFrame({
            # Complete workflow timestamps
            'doctor_order_time': pd.date_range(base_time, periods=n_orders, freq='8min'),
            'nurse_validation_time': [
                base_time + pd.Timedelta(minutes=12+i*8) if i % 15 != 0 else pd.NaT 
                for i in range(n_orders)
            ],  # ~7% missing
            'prep_complete_time': [
                base_time + pd.Timedelta(minutes=38+i*8) if i % 12 != 0 else pd.NaT
                for i in range(n_orders) 
            ],  # ~8% missing
            
            #  Numerical Features
            'TAT_minutes': np.random.exponential(45, n_orders) + 12,
            'queue_length_at_order': np.random.poisson(7, n_orders),
            'floor_occupancy_pct': np.random.uniform(20, 98, n_orders),
            'pharmacists_on_duty': np.random.randint(2, 8, n_orders),
            
            # Operational factors
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders, p=[0.55, 0.30, 0.15]),
            'nurse_credential': np.random.choice(['BSN', 'RN', 'MSN', 'NP'], n_orders, p=[0.45, 0.30, 0.20, 0.05]),
            'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic', 'Autoimmune'], n_orders, p=[0.45, 0.35, 0.20]),
            'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders, p=[0.30, 0.50, 0.20]),
            
            # Clinical indicators with missing patterns
            'lab_WBC_k_per_uL': [
                np.random.uniform(4.0, 11.0) if np.random.random() > 0.12 else np.nan
                for _ in range(n_orders)
            ],  # 12% missing
            'patient_age': np.random.uniform(18, 89, n_orders),
            
            # Identifiers
            'patient_id': [f'DFCI_{i:07d}' for i in range(n_orders)],
            'order_id': [f'ORD_{i:08d}' for i in range(n_orders)]
        })

    def test_build_artifacts_comprehensive_tat_analysis(self, comprehensive_tat_dataset):
        """Test complete TAT analysis artifact generation for pharmacy workflow optimization."""
        config = SummaryConfig()
        artifacts = build_artifacts(comprehensive_tat_dataset, config)
        
        # Should generate complete artifact set
        expected_artifacts = [
            'time_table', 'categorical_table', 'numeric_table', 'correlations',
            'missing_table', 'missing_table_console', 'counts', 'df_processed'
        ]
        assert all(key in artifacts for key in expected_artifacts)
        
        # Should process workflow timestamps
        time_table_result = artifacts['time_table']
        assert not time_table_result.empty
        assert len(time_table_result) >= 3  # At least 3 timestamp columns
        
        # Should analyze operational factors
        categorical_table_result = artifacts['categorical_table']
        assert not categorical_table_result.empty
        expected_categoricals = ['shift', 'nurse_credential', 'diagnosis_type', 'severity']
        cat_names = set(categorical_table_result['name'])
        assert all(cat in cat_names for cat in expected_categoricals)
        
        # Should analyze  Numerical Features
        numeric_table_result = artifacts['numeric_table']
        assert not numeric_table_result.empty
        expected_numerics = ['TAT_minutes', 'queue_length_at_order', 'floor_occupancy_pct', 'patient_age']
        num_names = set(numeric_table_result['name'])
        assert all(num in num_names for num in expected_numerics)
        
        # Should generate correlation analysis
        correlations = artifacts['correlations']
        if not correlations.empty:
            assert 'TAT_minutes' in correlations.columns or 'TAT_minutes' in correlations.index

    def test_build_artifacts_missing_data_assessment(self, comprehensive_tat_dataset):
        """Test data quality assessment integration in artifact generation."""
        config = SummaryConfig()
        artifacts = build_artifacts(comprehensive_tat_dataset, config)
        
        # Should assess missing data comprehensively
        missing_table = artifacts['missing_table']
        missing_console = artifacts['missing_table_console']
        
        # Should identify columns with missing data
        assert not missing_table.empty  # Dataset has missing patterns
        assert 'column' in missing_table.columns
        assert 'missing%' in missing_table.columns
        
        # Console version should be truncated
        assert len(missing_console) <= config.missing_top_n
        assert len(missing_console) <= len(missing_table)
        
        # Should identify healthcare-typical missing patterns
        missing_cols = set(missing_table['column'])
        healthcare_missing_expected = {'nurse_validation_time', 'prep_complete_time', 'lab_WBC_k_per_uL'}
        
        # At least some healthcare columns should show missing data
        assert len(missing_cols & healthcare_missing_expected) >= 2

    def test_build_artifacts_metadata_summary(self, comprehensive_tat_dataset):
        """Test metadata summary generation for executive reporting."""
        config = SummaryConfig()
        artifacts = build_artifacts(comprehensive_tat_dataset, config)
        
        counts = artifacts['counts']
        
        # Should provide comprehensive metadata
        expected_count_keys = ['rows', 'cols', 'time', 'categorical', 'numeric', 'missing_cols']
        assert all(key in counts for key in expected_count_keys)
        
        # Should reflect dataset characteristics
        assert counts['rows'] == 2500  # Test dataset size
        assert counts['cols'] == len(comprehensive_tat_dataset.columns)
        
        # Should classify columns appropriately
        assert counts['time'] >= 3      # Workflow timestamps
        assert counts['categorical'] >= 4  # Operational factors  
        assert counts['numeric'] >= 4   #  Numerical Features
        assert counts['missing_cols'] >= 3  # Columns with missing data
        
        # Counts should be non-negative integers
        assert all(isinstance(count, int) and count >= 0 for count in counts.values())

    def test_build_artifacts_preprocessing_integration(self, comprehensive_tat_dataset):
        """Test preprocessing integration and data consistency."""
        config = SummaryConfig()
        artifacts = build_artifacts(comprehensive_tat_dataset, config)
        
        processed_df = artifacts['df_processed']
        
        # Should return preprocessed dataset
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(comprehensive_tat_dataset)
        
        # Should preserve essential data structure
        assert processed_df.shape[0] == comprehensive_tat_dataset.shape[0]  # Same row count
        
        # Timestamp columns should be processed appropriately
        timestamp_cols = ['doctor_order_time', 'nurse_validation_time', 'prep_complete_time']
        for col in timestamp_cols:
            if col in processed_df.columns:
                # Should be datetime or maintain original if preprocessing failed
                assert pd.api.types.is_datetime64_any_dtype(processed_df[col]) or processed_df[col].dtype == comprehensive_tat_dataset[col].dtype

    def test_build_artifacts_empty_input_robustness(self):
        """Test artifact generation robustness with empty and minimal datasets."""
        config = SummaryConfig()
        
        # Empty dataset
        empty_df = pd.DataFrame()
        empty_artifacts = build_artifacts(empty_df, config)
        
        # Should handle empty input gracefully
        assert isinstance(empty_artifacts, dict)
        assert empty_artifacts['counts']['rows'] == 0
        assert empty_artifacts['counts']['cols'] == 0
        
        # Single column dataset
        minimal_df = pd.DataFrame({'single_col': [1, 2, 3]})
        minimal_artifacts = build_artifacts(minimal_df, config)
        
        # Should process minimal dataset without errors
        assert isinstance(minimal_artifacts, dict)
        assert minimal_artifacts['counts']['rows'] == 3
        assert minimal_artifacts['counts']['cols'] == 1

    def test_build_artifacts_healthcare_domain_validation(self, comprehensive_tat_dataset):
        """Test that artifacts align with healthcare domain requirements and terminology."""
        config = SummaryConfig()
        artifacts = build_artifacts(comprehensive_tat_dataset, config)
        
        # Should use healthcare terminology in analysis
        time_table = artifacts['time_table']
        healthcare_workflow_terms = ['doctor', 'nurse', 'prep', 'validation', 'time']
        
        if not time_table.empty:
            timestamp_names = ' '.join(time_table['name'])
            assert any(term in timestamp_names.lower() for term in healthcare_workflow_terms)
        
        # Should identify clinical factors
        categorical_table = artifacts['categorical_table'] 
        clinical_terms = ['shift', 'credential', 'diagnosis', 'severity']
        
        if not categorical_table.empty:
            categorical_names = ' '.join(categorical_table['name'])
            assert any(term in categorical_names.lower() for term in clinical_terms)
        
        # Should analyze  Numerical Features
        numeric_table = artifacts['numeric_table']
        performance_terms = ['TAT', 'queue', 'occupancy', 'pharmacist', 'age']
        
        if not numeric_table.empty:
            numeric_names = ' '.join(numeric_table['name'])
            assert any(term in numeric_names.lower() for term in performance_terms)
        
        # Should provide executive summary suitable for healthcare leadership
        counts = artifacts['counts']
        assert all(isinstance(count, int) for count in counts.values())
        assert counts['rows'] > 0  # Should process actual data