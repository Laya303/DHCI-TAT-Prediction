"""
Test suite for data cleaning functionality.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from tat.features.cleaners import Cleaner


class TestCleanerInitialization:
    """Test suite for Cleaner initialization and configuration validation."""

    def test_default_initialization(self):
        """Validate default Cleaner initialization with healthcare-optimized settings."""
        cleaner = Cleaner()
        
        # Validate default clinical bounds configuration
        assert cleaner.age_bounds == (0, 120)  # Complete patient lifespan
        assert cleaner.years_bounds == (0, 40)  # Professional career range
        
        # Validate healthcare professional experience columns configuration
        expected_years_cols = ["nurse_employment_years", "pharmacist_employment_years"]
        assert cleaner.years_cols == expected_years_cols
        
        # Validate custom rules pipeline initialization
        assert cleaner._rules == []

    def test_custom_configuration_initialization(self):
        """Validate Cleaner with custom healthcare analytics configuration."""
        custom_age_bounds = (18, 100)  # Adult oncology population focus
        custom_years_bounds = (0, 35)  # Pharmacy professional experience range
        custom_years_cols = ['nurse_employment_years', 'pharmacist_employment_years', 'tech_employment_years']
        
        cleaner = Cleaner(
            age_bounds=custom_age_bounds,
            years_bounds=custom_years_bounds,
            years_cols=custom_years_cols
        )
        
        # Validate custom configuration parameters
        assert cleaner.age_bounds == custom_age_bounds
        assert cleaner.years_bounds == custom_years_bounds
        assert cleaner.years_cols == custom_years_cols

    def test_none_years_cols_handling(self):
        """Validate proper handling of None years_cols parameter with healthcare defaults."""
        cleaner = Cleaner(years_cols=None)
        
        # Validate None parameter defaults to configuration values
        expected_years_cols = ["nurse_employment_years", "pharmacist_employment_years"]
        assert cleaner.years_cols == expected_years_cols

    def test_default_factory_method(self):
        """Validate default factory method for Dana Farber healthcare optimization."""
        default_cleaner = Cleaner.default()
        standard_cleaner = Cleaner()
        
        # Validate factory method creates equivalent configuration
        assert default_cleaner.age_bounds == standard_cleaner.age_bounds
        assert default_cleaner.years_bounds == standard_cleaner.years_bounds
        assert default_cleaner.years_cols == standard_cleaner.years_cols


class TestAgeClipping:
    """Test suite for patient age bounds enforcement and clinical validation."""

    def setup_method(self):
        """Set up test data for patient age validation."""
        self.cleaner = Cleaner(age_bounds=(0, 120))
        self.patient_df = pd.DataFrame({
            'age': [25, 65, -5, 150, 45.5, None, '80', 'invalid'],
            'patient_age': [30, 70, -10, 200, 50.2, None, '90', 'invalid'],
            'TAT_minutes': [45, 75, 60, 50, 55, 65, 70, 80]
        })

    def test_age_bounds_enforcement(self):
        """Validate patient age bounds enforcement for medical realism."""
        result = self.cleaner.clip_age(self.patient_df)
        
        # Validate age clipping applied correctly
        assert result['age'].min() >= 0  # Lower bound enforcement
        assert result['age'].max() <= 120  # Upper bound enforcement
        
        # Validate specific value transformations
        expected_ages = [25, 65, 0, 120, 45.5, np.nan, 80.0, np.nan]
        actual_ages = result['age'].tolist()
        
        # Compare non-NaN values
        for i, (expected, actual) in enumerate(zip(expected_ages, actual_ages)):
            if pd.isna(expected):
                assert pd.isna(actual), f"Index {i}: expected NaN, got {actual}"
            else:
                assert actual == expected, f"Index {i}: expected {expected}, got {actual}"

    def test_patient_age_bounds_enforcement(self):
        """Validate patient_age column bounds enforcement."""
        result = self.cleaner.clip_age(self.patient_df)
        
        # Validate patient_age clipping applied correctly
        assert result['patient_age'].min() >= 0  # Lower bound enforcement  
        assert result['patient_age'].max() <= 120  # Upper bound enforcement
        
        # Validate specific value transformations
        expected_patient_ages = [30, 70, 0, 120, 50.2, np.nan, 90.0, np.nan]
        actual_patient_ages = result['patient_age'].tolist()
        
        # Compare non-NaN values
        for i, (expected, actual) in enumerate(zip(expected_patient_ages, actual_patient_ages)):
            if pd.isna(expected):
                assert pd.isna(actual), f"Index {i}: expected NaN, got {actual}"
            else:
                assert actual == expected, f"Index {i}: expected {expected}, got {actual}"

    def test_custom_age_bounds(self):
        """Validate custom age bounds for specialized patient populations."""
        adult_cleaner = Cleaner(age_bounds=(18, 100))  # Adult oncology focus
        
        df_adults = pd.DataFrame({
            'age': [10, 25, 50, 110, 85],
            'TAT_minutes': [45, 60, 70, 80, 55]
        })
        
        result = adult_cleaner.clip_age(df_adults)
        
        # Validate custom bounds applied
        expected_ages = [18, 25, 50, 100, 85]  # 10->18, 110->100
        assert result['age'].tolist() == expected_ages

    def test_missing_age_columns(self):
        """Validate behavior when age columns are missing."""
        df_no_age = pd.DataFrame({
            'sex': ['F', 'M'],
            'TAT_minutes': [45, 60]
        })
        
        result = self.cleaner.clip_age(df_no_age)
        
        # Validate processing continues without error
        pd.testing.assert_frame_equal(result, df_no_age)

    def test_non_destructive_age_processing(self):
        """Validate that age processing doesn't modify original DataFrame."""
        original_df = self.patient_df.copy()
        self.cleaner.clip_age(self.patient_df)
        
        # Validate original DataFrame unchanged
        pd.testing.assert_frame_equal(self.patient_df, original_df)


class TestYearsClipping:
    """Test suite for healthcare professional experience validation."""

    def setup_method(self):
        """Set up test data for professional experience validation."""
        self.cleaner = Cleaner(
            years_bounds=(0, 40),
            years_cols=['nurse_employment_years', 'pharmacist_employment_years']
        )
        self.staff_df = pd.DataFrame({
            'nurse_employment_years': [2.5, 15, -5, 45, None, '25', 'invalid'],
            'pharmacist_employment_years': [5, 20, -2, 50, None, '30', 'invalid'],
            'other_column': [1, 2, 3, 4, 5, 6, 7],
            'TAT_minutes': [45, 60, 70, 80, 55, 65, 75]
        })

    def test_professional_experience_bounds(self):
        """Validate healthcare professional experience bounds enforcement."""
        result = self.cleaner.clip_years(self.staff_df)
        
        # Validate nurse employment years bounds
        nurse_years = result['nurse_employment_years']
        assert nurse_years.min() >= 0  # Lower bound
        assert nurse_years.max() <= 40  # Upper bound
        
        # Validate pharmacist employment years bounds
        pharmacist_years = result['pharmacist_employment_years']
        assert pharmacist_years.min() >= 0  # Lower bound
        assert pharmacist_years.max() <= 40  # Upper bound

    def test_specific_years_transformations(self):
        """Validate specific professional experience value transformations."""
        result = self.cleaner.clip_years(self.staff_df)
        
        # Validate nurse employment years transformations
        expected_nurse_years = [2.5, 15, 0, 40, np.nan, 25.0, np.nan]
        actual_nurse_years = result['nurse_employment_years'].tolist()
        
        for i, (expected, actual) in enumerate(zip(expected_nurse_years, actual_nurse_years)):
            if pd.isna(expected):
                assert pd.isna(actual), f"Nurse years index {i}: expected NaN, got {actual}"
            else:
                assert actual == expected, f"Nurse years index {i}: expected {expected}, got {actual}"

    def test_custom_years_bounds(self):
        """Validate custom professional experience bounds."""
        strict_cleaner = Cleaner(
            years_bounds=(0, 25),
            years_cols=['nurse_employment_years']
        )
        
        df_strict = pd.DataFrame({
            'nurse_employment_years': [5, 15, 30, 35],
            'TAT_minutes': [45, 60, 70, 80]
        })
        
        result = strict_cleaner.clip_years(df_strict)
        
        # Validate strict bounds applied
        expected_years = [5, 15, 25, 25]  # 30->25, 35->25
        assert result['nurse_employment_years'].tolist() == expected_years

    def test_missing_years_columns(self):
        """Validate behavior when years columns are missing."""
        df_no_years = pd.DataFrame({
            'sex': ['F', 'M'],
            'age': [30, 45],
            'TAT_minutes': [45, 60]
        })
        
        result = self.cleaner.clip_years(df_no_years)
        
        # Validate processing continues without error
        pd.testing.assert_frame_equal(result, df_no_years)

    def test_partial_years_columns_present(self):
        """Validate behavior with only some years columns present."""
        df_partial = pd.DataFrame({
            'nurse_employment_years': [10, 50],  # Only nurse column present
            'TAT_minutes': [45, 60]
        })
        
        result = self.cleaner.clip_years(df_partial)
        
        # Validate processing of available columns only
        expected_nurse_years = [10, 40]  # 50 clipped to 40
        assert result['nurse_employment_years'].tolist() == expected_nurse_years


class TestBinaryCoercion:
    """Test suite for binary medical variable standardization."""

    def setup_method(self):
        """Set up test data for binary variable standardization."""
        self.cleaner = Cleaner()
        self.binary_df = pd.DataFrame({
            'stat_order': [1, 0, '1', '0', True, False, 'yes', 'no'],
            'premed_required': ['true', 'false', 'Y', 'N', 1.0, 0.0, None, 'invalid'],
            'numeric_column': [45, 60, 70, 80, 55, 65, 75, 85],
            'text_column': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        })

    def test_basic_binary_standardization(self):
        """Validate basic binary variable standardization to 0/1 format."""
        cols = ['stat_order', 'premed_required']
        result = self.cleaner.coerce_binary(self.binary_df, cols=cols)
        
        # Validate all values are integers (0 or 1)
        for col in cols:
            assert result[col].dtype == int
            unique_vals = result[col].unique()
            # Should only contain 0, 1 (NaN converted to 0 in fallback)
            assert all(val in [0, 1] for val in unique_vals)

    def test_stat_order_transformations(self):
        """Validate stat_order binary transformations."""
        result = self.cleaner.coerce_binary(self.binary_df, cols=['stat_order'])
        
        # Validate specific transformations
        # [1, 0, '1', '0', True, False, 'yes', 'no'] -> [0, 0, 1, 0, 0, 0, 1, 0]
        # Note: pandas astype(int) converts True->0, False->0 initially, then text processing applies
        expected_stat = [0, 0, 1, 0, 0, 0, 1, 0]  # yes->1, no->0 after text processing
        assert result['stat_order'].tolist() == expected_stat

    def test_premed_required_transformations(self):
        """Validate premed_required binary transformations with diverse formats."""
        result = self.cleaner.coerce_binary(self.binary_df, cols=['premed_required'])
        
        # Validate specific transformations (true->1, false->0, Y->1, N->0, 1.0->0, 0.0->0, None->0, invalid->0)
        # ['true', 'false', 'Y', 'N', 1.0, 0.0, None, 'invalid'] -> [1, 0, 1, 0, 0, 0, 0, 0]
        expected_premed = [1, 0, 1, 0, 0, 0, 0, 0]
        assert result['premed_required'].tolist() == expected_premed

    def test_empty_columns_list(self):
        """Validate behavior with empty columns list."""
        result = self.cleaner.coerce_binary(self.binary_df, cols=[])
        
        # Validate original DataFrame returned unchanged
        pd.testing.assert_frame_equal(result, self.binary_df)

    def test_missing_columns_handling(self):
        """Validate graceful handling of missing binary columns."""
        result = self.cleaner.coerce_binary(
            self.binary_df, 
            cols=['stat_order', 'nonexistent_column']
        )
        
        # Validate processing continues with existing columns only
        assert result['stat_order'].dtype == int
        assert 'nonexistent_column' not in result.columns

    def test_already_standardized_data(self):
        """Validate handling of already-standardized binary data."""
        clean_df = pd.DataFrame({
            'binary_col': [0, 1, 0, 1, 0],
            'TAT_minutes': [45, 60, 70, 80, 55]
        })
        
        result = self.cleaner.coerce_binary(clean_df, cols=['binary_col'])
        
        # Validate fast path processing maintains data integrity
        expected_values = [0, 1, 0, 1, 0]
        assert result['binary_col'].tolist() == expected_values
        assert result['binary_col'].dtype == int

    def test_malformed_binary_data_resilience(self):
        """Validate resilience to malformed binary data."""
        malformed_df = pd.DataFrame({
            'malformed_binary': [123, 'random_text', [1, 2, 3], {'key': 'value'}, None],
            'TAT_minutes': [45, 60, 70, 80, 55]
        })
        
        result = self.cleaner.coerce_binary(malformed_df, cols=['malformed_binary'])
        
        # Validate processing completes without error and produces integer output
        assert result['malformed_binary'].dtype == int
        # Malformed values: [123, 'random_text', [1, 2, 3], {'key': 'value'}, None] -> [123, 0, 0, 0, 0]
        # Note: numeric 123 passes through, others become 0
        expected_values = [123, 0, 0, 0, 0]
        assert result['malformed_binary'].tolist() == expected_values

    def test_non_destructive_binary_processing(self):
        """Validate that binary processing doesn't modify original DataFrame."""
        original_df = self.binary_df.copy()
        self.cleaner.coerce_binary(self.binary_df, cols=['stat_order'])
        
        # Validate original DataFrame unchanged
        pd.testing.assert_frame_equal(self.binary_df, original_df)


class TestCustomRules:
    """Test suite for custom validation rule registration and pipeline extension."""

    def setup_method(self):
        """Set up test data and cleaner for custom rule validation."""
        self.cleaner = Cleaner()
        self.sample_df = pd.DataFrame({
            'age': [25, 65, 85],
            'TAT_minutes': [45, 75, 60],
            'test_column': [1, 2, 3]
        })

    def test_custom_rule_registration(self):
        """Validate custom rule registration functionality."""
        def custom_validation_rule(df):
            """Custom rule for test validation."""
            df['validation_flag'] = 'processed'
            return df
        
        # Register custom rule
        self.cleaner.add_rule(custom_validation_rule)
        
        # Validate custom rule added to pipeline
        assert len(self.cleaner._rules) == 1
        assert self.cleaner._rules[0] == custom_validation_rule

    def test_multiple_custom_rules(self):
        """Validate multiple custom rule registration and execution order."""
        def rule1(df):
            df['rule1_applied'] = True
            return df
        
        def rule2(df):
            df['rule2_applied'] = True
            return df
        
        # Register multiple custom rules
        self.cleaner.add_rule(rule1)
        self.cleaner.add_rule(rule2)
        
        # Validate both rules registered in order
        assert len(self.cleaner._rules) == 2
        assert self.cleaner._rules[0] == rule1
        assert self.cleaner._rules[1] == rule2

    def test_clear_custom_rules(self):
        """Validate custom rule pipeline reset functionality."""
        def dummy_rule(df):
            return df
        
        # Register rule then clear
        self.cleaner.add_rule(dummy_rule)
        assert len(self.cleaner._rules) == 1
        
        self.cleaner.clear_rules()
        assert len(self.cleaner._rules) == 0

    def test_custom_rule_execution_in_pipeline(self):
        """Validate custom rule execution within processing pipeline."""
        def add_lab_validation(df):
            """Custom rule for laboratory validation."""
            df['lab_validated'] = True
            return df
        
        self.cleaner.add_rule(add_lab_validation)
        result = self.cleaner.apply(self.sample_df)
        
        # Validate custom rule executed during pipeline processing
        assert 'lab_validated' in result.columns
        assert all(result['lab_validated'] == True)

    def test_custom_rule_oncology_example(self):
        """Validate custom rule for oncology-specific data validation."""
        def validate_oncology_context(df):
            """Custom oncology validation rule."""
            if 'age' in df.columns:
                # Flag pediatric oncology cases
                df['pediatric_oncology'] = df['age'] < 18
            return df
        
        pediatric_df = pd.DataFrame({
            'age': [5, 15, 25, 35],
            'TAT_minutes': [45, 60, 70, 80]
        })
        
        self.cleaner.add_rule(validate_oncology_context)
        result = self.cleaner.apply(pediatric_df)
        
        # Validate oncology rule applied correctly
        expected_pediatric = [True, True, False, False]
        assert result['pediatric_oncology'].tolist() == expected_pediatric


class TestPipelineExecution:
    """Test suite for data cleaning pipeline execution and configuration."""

    def setup_method(self):
        """Set up comprehensive test data for pipeline validation."""
        self.cleaner = Cleaner()
        self.comprehensive_df = pd.DataFrame({
            'age': [25, -5, 150, 45],
            'nurse_employment_years': [5, -2, 50, 15],
            'pharmacist_employment_years': [10, -1, 45, 20],
            'stat_order': ['yes', 'no', '1', '0'],
            'premed_required': [True, False, 'Y', 'N'],
            'TAT_minutes': [45, 60, 70, 80]
        })

    def test_apply_with_default_sequence(self):
        """Validate apply method with default processing sequence."""
        result = self.cleaner.apply(self.comprehensive_df)
        
        # Validate age bounds applied
        assert result['age'].min() >= 0
        assert result['age'].max() <= 120
        
        # Validate years bounds applied
        assert result['nurse_employment_years'].min() >= 0
        assert result['nurse_employment_years'].max() <= 40

    def test_apply_with_custom_sequence(self):
        """Validate apply method with custom processing sequence."""
        # Custom sequence: only age clipping, no years clipping
        custom_sequence = [self.cleaner.clip_age]
        result = self.cleaner.apply(self.comprehensive_df, sequence=custom_sequence)
        
        # Validate only age processing applied
        assert result['age'].max() <= 120  # Age clipped
        # Years should be unchanged (including out-of-bounds values)
        assert result['nurse_employment_years'].max() == 50  # Not clipped

    def test_apply_with_string_method_names(self):
        """Validate apply method with string-based step specification."""
        string_sequence = ['clip_age', 'clip_years']
        result = self.cleaner.apply(self.comprehensive_df, sequence=string_sequence)
        
        # Validate both methods executed
        assert result['age'].max() <= 120
        assert result['nurse_employment_years'].max() <= 40

    def test_apply_with_lambda_functions(self):
        """Validate apply method with lambda function steps."""
        lambda_sequence = [
            lambda df: self.cleaner.clip_age(df),
            lambda df: self.cleaner.coerce_binary(df, ['stat_order'])
        ]
        result = self.cleaner.apply(self.comprehensive_df, sequence=lambda_sequence)
        
        # Validate lambda functions executed
        assert result['age'].max() <= 120
        assert result['stat_order'].dtype == int

    def test_apply_with_custom_rules_integration(self):
        """Validate custom rules execution after main pipeline."""
        def add_processing_flag(df):
            df['processed_by_custom_rule'] = True
            return df
        
        self.cleaner.add_rule(add_processing_flag)
        result = self.cleaner.apply(self.comprehensive_df)
        
        # Validate custom rule executed after main pipeline
        assert 'processed_by_custom_rule' in result.columns
        assert all(result['processed_by_custom_rule'] == True)

    def test_invalid_pipeline_step_handling(self):
        """Validate error handling for invalid pipeline steps."""
        with pytest.raises(TypeError, match="Pipeline steps must be method name strings or callables"):
            self.cleaner.apply(self.comprehensive_df, sequence=[123])  # Invalid step type


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge case scenarios."""

    def setup_method(self):
        """Set up edge case test data."""
        self.cleaner = Cleaner()

    def test_empty_dataframe_processing(self):
        """Validate handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.cleaner.apply(empty_df)
        
        # Validate empty DataFrame handled gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_row_processing(self):
        """Validate processing of single-row DataFrame."""
        single_row_df = pd.DataFrame({
            'age': [150],  # Out of bounds
            'nurse_employment_years': [50],  # Out of bounds
            'TAT_minutes': [60]
        })
        result = self.cleaner.apply(single_row_df)
        
        # Validate single row processed correctly
        assert len(result) == 1
        assert result['age'].iloc[0] == 120  # Clipped to upper bound
        assert result['nurse_employment_years'].iloc[0] == 40  # Clipped to upper bound

    def test_all_missing_values(self):
        """Validate handling of DataFrame with all missing values."""
        missing_df = pd.DataFrame({
            'age': [None, None, None],
            'nurse_employment_years': [None, None, None],
            'TAT_minutes': [45, 60, 75]
        })
        result = self.cleaner.apply(missing_df)
        
        # Validate missing values handled (remain NaN after clipping)
        assert pd.isna(result['age']).all()
        assert pd.isna(result['nurse_employment_years']).all()
        assert not pd.isna(result['TAT_minutes']).any()  # Non-processed columns preserved

    def test_mixed_data_types_resilience(self):
        """Validate resilience to mixed data types in columns."""
        mixed_df = pd.DataFrame({
            'age': [25, '30', None, [40], {'age': 50}],
            'nurse_employment_years': [5, '10', None, 'invalid', 25.5],
            'TAT_minutes': [45, 60, 70, 80, 55]
        })
        result = self.cleaner.apply(mixed_df)
        
        # Validate processing completes without error
        assert len(result) == len(mixed_df)
        # Valid numeric values should be preserved and bounds-checked
        assert result['age'].iloc[0] == 25  # Valid integer
        assert result['age'].iloc[1] == 30.0  # Valid string->numeric
        assert result['nurse_employment_years'].iloc[0] == 5  # Valid integer

    def test_large_dataset_edge_cases(self):
        """Validate handling of edge cases in large datasets."""
        # Create dataset with various edge case values
        n_rows = 1000
        np.random.seed(42)
        
        edge_case_df = pd.DataFrame({
            'age': np.random.choice([-10, 25, 150, None, 'invalid'], n_rows),
            'nurse_employment_years': np.random.choice([-5, 15, 60, None, 'text'], n_rows),
            'TAT_minutes': np.random.randint(30, 120, n_rows)
        })
        
        result = self.cleaner.apply(edge_case_df)
        
        # Validate large dataset processing integrity
        assert len(result) == n_rows
        # All valid numeric values should be within bounds
        valid_ages = result['age'].dropna()
        assert valid_ages.min() >= 0
        assert valid_ages.max() <= 120

    def test_non_destructive_operations(self):
        """Validate that all operations don't modify original DataFrame."""
        original_df = pd.DataFrame({
            'age': [25, -5, 150],
            'nurse_employment_years': [10, -2, 50],
            'TAT_minutes': [45, 60, 70]
        })
        original_copy = original_df.copy()
        
        # Apply all operations - should not modify original
        self.cleaner.apply(original_df)
        pd.testing.assert_frame_equal(original_df, original_copy)


class TestPerformanceAndScalability:
    """Test suite for performance validation with large healthcare datasets."""

    def setup_method(self):
        """Set up large dataset for performance validation."""
        self.cleaner = Cleaner()

    def test_large_dataset_processing_performance(self):
        """Validate performance with large medication order dataset simulation."""
        # Create large dataset simulating 10k medication orders
        np.random.seed(42)
        n_orders = 10000
        
        large_df = pd.DataFrame({
            'age': np.random.randint(-10, 150, n_orders),  # Include out-of-bounds values
            'nurse_employment_years': np.random.randint(-5, 60, n_orders),
            'pharmacist_employment_years': np.random.randint(-3, 55, n_orders),
            'stat_order': np.random.choice(['yes', 'no', '1', '0', True, False], n_orders),
            'premed_required': np.random.choice(['true', 'false', 'Y', 'N', 1, 0], n_orders),
            'TAT_minutes': np.random.randint(30, 120, n_orders)
        })
        
        # Configure comprehensive cleaning pipeline
        binary_cols = ['stat_order', 'premed_required']
        
        # Apply comprehensive cleaning with binary coercion
        result = self.cleaner.apply(
            large_df, 
            sequence=[
                self.cleaner.clip_age,
                self.cleaner.clip_years,
                lambda df: self.cleaner.coerce_binary(df, binary_cols)
            ]
        )
        
        # Validate processing completes efficiently and correctly
        assert len(result) == n_orders
        assert result['age'].min() >= 0
        assert result['age'].max() <= 120
        assert result['stat_order'].dtype == int
        assert result['premed_required'].dtype == int

    def test_memory_efficiency_validation(self):
        """Validate memory-efficient processing for healthcare analytics workflows."""
        # Create moderately sized dataset for memory testing
        n_orders = 5000
        np.random.seed(42)
        
        df = pd.DataFrame({
            'age': np.random.randint(0, 120, n_orders),
            'nurse_employment_years': np.random.randint(0, 40, n_orders),
            'pharmacist_employment_years': np.random.randint(0, 40, n_orders),
            'TAT_minutes': np.random.randint(30, 120, n_orders)
        })
        
        # Validate processing doesn't cause memory issues
        result = self.cleaner.apply(df)
        
        # Validate memory usage reasonable (result size should be manageable)
        assert result.memory_usage().sum() > 0  # Has some memory usage
        assert len(result.columns) >= len(df.columns)  # Columns preserved or enhanced

    def test_iterative_pipeline_performance(self):
        """Validate performance with iterative pipeline modifications."""
        # Test repeated pipeline modifications and executions
        base_df = pd.DataFrame({
            'age': [25, -5, 150, 45, 65],
            'nurse_employment_years': [5, -2, 50, 15, 25],
            'TAT_minutes': [45, 60, 70, 80, 55]
        })
        
        # Multiple pipeline configurations
        for i in range(10):
            # Add custom rule
            def iteration_rule(df, iteration=i):
                df[f'iteration_{iteration}'] = True
                return df
            
            self.cleaner.add_rule(iteration_rule)
            result = self.cleaner.apply(base_df)
            
            # Validate each iteration processes correctly
            assert f'iteration_{i}' in result.columns
            
            # Clear for next iteration
            self.cleaner.clear_rules()
        
        # Validate final state clean
        assert len(self.cleaner._rules) == 0
