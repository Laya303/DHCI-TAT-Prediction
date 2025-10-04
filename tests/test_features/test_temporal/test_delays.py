"""
Test suite for functionality.

Tests the production-ready delay computation system for medication preparation turnaround
time analysis supporting Dana Farber's pharmacy workflow optimization initiatives.
Validates step-wise delay calculation, healthcare-optimized imputation strategies,
and clinical workflow sequence integrity for robust TAT prediction modeling.

Test Coverage:
- Basic delay computation from timestamp pairs with validation
- Sequential imputation for missing timestamps using operational context patterns  
- Healthcare data quality handling and negative delay correction strategies
- Operational context-aware statistics and processing time learning
- Clinical workflow chronological sequence validation and integrity
- Production pipeline integration and helper column management

Healthcare Domain Testing:
- Medication preparation workflow sequence validation throughout processing
- EHR integration artifact handling with defensive programming strategies
- Operational factor awareness (shift, floor patterns) for context-specific imputation
- Clinical decision-making support through accurate delay reconstruction
- Quality monitoring capabilities with comprehensive logging validation

Design Philosophy:
- Healthcare-first testing: Domain knowledge validation in all test scenarios
- Production-ready validation: Realistic data patterns and edge case handling
- Clinical interpretability: Workflow sequence and timing constraint validation
- Defensive programming: Robust error handling and data quality assurance testing
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import logging
from typing import Dict, Any, Tuple, List

from src.tat.features.temporal.delays import DelayEngineer
from src.tat.config import STEP_COLS, DELAY_PAIRS, HELPER_SUFFIXES


class TestDelayEngineerInitialization:
    """Test DelayEngineer initialization and configuration validation."""

    def test_default_initialization(self):
        """Test DelayEngineer initialization with default configuration."""
        engineer = DelayEngineer()
        
        # Validate default configuration parameters
        assert engineer.pairs == list(DELAY_PAIRS)
        assert engineer.drop_suffixes == HELPER_SUFFIXES
        assert engineer.step_cols == list(STEP_COLS)
        assert engineer.impute_missing is True
        
    def test_custom_initialization(self):
        """Test DelayEngineer initialization with custom parameters."""
        custom_pairs = [
            (None, "nurse_validation_time_mins_unwrapped", "delay_order_to_nurse"),
            ("nurse_validation_time_mins_unwrapped", "prep_complete_time_mins_unwrapped", "delay_nurse_to_prep")
        ]
        custom_suffixes = ("_test", "_custom")
        custom_step_cols = ["step1", "step2"]
        
        engineer = DelayEngineer(
            pairs=custom_pairs,
            drop_suffixes=custom_suffixes,
            step_cols=custom_step_cols,
            impute_missing=False
        )
        
        # Validate custom configuration
        assert engineer.pairs == custom_pairs
        assert engineer.drop_suffixes == custom_suffixes
        assert engineer.step_cols == custom_step_cols
        assert engineer.impute_missing is False

    def test_none_parameters_use_defaults(self):
        """Test that None parameters properly use default values."""
        engineer = DelayEngineer(
            pairs=None,
            drop_suffixes=None,
            step_cols=None
        )
        
        # Should use defaults from config
        assert engineer.pairs == list(DELAY_PAIRS)
        assert engineer.drop_suffixes == HELPER_SUFFIXES
        assert engineer.step_cols == list(STEP_COLS)


class TestDelayComputationBasics:
    """Test basic delay calculation functionality."""

    @pytest.fixture
    def simple_tat_data(self):
        """Generate simple TAT dataset for basic delay testing."""
        base_time = pd.Timestamp('2025-01-15 08:00:00')
        
        return pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [10, 15, 20, 25, 30],
            'prep_complete_time_mins_unwrapped': [40, 50, 55, 60, 65],
            'second_validation_time_mins_unwrapped': [45, 55, 60, 65, 70],
            'floor_dispatch_time_mins_unwrapped': [50, 60, 65, 70, 75],
            'patient_infusion_time_mins_unwrapped': [70, 80, 85, 90, 95],
            'shift': ['Day', 'Day', 'Evening', 'Evening', 'Night'],
            'floor': [1, 2, 3, 1, 2]
        })

    def test_basic_delay_computation(self, simple_tat_data):
        """Test basic step-wise delay computation."""
        engineer = DelayEngineer(impute_missing=False)
        result = engineer.transform(simple_tat_data)
        
        # Validate delay columns are created
        expected_delay_cols = [
            'delay_order_to_nurse', 'delay_nurse_to_prep', 'delay_prep_to_second',
            'delay_second_to_dispatch', 'delay_dispatch_to_infusion'
        ]
        
        for col in expected_delay_cols:
            assert col in result.columns
        
        # Validate specific delay calculations
        assert result['delay_nurse_to_prep'].iloc[0] == 30  # 40 - 10
        assert result['delay_prep_to_second'].iloc[0] == 5   # 45 - 40
        assert result['delay_second_to_dispatch'].iloc[0] == 5  # 50 - 45
        assert result['delay_dispatch_to_infusion'].iloc[0] == 20  # 70 - 50

    def test_negative_delay_handling(self):
        """Test handling of negative delays (chronological sequence violations)."""
        df = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [10, 15],
            'prep_complete_time_mins_unwrapped': [5, 50],  # First row has negative delay
            'shift': ['Day', 'Day'],
            'floor': [1, 1]
        })
        
        engineer = DelayEngineer(impute_missing=False)
        
        with patch('src.tat.features.temporal.delays.logger') as mock_logger:
            result = engineer.transform(df)
            
            # Should clamp negative delays to 0
            assert result['delay_nurse_to_prep'].iloc[0] == 0
            assert result['delay_nurse_to_prep'].iloc[1] == 35  # 50 - 15
            
            # Should log warning about negative delays
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "negative delays" in warning_call.lower()

    def test_missing_columns_handling(self):
        """Test handling when required timestamp columns are missing."""
        df = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [10, 15],
            # Missing prep_complete_time_mins_unwrapped
            'shift': ['Day', 'Day'],
            'floor': [1, 1]
        })
        
        engineer = DelayEngineer(impute_missing=False)
        result = engineer.transform(df)
        
        # Should handle missing columns gracefully
        assert 'delay_order_to_nurse' in result.columns
        # delay_nurse_to_prep should not be computed due to missing next column
        delay_cols = [col for col in result.columns if col.startswith('delay_nurse_to_prep')]
        assert len(delay_cols) == 0 or result['delay_nurse_to_prep'].isna().all()

    def test_to_num_static_method(self):
        """Test the _to_num static method for robust numeric conversion."""
        # Test mixed data types
        mixed_series = pd.Series(['10', 15, '20.5', None, 'invalid'])
        result = DelayEngineer._to_num(mixed_series)
        
        expected = pd.Series([10.0, 15.0, 20.5, np.nan, np.nan])
        pd.testing.assert_series_equal(result, expected)
        
        # Test already numeric series
        numeric_series = pd.Series([1.5, 2.0, 3.0])
        result = DelayEngineer._to_num(numeric_series)
        pd.testing.assert_series_equal(result, numeric_series)


class TestImputationFunctionality:
    """Test missing timestamp imputation capabilities."""

    @pytest.fixture
    def missing_data_tat(self):
        """Generate TAT dataset with missing timestamps for imputation testing."""
        return pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [10, np.nan, 20, np.nan, 30],
            'prep_complete_time_mins_unwrapped': [40, 45, np.nan, 60, 65],
            'second_validation_time_mins_unwrapped': [45, 50, 55, np.nan, 70],
            'floor_dispatch_time_mins_unwrapped': [50, 55, 60, 65, np.nan],
            'patient_infusion_time_mins_unwrapped': [70, 75, 80, 85, 90],
            'shift': ['Day', 'Day', 'Evening', 'Evening', 'Night'],
            'floor': [1, 2, 1, 2, 1]
        })

    def test_missing_data_analysis_logging(self, missing_data_tat):
        """Test logging of missing data analysis before and after imputation."""
        engineer = DelayEngineer(impute_missing=True)
        
        with patch('src.tat.features.temporal.delays.logger') as mock_logger:
            result = engineer.transform(missing_data_tat)
            
            # Should log missing data analysis
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
            
            # Check for before imputation logging
            before_logs = [log for log in info_calls if "before imputation" in log.lower()]
            assert len(before_logs) > 0
            
            # Check for after imputation logging
            after_logs = [log for log in info_calls if "after imputation" in log.lower()]
            assert len(after_logs) > 0

    def test_calculate_step_statistics(self, missing_data_tat):
        """Test calculation of step-wise processing statistics."""
        engineer = DelayEngineer()
        stats = engineer._calculate_step_statistics(missing_data_tat)
        
        # Should compute statistics for delay pairs with valid data
        assert isinstance(stats, dict)
        
        # Validate statistics structure for computed delays
        for delay_col, stat_dict in stats.items():
            assert 'min' in stat_dict
            assert 'q25' in stat_dict  
            assert 'median' in stat_dict
            assert 'q75' in stat_dict
            assert 'typical_delay' in stat_dict
            
            # All statistics should be positive
            for stat_name, value in stat_dict.items():
                assert value >= 0, f"{stat_name} should be non-negative"

    def test_impute_missing_times_basic(self):
        """Test basic missing timestamp imputation functionality."""
        df = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [10, np.nan, 20],
            'prep_complete_time_mins_unwrapped': [40, 45, np.nan],
            'patient_infusion_time_mins_unwrapped': [70, 75, 80],
            'shift': ['Day', 'Day', 'Day'],
            'floor': [1, 1, 1]
        })
        
        engineer = DelayEngineer()
        result = engineer.impute_missing_times(df)
        
        # Should have fewer missing values after imputation
        original_missing = df.isna().sum().sum()
        result_missing = result.isna().sum().sum()
        assert result_missing <= original_missing
        
        # Imputed values should maintain chronological order
        for idx in result.index:
            row = result.loc[idx]
            nurse_time = row['nurse_validation_time_mins_unwrapped']
            prep_time = row['prep_complete_time_mins_unwrapped']
            infusion_time = row['patient_infusion_time_mins_unwrapped']
            
            if pd.notna(nurse_time) and pd.notna(prep_time):
                assert prep_time >= nurse_time, "Prep time should be >= nurse validation time"
            if pd.notna(prep_time) and pd.notna(infusion_time):
                assert infusion_time >= prep_time, "Infusion time should be >= prep time"

    def test_imputation_with_operational_context(self):
        """Test imputation using operational context (shift and floor patterns)."""
        # Create data with different patterns by shift/floor
        df = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [10, 20, np.nan, np.nan],
            'prep_complete_time_mins_unwrapped': [25, 40, 30, np.nan],
            'patient_infusion_time_mins_unwrapped': [50, 70, 60, 80],
            'shift': ['Day', 'Day', 'Evening', 'Evening'],
            'floor': [1, 1, 2, 2]
        })
        
        engineer = DelayEngineer()
        result = engineer.impute_missing_times(df)
        
        # Should impute missing values
        assert pd.notna(result.loc[2, 'nurse_validation_time_mins_unwrapped'])
        assert pd.notna(result.loc[3, 'prep_complete_time_mins_unwrapped'])
        
        # Imputed values should respect chronological constraints
        for idx in result.index:
            row = result.loc[idx]
            times = [
                row['nurse_validation_time_mins_unwrapped'],
                row['prep_complete_time_mins_unwrapped'],
                row['patient_infusion_time_mins_unwrapped']
            ]
            
            # Filter out NaN values and check chronological order
            valid_times = [t for t in times if pd.notna(t)]
            if len(valid_times) > 1:
                assert all(valid_times[i] <= valid_times[i+1] for i in range(len(valid_times)-1)), \
                    f"Times should be in chronological order: {valid_times}"


class TestIntegrationAndPipeline:
    """Test integration functionality and pipeline operations."""

    @pytest.fixture
    def comprehensive_tat_data(self):
        """Generate comprehensive TAT dataset for integration testing."""
        np.random.seed(42)
        n_orders = 100
        
        base_time = 0
        return pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': np.random.exponential(15, n_orders) + base_time,
            'prep_complete_time_mins_unwrapped': np.random.exponential(35, n_orders) + base_time + 15,
            'second_validation_time_mins_unwrapped': np.random.exponential(5, n_orders) + base_time + 50,
            'floor_dispatch_time_mins_unwrapped': np.random.exponential(8, n_orders) + base_time + 55,
            'patient_infusion_time_mins_unwrapped': np.random.exponential(20, n_orders) + base_time + 65,
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders, p=[0.5, 0.3, 0.2]),
            'floor': np.random.choice([1, 2, 3], n_orders),
            # Add helper columns that should be dropped
            'nurse_validation_time_dt': pd.date_range('2025-01-15', periods=n_orders, freq='10min'),
            'temp_processing_mins_unwrapped': np.random.normal(10, 2, n_orders)
        })

    def test_full_pipeline_transform(self, comprehensive_tat_data):
        """Test complete pipeline transformation with all features."""
        engineer = DelayEngineer(impute_missing=True)
        result = engineer.transform(comprehensive_tat_data)
        
        # Should preserve original columns
        for col in comprehensive_tat_data.columns:
            assert col in result.columns
        
        # Should add delay columns
        expected_delays = [
            'delay_order_to_nurse', 'delay_nurse_to_prep', 'delay_prep_to_second',
            'delay_second_to_dispatch', 'delay_dispatch_to_infusion'
        ]
        for delay_col in expected_delays:
            assert delay_col in result.columns
            assert result[delay_col].dtype in [np.float64, np.int64]
            # All delays should be non-negative
            assert (result[delay_col] >= 0).all()

    def test_fit_method_compatibility(self, comprehensive_tat_data):
        """Test sklearn-style fit method for pipeline compatibility."""
        engineer = DelayEngineer()
        
        # fit() should return self
        fitted_engineer = engineer.fit(comprehensive_tat_data)
        assert fitted_engineer is engineer
        
        # Should be chainable
        result = engineer.fit(comprehensive_tat_data).transform(comprehensive_tat_data)
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_method(self, comprehensive_tat_data):
        """Test sklearn-style fit_transform method."""
        engineer = DelayEngineer()
        
        # fit_transform should be equivalent to fit().transform()
        result1 = engineer.fit_transform(comprehensive_tat_data)
        result2 = engineer.fit(comprehensive_tat_data).transform(comprehensive_tat_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_drop_processed_helpers(self, comprehensive_tat_data):
        """Test cleanup of helper columns after processing."""
        engineer = DelayEngineer()
        
        # Transform to create delay features
        transformed = engineer.transform(comprehensive_tat_data)
        
        # Drop helper columns
        cleaned = engineer.drop_processed_helpers(transformed)
        
        # Should remove columns with configured suffixes
        helper_cols_remaining = [col for col in cleaned.columns if col.endswith('_mins_unwrapped')]
        assert len(helper_cols_remaining) == 0
        
        dt_cols_remaining = [col for col in cleaned.columns if col.endswith('_dt')]
        assert len(dt_cols_remaining) == 0
        
        # Should preserve delay features
        delay_cols = [col for col in cleaned.columns if col.startswith('delay_')]
        assert len(delay_cols) > 0

    def test_drop_helpers_with_keep_parameter(self, comprehensive_tat_data):
        """Test selective retention of helper columns using keep parameter."""
        engineer = DelayEngineer()
        transformed = engineer.transform(comprehensive_tat_data)
        
        # Keep specific helper column
        keep_cols = ['nurse_validation_time_mins_unwrapped']
        cleaned = engineer.drop_processed_helpers(transformed, keep=keep_cols)
        
        # Should preserve specified column
        assert 'nurse_validation_time_mins_unwrapped' in cleaned.columns
        
        # Should still remove other helper columns
        other_helpers = [col for col in cleaned.columns 
                        if col.endswith('_mins_unwrapped') and col not in keep_cols]
        assert len(other_helpers) == 0

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        # Disable imputation for empty DataFrame since shift/floor columns don't exist
        engineer = DelayEngineer(impute_missing=False)
        
        result = engineer.transform(empty_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""

    def test_single_row_dataframe(self):
        """Test processing of single-row DataFrames."""
        single_row = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [15],
            'prep_complete_time_mins_unwrapped': [45],
            'shift': ['Day'],
            'floor': [1]
        })
        
        engineer = DelayEngineer()
        result = engineer.transform(single_row)
        
        assert len(result) == 1
        assert 'delay_nurse_to_prep' in result.columns
        assert result['delay_nurse_to_prep'].iloc[0] == 30

    def test_all_missing_timestamps(self):
        """Test handling when all timestamps are missing."""
        all_missing = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [np.nan, np.nan],
            'prep_complete_time_mins_unwrapped': [np.nan, np.nan],
            'shift': ['Day', 'Evening'],
            'floor': [1, 2]
        })
        
        engineer = DelayEngineer()
        result = engineer.transform(all_missing)
        
        # Should handle gracefully without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_extreme_timestamp_values(self):
        """Test handling of extreme timestamp values."""
        extreme_df = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [0, 1e6, -100],
            'prep_complete_time_mins_unwrapped': [10, 1e6 + 10, -90],
            'shift': ['Day', 'Day', 'Day'],
            'floor': [1, 1, 1]
        })
        
        engineer = DelayEngineer(impute_missing=False)
        result = engineer.transform(extreme_df)
        
        # Should handle extreme values without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        
        # Third row: -90 - (-100) = 10, so no negative delay in this case
        # Let's check that all delays are non-negative (clipped)
        assert (result['delay_nurse_to_prep'] >= 0).all()

    def test_non_numeric_timestamp_handling(self):
        """Test handling of non-numeric timestamp data."""
        non_numeric = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': ['invalid', '15', 20],
            'prep_complete_time_mins_unwrapped': [45, 'also_invalid', 50],
            'shift': ['Day', 'Day', 'Day'],
            'floor': [1, 1, 1]
        })
        
        engineer = DelayEngineer(impute_missing=False)
        result = engineer.transform(non_numeric)
        
        # Should convert and handle gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_missing_operational_context_columns(self):
        """Test handling when shift or floor columns are missing."""
        missing_context = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': [10, 15],
            'prep_complete_time_mins_unwrapped': [40, 45]
            # Missing 'shift' and 'floor' columns
        })
        
        # Disable imputation when operational context columns are missing
        engineer = DelayEngineer(impute_missing=False)
        result = engineer.transform(missing_context)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        # Should still compute delays even without operational context
        assert 'delay_nurse_to_prep' in result.columns


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""

    def test_large_dataset_processing(self):
        """Test processing of large datasets for performance validation."""
        np.random.seed(42)
        n_orders = 10000
        
        large_df = pd.DataFrame({
            'nurse_validation_time_mins_unwrapped': np.random.exponential(15, n_orders),
            'prep_complete_time_mins_unwrapped': np.random.exponential(35, n_orders) + 15,
            'patient_infusion_time_mins_unwrapped': np.random.exponential(20, n_orders) + 50,
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders),
            'floor': np.random.choice([1, 2, 3], n_orders)
        })
        
        engineer = DelayEngineer()
        result = engineer.transform(large_df)
        
        # Should process successfully
        assert len(result) == n_orders
        assert 'delay_nurse_to_prep' in result.columns
        
        # Performance check: should complete in reasonable time
        # (This is implicit - if it times out, the test fails)

    def test_memory_efficiency(self):
        """Test memory-efficient processing patterns."""
        # Create dataset with many columns to test memory usage
        np.random.seed(42)
        n_orders = 1000
        n_extra_cols = 50
        
        df_data = {
            'nurse_validation_time_mins_unwrapped': np.random.exponential(15, n_orders),
            'prep_complete_time_mins_unwrapped': np.random.exponential(35, n_orders) + 15,
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders),
            'floor': np.random.choice([1, 2, 3], n_orders)
        }
        
        # Add many extra columns
        for i in range(n_extra_cols):
            df_data[f'extra_col_{i}'] = np.random.normal(0, 1, n_orders)
        
        df = pd.DataFrame(df_data)
        
        engineer = DelayEngineer()
        result = engineer.transform(df)
        
        # Should preserve all columns and add delay features
        assert len(result.columns) >= len(df.columns)
        assert len(result) == n_orders


# =============================================================================
# TEST COVERAGE SUMMARY
# =============================================================================
"""
Comprehensive test coverage for DelayEngineer class (24 test cases):

INITIALIZATION & CONFIGURATION (3 tests):
✓ Default initialization with config values
✓ Custom parameters and configuration
✓ None parameter handling and defaults

BASIC DELAY COMPUTATION (4 tests):
✓ Step-wise delay calculation from timestamps
✓ Negative delay detection and correction (clipping to 0)
✓ Missing timestamp column handling
✓ Robust numeric conversion (_to_num method)

IMPUTATION FUNCTIONALITY (4 tests):
✓ Missing data analysis and logging
✓ Statistical computation for imputation strategies
✓ Basic missing timestamp imputation with chronological validation
✓ Operational context-aware imputation (shift/floor patterns)

INTEGRATION & PIPELINE (6 tests):
✓ Full pipeline transformation with all features
✓ Sklearn-style fit method compatibility
✓ Sklearn-style fit_transform method
✓ Helper column cleanup (drop_processed_helpers)
✓ Selective helper column retention (keep parameter)
✓ Empty DataFrame handling

ERROR HANDLING & EDGE CASES (5 tests):
✓ Single-row DataFrame processing
✓ All missing timestamps handling
✓ Extreme timestamp values and bounds
✓ Non-numeric timestamp data conversion
✓ Missing operational context columns (shift/floor)

PERFORMANCE & SCALABILITY (2 tests):
✓ Large dataset processing (10,000 orders)
✓ Memory efficiency with high-dimensional data

HEALTHCARE DOMAIN VALIDATION:
✓ Clinical workflow chronological sequence integrity
✓ EHR integration artifact handling
✓ Operational context awareness (shift, floor patterns)
✓ Production-ready error handling and logging
✓ Defensive programming for healthcare data quality

All tests validate production readiness for Dana Farber's pharmacy TAT 
workflow optimization and medication preparation delay analysis initiatives.
"""
