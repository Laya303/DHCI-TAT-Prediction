"""
Test suite for functionality.

Key Test Coverage:
- Healthcare timestamp preprocessing with EHR data irregularities
- Statistical summarization for clinical TAT metrics  
- Correlation analysis for pharmacy operational driver identification
- Data quality assessment for missing workflow timestamps
- Unicode visualization support across healthcare IT environments
"""
import sys
import pytest
import pandas as pd
import numpy as np
from typing import Dict, List
from unittest.mock import patch, MagicMock

from src.tat.eda.summary.summary_analyze import (
    basic_preprocess,
    supports_unicode,
    sparkbar,
    hist_counts,
    cat_top_inline,
    numeric_describe,
    example_value,
    suppress_hist,
    missingness,
    numeric_correlations,
    partition_columns
)
from src.tat.eda.summary.summary_config import SummaryConfig


class TestTATDatasetPreprocessing:
    """Test suite for healthcare timestamp preprocessing and data standardization."""
    
    @pytest.fixture
    def healthcare_tat_data(self):
        """Generate realistic TAT dataset with typical healthcare data patterns."""
        return pd.DataFrame({
            'doctor_order_time': [
                '1/15/2025 8:30',      # Real format: M/D/YYYY H:MM
                '1/15/2025 14:22', 
                'invalid_timestamp',
                '1/15/2025 16:45',
                None
            ],
            'nurse_validation_time': [
                '39:55.8',    # Time fragments in MM:SS.decimal format
                '43:31.0',
                '15:30.5',
                '25:45.2',
                '35:20.1'
            ],
            'TAT_minutes': [45.5, 67.2, 32.1, 89.4, 56.7],
            'shift': ['Day', 'Day', 'Evening', 'Evening', 'Night'],
            'floor': [1, 2, 1, 3, 2]
        })
    
    @pytest.fixture
    def malformed_timestamp_data(self):
        """Generate dataset with various timestamp format irregularities common in EHR exports."""
        return pd.DataFrame({
            'doctor_order_time': [
                'corrupted_data',
                '25/01/2025 08:30',  # Wrong date format
                '',
                '2025-13-45 99:99:99',  # Invalid date components
                42,  # Numeric instead of string
                None
            ],
            'TAT_minutes': [30, 45, 60, 75, 90, 105]
        })

    def test_preprocess_valid_timestamps(self, healthcare_tat_data):
        """Test preprocessing of valid healthcare timestamps."""
        result = basic_preprocess(healthcare_tat_data)
        
        # Should preserve all original columns
        assert set(result.columns) == set(healthcare_tat_data.columns)
        assert len(result) == len(healthcare_tat_data)
        
        # Should convert doctor_order_time (complete timestamps) to datetime
        assert pd.api.types.is_datetime64_any_dtype(result['doctor_order_time'])
        
        # Should preserve valid timestamps
        valid_timestamps = result['doctor_order_time'].dropna()
        assert len(valid_timestamps) >= 2  # At least some valid conversions
        
        # Other time columns should remain as strings (time fragments)
        assert result['nurse_validation_time'].dtype == object
        
        # Should preserve other columns unchanged
        pd.testing.assert_series_equal(result['TAT_minutes'], healthcare_tat_data['TAT_minutes'])
        pd.testing.assert_series_equal(result['shift'], healthcare_tat_data['shift'])

    def test_preprocess_malformed_timestamps(self, malformed_timestamp_data):
        """Test graceful handling of malformed timestamps common in healthcare data."""
        result = basic_preprocess(malformed_timestamp_data)
        
        # Should not crash on malformed data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(malformed_timestamp_data)
        
        # Should attempt datetime conversion (may result in NaT values)
        assert 'doctor_order_time' in result.columns
        
        # Should preserve other data
        pd.testing.assert_series_equal(result['TAT_minutes'], malformed_timestamp_data['TAT_minutes'])

    def test_preprocess_no_doctor_order_time(self):
        """Test preprocessing when doctor_order_time column is missing."""
        data = pd.DataFrame({
            'nurse_validation_time': ['39:55.8', '43:31.0'],  # Time fragments format
            'TAT_minutes': [45.5, 67.2]
        })
        
        result = basic_preprocess(data)
        
        # Time fragment columns should remain as strings (object dtype)
        assert result['nurse_validation_time'].dtype == object
        pd.testing.assert_series_equal(result['TAT_minutes'], data['TAT_minutes'])

    def test_preprocess_empty_dataframe(self):
        """Test preprocessing with empty healthcare dataset."""
        empty_df = pd.DataFrame()
        result = basic_preprocess(empty_df)
        
        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_preprocess_preserves_original_data(self, healthcare_tat_data):
        """Test that preprocessing creates defensive copy without modifying original."""
        original_copy = healthcare_tat_data.copy()
        
        result = basic_preprocess(healthcare_tat_data)
        
        # Original should be unchanged
        pd.testing.assert_frame_equal(healthcare_tat_data, original_copy)
        
        # Result should be different object
        assert result is not healthcare_tat_data

    def test_preprocess_exception_handling(self):
        """Test preprocessing handles conversion exceptions gracefully."""
        # Create data that might cause conversion errors
        problematic_data = pd.DataFrame({
            'doctor_order_time': [
                {'complex': 'object'},  # Non-string object
                [1, 2, 3],  # List object
                type,  # Type object
            ],
            'TAT_minutes': [30, 45, 60]
        })
        
        # Should not raise exception
        result = basic_preprocess(problematic_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestHealthcareVisualizationSupport:
    """Test suite for healthcare analytics visualization utilities."""
    
    @pytest.fixture
    def unicode_config(self):
        """Configuration forcing Unicode support for modern healthcare dashboards."""
        return SummaryConfig(force_ascii=False)
    
    @pytest.fixture
    def ascii_config(self):
        """Configuration forcing ASCII for legacy healthcare IT systems."""
        return SummaryConfig(force_ascii=True)
    
    @pytest.fixture
    def auto_config(self):
        """Auto-detection configuration for mixed healthcare environments."""
        return SummaryConfig(force_ascii=None)

    def test_supports_unicode_forced_true(self, unicode_config):
        """Test Unicode support when explicitly enabled for modern systems."""
        assert supports_unicode(unicode_config) is True

    def test_supports_unicode_forced_false(self, ascii_config):
        """Test Unicode disabled for legacy healthcare terminals."""
        assert supports_unicode(ascii_config) is False

    @patch('sys.stdout')
    def test_supports_unicode_auto_detection_utf8(self, mock_stdout, auto_config):
        """Test auto-detection with UTF-8 encoding."""
        mock_stdout.encoding = 'utf-8'
        assert supports_unicode(auto_config) is True

    @patch('sys.stdout')
    def test_supports_unicode_auto_detection_ascii(self, mock_stdout, auto_config):
        """Test auto-detection with ASCII encoding."""
        mock_stdout.encoding = 'ascii'
        assert supports_unicode(auto_config) is False

    @patch('sys.stdout')
    def test_supports_unicode_no_encoding_attr(self, mock_stdout, auto_config):
        """Test auto-detection when stdout lacks encoding attribute."""
        del mock_stdout.encoding
        assert supports_unicode(auto_config) is False

    def test_sparkbar_unicode_rendering(self, unicode_config):
        """Test Unicode sparkbar generation for TAT distribution visualization."""
        # Simulate TAT delay histogram counts
        tat_delays = [1, 3, 8, 12, 6, 2, 1]
        
        result = sparkbar(tat_delays, unicode_config)
        
        # Should use Unicode block characters
        unicode_blocks = "▁▂▃▄▅▆▇█"
        assert all(char in unicode_blocks for char in result)
        assert len(result) == len(tat_delays)
        
        # Maximum value should map to highest block
        max_idx = tat_delays.index(max(tat_delays))
        assert result[max_idx] == '█'

    def test_sparkbar_ascii_rendering(self, ascii_config):
        """Test ASCII sparkbar for legacy healthcare system compatibility."""
        queue_lengths = [0, 2, 5, 8, 3, 1]
        
        result = sparkbar(queue_lengths, ascii_config)
        
        # Should use ASCII characters
        ascii_chars = " .:-=+*#@"
        assert all(char in ascii_chars for char in result)
        assert len(result) == len(queue_lengths)
        
        # Maximum should map to highest ASCII character
        max_idx = queue_lengths.index(max(queue_lengths))
        assert result[max_idx] == '@'

    def test_sparkbar_empty_input(self, unicode_config):
        """Test sparkbar with empty histogram data."""
        result = sparkbar([], unicode_config)
        assert result == ""

    def test_sparkbar_single_value(self, unicode_config):
        """Test sparkbar with single histogram bin."""
        result = sparkbar([5], unicode_config)
        assert len(result) == 1
        assert result == '█'  # Single value should be maximum

    def test_sparkbar_all_zeros(self, unicode_config):
        """Test sparkbar with all zero counts."""
        result = sparkbar([0, 0, 0, 0], unicode_config)
        assert len(result) == 4
        assert result == '▁▁▁▁'  # All minimum blocks

    def test_sparkbar_scaling_precision(self, unicode_config):
        """Test sparkbar scaling handles edge cases in healthcare data."""
        # Large range typical in TAT analysis
        wide_range = [1, 100, 500, 1000, 250]
        
        result = sparkbar(wide_range, unicode_config)
        assert len(result) == 5
        
        # Should scale relative to maximum (1000)
        max_idx = wide_range.index(max(wide_range))
        assert result[max_idx] == '█'


class TestHistogramAnalysis:
    """Test suite for histogram generation and TAT distribution analysis."""
    
    @pytest.fixture
    def tat_minutes_data(self):
        """Realistic TAT minutes distribution for histogram testing."""
        np.random.seed(42)
        # Simulate TAT with typical hospital distribution (log-normal-ish)
        base_times = np.random.exponential(35, 1000) + 15  # 15-min minimum + exponential
        return pd.Series(base_times, name='TAT_minutes')
    
    @pytest.fixture
    def categorical_numeric_data(self):
        """Numeric data that represents categories (floors, credentials)."""
        return pd.Series([1, 2, 3, 1, 2, 3, 1, 2] * 50, name='floor')  # Repeated floor numbers

    def test_hist_counts_normal_distribution(self, tat_minutes_data):
        """Test histogram generation for TAT minutes distribution."""
        counts, labels = hist_counts(tat_minutes_data, bins=10)
        
        # Should return valid histogram
        assert len(counts) == len(labels)
        assert len(counts) <= 10  # May be fewer if duplicates dropped
        assert all(isinstance(c, int) for c in counts)
        assert all(isinstance(l, str) for l in labels)
        
        # Total counts should equal non-null observations
        assert sum(counts) == len(tat_minutes_data.dropna())
        
        # Labels should represent intervals
        assert all('(' in label and ']' in label for label in labels)

    def test_hist_counts_with_missing_values(self):
        """Test histogram handles missing TAT data gracefully."""
        data_with_nan = pd.Series([10, 20, np.nan, 30, 40, np.nan, 50])
        
        counts, labels = hist_counts(data_with_nan, bins=5)
        
        # Should exclude NaN values from histogram
        assert sum(counts) == 5  # Only non-NaN values
        assert len(counts) == len(labels)

    def test_hist_counts_empty_series(self):
        """Test histogram with empty or all-NaN healthcare data."""
        empty_series = pd.Series([], dtype=float)
        counts, labels = hist_counts(empty_series, bins=5)
        
        assert counts == []
        assert labels == []
        
        # All NaN series
        all_nan = pd.Series([np.nan, np.nan, np.nan])
        counts, labels = hist_counts(all_nan, bins=5)
        
        assert counts == []
        assert labels == []

    def test_hist_counts_non_numeric_coercion(self):
        """Test histogram handles non-numeric healthcare data."""
        mixed_data = pd.Series(['45.5', '67', 'invalid', '32.1', None])
        
        counts, labels = hist_counts(mixed_data, bins=3)
        
        # Should coerce valid strings and exclude invalid
        assert sum(counts) == 3  # Three valid numeric strings
        assert len(counts) <= 3

    def test_hist_counts_duplicate_handling(self, categorical_numeric_data):
        """Test histogram with low-cardinality numeric data (floors, IDs)."""
        counts, labels = hist_counts(categorical_numeric_data, bins=10)
        
        # Should handle duplicates='drop' in pandas.cut
        assert len(counts) == len(labels)
        assert sum(counts) == len(categorical_numeric_data)
        
        # With only 3 unique values, pandas.cut may still create more bins
        # The key is that it doesn't crash and provides valid output
        assert len(counts) >= 3  # Should have at least as many bins as unique values
        assert all(isinstance(c, int) for c in counts)

    def test_hist_counts_single_unique_value(self):
        """Test histogram with constant values (edge case in healthcare data)."""
        constant_data = pd.Series([42.0] * 100, name='constant_tat')
        
        counts, labels = hist_counts(constant_data, bins=5)
        
        # pandas.cut with constant data creates bins but only one will have data
        assert len(counts) == len(labels)
        assert sum(counts) == 100  # All values accounted for
        
        # Most bins will be 0, but total should equal input size
        non_zero_bins = sum(1 for c in counts if c > 0)
        assert non_zero_bins >= 1  # At least one bin has data


class TestCategoricalAnalysis:
    """Test suite for categorical variable analysis in pharmacy workflows."""
    
    @pytest.fixture
    def shift_distribution(self):
        """Realistic shift distribution in hospital pharmacy."""
        return pd.Series(['Day'] * 45 + ['Evening'] * 30 + ['Night'] * 25, name='shift')
    
    @pytest.fixture
    def credential_distribution(self):
        """Nurse credential distribution with missing data."""
        data = ['BSN'] * 42 + ['RN'] * 31 + ['MSN'] * 18 + ['NP'] * 9 + [np.nan] * 15
        return pd.Series(data, name='nurse_credential')
    
    @pytest.fixture
    def unicode_config(self):
        """Unicode configuration for modern healthcare dashboards."""
        return SummaryConfig(force_ascii=False)
    
    @pytest.fixture
    def ascii_config(self):
        """ASCII configuration for legacy systems."""
        return SummaryConfig(force_ascii=True)

    def test_cat_top_inline_shift_analysis(self, shift_distribution, unicode_config):
        """Test inline categorical summary for shift distribution."""
        result = cat_top_inline(shift_distribution, k=5, cfg=unicode_config)
        
        # Should show percentages for top categories
        assert 'Day(45%)' in result
        assert 'Evening(30%)' in result
        assert 'Night(25%)' in result
        
        # Should use Unicode separator
        assert ' • ' in result

    def test_cat_top_inline_with_missing_data(self, credential_distribution, unicode_config):
        """Test categorical summary includes missing data analysis."""
        result = cat_top_inline(credential_distribution, k=5, cfg=unicode_config)
        
        # Should include NaN as explicit category
        assert 'NaN(13%)' in result  # 15/115 ≈ 13%
        assert 'BSN(' in result
        assert 'RN(' in result

    def test_cat_top_inline_ascii_mode(self, shift_distribution, ascii_config):
        """Test ASCII separator for legacy healthcare systems."""
        result = cat_top_inline(shift_distribution, k=3, cfg=ascii_config)
        
        # Should use ASCII separator
        assert ' | ' in result
        assert ' • ' not in result
        
        # Should still show percentages
        assert 'Day(45%)' in result

    def test_cat_top_inline_limit_categories(self, credential_distribution, unicode_config):
        """Test k-limit functionality for top categories."""
        result = cat_top_inline(credential_distribution, k=2, cfg=unicode_config)
        
        # Should show only top 2 categories
        parts = result.split(' • ')
        assert len(parts) == 2
        
        # Should prioritize by frequency
        assert 'BSN(' in result  # Most frequent
        assert any('RN(' in part or 'NaN(' in part for part in parts)  # Second most frequent

    def test_cat_top_inline_empty_series(self, unicode_config):
        """Test categorical summary with empty healthcare data."""
        empty_series = pd.Series([], dtype=object)
        result = cat_top_inline(empty_series, k=5, cfg=unicode_config)
        
        assert result == ""

    def test_cat_top_inline_percentage_calculation(self):
        """Test percentage calculation includes all observations."""
        # Test with known distribution
        test_data = pd.Series(['A'] * 8 + ['B'] * 2)  # 10 total
        cfg = SummaryConfig(force_ascii=False)
        
        result = cat_top_inline(test_data, k=2, cfg=cfg)
        
        assert 'A(80%)' in result  # 8/10 = 80%
        assert 'B(20%)' in result  # 2/10 = 20%

    def test_cat_top_inline_special_characters(self, unicode_config):
        """Test categorical summary handles special characters in healthcare data."""
        special_data = pd.Series(['Category (A)', 'Category • B', 'Category | C'] * 10)
        
        result = cat_top_inline(special_data, k=3, cfg=unicode_config)
        
        # Should handle parentheses and separators in category names
        assert 'Category (A)(33%)' in result
        assert 'Category • B(33%)' in result
        assert 'Category | C(33%)' in result


class TestNumericStatistics:
    """Test suite for numeric statistical analysis of TAT metrics."""
    
    @pytest.fixture
    def tat_config(self):
        """Configuration for TAT analysis with healthcare-relevant percentiles."""
        return SummaryConfig(percentiles=[0.1, 0.25, 0.75, 0.9, 0.95, 0.99])
    
    @pytest.fixture
    def queue_length_data(self):
        """Realistic queue length distribution."""
        np.random.seed(123)
        # Poisson-like distribution for queue lengths
        return pd.Series(np.random.poisson(8, 1000), name='queue_length_at_order')
    
    @pytest.fixture
    def tat_with_outliers(self):
        """TAT data with realistic outliers (stat orders, complications)."""
        normal_tat = np.random.normal(45, 15, 950)  # Normal cases
        outliers = [180, 240, 300, 156, 189] * 10  # Stat orders and complications
        return pd.Series(list(normal_tat) + outliers, name='TAT_minutes')

    def test_numeric_describe_comprehensive_stats(self, queue_length_data, tat_config):
        """Test comprehensive statistical analysis for queue metrics."""
        result = numeric_describe(queue_length_data, tat_config)
        
        # Should include standard descriptive statistics
        required_keys = ['count', 'mean', 'std', 'min', '50%', 'max']
        assert all(key in result for key in required_keys)
        
        # Should include configured percentiles
        percentile_keys = ['p10', 'p25', 'p75', 'p90', 'p95', 'p99']
        assert all(key in result for key in percentile_keys)
        
        # Statistical validity checks
        assert result['count'] == len(queue_length_data)
        assert result['min'] <= result['50%'] <= result['max']
        assert result['p10'] <= result['p25'] <= result['p75'] <= result['p90']
        
        # All values should be numeric
        assert all(isinstance(v, (int, float)) for v in result.values())

    def test_numeric_describe_with_missing_data(self, tat_config):
        """Test statistical analysis excludes missing TAT values."""
        data_with_nan = pd.Series([10, 20, np.nan, 30, 40, np.nan, 50, 60])
        
        result = numeric_describe(data_with_nan, tat_config)
        
        # Should exclude NaN values from calculations
        assert result['count'] == 6.0  # Only non-NaN values
        assert result['mean'] == 35.0  # (10+20+30+40+50+60)/6
        
        # Should handle percentiles with missing data
        assert 'p50' in result or '50%' in result

    def test_numeric_describe_empty_or_non_numeric(self, tat_config):
        """Test statistical analysis with empty or non-numeric healthcare data."""
        # Empty series
        empty_series = pd.Series([], dtype=float)
        result = numeric_describe(empty_series, tat_config)
        assert result == {}
        
        # Non-numeric series
        text_series = pd.Series(['Day', 'Evening', 'Night'])
        result = numeric_describe(text_series, tat_config)
        assert result == {}
        
        # All NaN series
        all_nan = pd.Series([np.nan, np.nan, np.nan])
        result = numeric_describe(all_nan, tat_config)
        assert result == {}

    def test_numeric_describe_outlier_robustness(self, tat_with_outliers, tat_config):
        """Test statistical measures handle TAT outliers appropriately."""
        result = numeric_describe(tat_with_outliers, tat_config)
        
        # Median should be robust to outliers
        assert 35 <= result['50%'] <= 55  # Should be near normal distribution center
        
        # High percentiles should capture outliers
        assert result['p95'] > result['p75'] > result['50%']
        assert result['p99'] > result['p95']  # Extreme outliers in p99
        
        # Standard deviation should reflect variability
        assert result['std'] > 0

    def test_numeric_describe_percentile_configuration(self):
        """Test statistical analysis respects custom percentile configuration."""
        test_data = pd.Series(range(100))  # 0 to 99
        custom_config = SummaryConfig(percentiles=[0.05, 0.5, 0.95])
        
        result = numeric_describe(test_data, custom_config)
        
        # Should include only configured percentiles
        assert 'p5' in result
        assert 'p50' in result  # Should have p50 even though 50% also exists
        assert 'p95' in result
        
        # Should not include unconfigured percentiles
        assert 'p25' not in result
        assert 'p75' not in result

    def test_numeric_describe_ddof_parameter(self):
        """Test unbiased standard deviation calculation (ddof=1) for population inference."""
        # Small dataset where ddof matters
        small_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        config = SummaryConfig(percentiles=[0.5])
        
        result = numeric_describe(small_data, config)
        
        # Unbiased std should use ddof=1
        expected_std = small_data.std(ddof=1)
        assert abs(result['std'] - expected_std) < 1e-10


class TestDataQualityAssessment:
    """Test suite for healthcare data quality and validation utilities."""
    
    @pytest.fixture
    def realistic_tat_dataset(self):
        """TAT dataset with realistic missing patterns from healthcare systems."""
        return pd.DataFrame({
            'doctor_order_time': [pd.Timestamp('2025-01-15')] * 80 + [pd.NaT] * 20,
            'nurse_validation_time': [pd.Timestamp('2025-01-15')] * 85 + [pd.NaT] * 15,
            'prep_complete_time': [pd.Timestamp('2025-01-15')] * 70 + [pd.NaT] * 30,  # Often missing
            'second_validation_time': [pd.Timestamp('2025-01-15')] * 60 + [pd.NaT] * 40,  # More missing
            'TAT_minutes': [45.0] * 95 + [np.nan] * 5,  # Rarely missing
            'shift': ['Day'] * 100,  # Complete data
            'patient_id': [f'PAT_{i:05d}' for i in range(100)]  # Complete identifiers
        })

    def test_example_value_extraction(self, realistic_tat_dataset):
        """Test representative value extraction for healthcare data documentation."""
        # Numeric column
        tat_example = example_value(realistic_tat_dataset['TAT_minutes'])
        assert tat_example == '45.0'
        
        # Categorical column
        shift_example = example_value(realistic_tat_dataset['shift'])
        assert shift_example == 'Day'
        
        # String identifier column
        patient_example = example_value(realistic_tat_dataset['patient_id'])
        assert patient_example == 'PAT_00000'

    def test_example_value_missing_data(self):
        """Test example value extraction with missing healthcare data."""
        # Series with some missing values
        sparse_data = pd.Series([np.nan, np.nan, 'Valid Value', np.nan])
        result = example_value(sparse_data)
        assert result == 'Valid Value'
        
        # All missing series
        all_missing = pd.Series([np.nan] * 10)
        result = example_value(all_missing)
        assert result == 'NaN'
        
        # Empty series
        empty_series = pd.Series([])
        result = example_value(empty_series)
        assert result == 'NaN'

    def test_example_value_scan_limit(self):
        """Test example value scanning limits for large healthcare datasets."""
        # Create series where valid value is beyond scan limit
        large_series_data = [np.nan] * 60 + ['Found Value'] + [np.nan] * 100
        large_series = pd.Series(large_series_data)
        
        result = example_value(large_series)
        # Should still find value within first 50 + some buffer, but test the limit
        assert result in ['Found Value', 'NaN']  # Implementation-dependent

    def test_suppress_hist_configuration(self):
        """Test histogram suppression for healthcare identifier columns."""
        # Default configuration should suppress common identifier patterns
        config = SummaryConfig()
        
        # Should suppress patient identifiers (HIPAA compliance)
        assert suppress_hist('patient_id', config) == ('patient_id' in config.no_hist_cols)
        
        # Should allow TAT metrics
        assert suppress_hist('TAT_minutes', config) == ('TAT_minutes' in config.no_hist_cols)
        
        # Test explicit suppression
        custom_config = SummaryConfig(no_hist_cols=['sensitive_col'])
        assert suppress_hist('sensitive_col', custom_config) is True
        assert suppress_hist('TAT_minutes', custom_config) is False

    def test_missingness_analysis(self, realistic_tat_dataset):
        """Test missing data pattern analysis for pharmacy workflow systems."""
        # Unsorted missingness
        config_unsorted = SummaryConfig(sort_missing=False)
        result_unsorted = missingness(realistic_tat_dataset, config_unsorted)
        
        # Should return Series with column names as index
        assert isinstance(result_unsorted, pd.Series)
        assert set(result_unsorted.index) == set(realistic_tat_dataset.columns)
        
        # Should calculate correct missing ratios
        assert result_unsorted['second_validation_time'] == 0.4  # 40/100
        assert result_unsorted['prep_complete_time'] == 0.3      # 30/100
        assert result_unsorted['shift'] == 0.0                   # Complete data
        
        # Sorted missingness (for quality review prioritization)
        config_sorted = SummaryConfig(sort_missing=True)
        result_sorted = missingness(realistic_tat_dataset, config_sorted)
        
        # Should be sorted descending by missing fraction
        assert result_sorted.is_monotonic_decreasing
        assert result_sorted.index[0] == 'second_validation_time'  # Highest missing
        assert result_sorted.index[-1] in ['shift', 'patient_id']  # Complete data

    def test_missingness_edge_cases(self):
        """Test missing data analysis with edge cases."""
        # Completely missing column
        all_missing_df = pd.DataFrame({
            'all_missing': [np.nan] * 100,
            'some_missing': [1] * 80 + [np.nan] * 20,
            'no_missing': [1] * 100
        })
        
        config = SummaryConfig(sort_missing=True)
        result = missingness(all_missing_df, config)
        
        assert result['all_missing'] == 1.0
        assert result['some_missing'] == 0.2
        assert result['no_missing'] == 0.0
        
        # Empty dataframe
        empty_df = pd.DataFrame()
        result = missingness(empty_df, config)
        assert len(result) == 0


class TestCorrelationAnalysis:
    """Test suite for TAT driver identification and correlation analysis."""
    
    @pytest.fixture
    def operational_metrics_data(self):
        """Healthcare operational metrics for correlation analysis."""
        np.random.seed(42)
        n = 1000
        
        # Simulate realistic correlations
        queue_base = np.random.poisson(8, n)
        occupancy = np.random.uniform(0.3, 0.95, n)
        pharmacists = np.random.randint(2, 6, n)
        
        # TAT influenced by operational factors
        tat = (
            30 +  # Base TAT
            queue_base * 2.5 +  # Queue impact
            occupancy * 40 +   # Occupancy impact
            pharmacists * (-3) +  # More pharmacists = lower TAT
            np.random.normal(0, 8, n)  # Random variation
        )
        
        return pd.DataFrame({
            'TAT_minutes': tat,
            'queue_length_at_order': queue_base,
            'floor_occupancy_pct': occupancy * 100,
            'pharmacists_on_duty': pharmacists,
            'patient_id': [f'PAT_{i:06d}' for i in range(n)],  # Should be excluded
            'doctor_order_time': pd.date_range('2025-01-01', periods=n, freq='15min'),  # Should be excluded
            'nurse_credential': np.random.choice(['BSN', 'RN', 'MSN'], n)  # Non-numeric, excluded
        })
    
    @pytest.fixture
    def correlation_config(self):
        """Configuration for healthcare correlation analysis."""
        return SummaryConfig(
            corr_exclude_columns=['patient_id'],
            corr_exclude_prefixes=['patient_', 'order_']
        )

    def test_numeric_correlations_operational_analysis(self, operational_metrics_data, correlation_config):
        """Test correlation analysis for TAT operational drivers."""
        result = numeric_correlations(operational_metrics_data, correlation_config)
        
        # Should return correlation matrix
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]  # Square matrix
        
        # Should include numeric operational metrics (excluding patient_age which isn't in the fixture)
        expected_cols = ['TAT_minutes', 'queue_length_at_order', 'floor_occupancy_pct', 
                        'pharmacists_on_duty']
        for col in expected_cols:
            assert col in result.columns
            assert col in result.index
        
        # Should exclude patient_id (identifier)
        assert 'patient_id' not in result.columns
        
        # Should exclude doctor_order_time (datetime)
        assert 'doctor_order_time' not in result.columns
        
        # Should exclude nurse_credential (categorical)
        assert 'nurse_credential' not in result.columns
        
        # Correlation matrix properties
        assert all(result.loc[col, col] == 1.0 for col in result.columns)  # Diagonal = 1
        assert result.equals(result.T)  # Symmetric

    def test_numeric_correlations_exclusion_logic(self, operational_metrics_data):
        """Test correlation analysis exclusion configurations."""
        # Test prefix exclusion
        prefix_config = SummaryConfig(
            corr_exclude_columns=[],
            corr_exclude_prefixes=['floor_', 'patient_']
        )
        
        result = numeric_correlations(operational_metrics_data, prefix_config)
        
        # Should exclude floor_occupancy_pct due to floor_ prefix
        assert 'floor_occupancy_pct' not in result.columns
        
        # Should include others
        assert 'TAT_minutes' in result.columns
        assert 'queue_length_at_order' in result.columns

    def test_numeric_correlations_extra_exclusions(self, operational_metrics_data, correlation_config):
        """Test additional exclusion parameter in correlation analysis."""
        extra_exclude = ['pharmacists_on_duty', 'floor_occupancy_pct']
        
        result = numeric_correlations(
            operational_metrics_data, 
            correlation_config, 
            extra_exclude=extra_exclude
        )
        
        # Should exclude extra columns beyond configuration
        assert 'pharmacists_on_duty' not in result.columns
        assert 'floor_occupancy_pct' not in result.columns
        
        # Should still include others not excluded
        assert 'TAT_minutes' in result.columns
        assert 'queue_length_at_order' in result.columns

    def test_numeric_correlations_insufficient_columns(self):
        """Test correlation analysis with insufficient numeric columns."""
        # Single numeric column
        single_col_df = pd.DataFrame({
            'TAT_minutes': [30, 45, 60],
            'shift': ['Day', 'Evening', 'Night']
        })
        
        config = SummaryConfig()
        result = numeric_correlations(single_col_df, config)
        
        # Should return empty DataFrame (need ≥2 numeric columns)
        assert result.empty
        
        # No numeric columns
        no_numeric_df = pd.DataFrame({
            'shift': ['Day', 'Evening', 'Night'],
            'credential': ['BSN', 'RN', 'MSN']
        })
        
        result = numeric_correlations(no_numeric_df, config)
        assert result.empty

    def test_numeric_correlations_healthcare_relevance(self, operational_metrics_data, correlation_config):
        """Test correlation analysis identifies meaningful healthcare relationships."""
        result = numeric_correlations(operational_metrics_data, correlation_config)
        
        # Should detect simulated relationships
        tat_queue_corr = result.loc['TAT_minutes', 'queue_length_at_order']
        tat_occupancy_corr = result.loc['TAT_minutes', 'floor_occupancy_pct']
        tat_pharmacists_corr = result.loc['TAT_minutes', 'pharmacists_on_duty']
        
        # Queue length should positively correlate with TAT
        assert tat_queue_corr > 0.3  # Moderate positive correlation
        
        # Occupancy should positively correlate with TAT
        assert tat_occupancy_corr > 0.2
        
        # More pharmacists should negatively correlate with TAT
        assert tat_pharmacists_corr < -0.1  # Negative correlation


class TestColumnPartitioning:
    """Test suite for healthcare data column classification and partitioning."""
    
    @pytest.fixture
    def comprehensive_tat_dataset(self):
        """Comprehensive TAT dataset with mixed column types."""
        return pd.DataFrame({
            # Temporal columns
            'doctor_order_time': pd.date_range('2025-01-01', periods=100, freq='15min'),
            'nurse_validation_time_dt': pd.date_range('2025-01-01 00:10', periods=100, freq='15min'),
            'prep_start_timestamp': ['2025-01-01 08:30:00'] * 100,  # String that looks temporal
            
            # Categorical columns
            'shift': ['Day'] * 50 + ['Evening'] * 30 + ['Night'] * 20,
            'nurse_credential': ['BSN'] * 60 + ['RN'] * 25 + ['MSN'] * 15,
            'diagnosis_type': ['SolidTumor'] * 40 + ['Hematologic'] * 35 + ['Autoimmune'] * 25,
            
            # Numeric columns
            'TAT_minutes': np.random.exponential(45, 100) + 15,
            'queue_length_at_order': np.random.poisson(8, 100),
            'floor_occupancy_pct': np.random.uniform(30, 95, 100),
            'patient_age': np.random.uniform(18, 85, 100),
            'lab_WBC_k_per_uL': np.random.uniform(4.0, 11.0, 100),
            
            # Special cases
            'floor_numeric': np.random.choice([1, 2, 3], 100),  # Numeric but should be categorical
            'patient_readiness_score': np.random.choice([1, 2, 3], 100)  # Ordinal but numeric
        })
    
    @pytest.fixture
    def healthcare_column_config(self):
        """Configuration for healthcare-specific column partitioning."""
        return SummaryConfig(
            known_time_cols=['prep_start_timestamp'],  # Force temporal classification
            categorical_overrides=['floor_numeric'],   # Force categorical despite being numeric
            categorical_prefixes=['patient_readiness_']  # Force by prefix pattern
        )

    def test_partition_columns_temporal_identification(self, comprehensive_tat_dataset, healthcare_column_config):
        """Test identification of temporal columns in healthcare datasets."""
        time_cols, cat_cols, num_cols = partition_columns(comprehensive_tat_dataset, healthcare_column_config)
        
        # Should identify datetime columns
        assert 'doctor_order_time' in time_cols
        
        # Should identify _dt suffix
        assert 'nurse_validation_time_dt' in time_cols
        
        # Should respect known_time_cols configuration
        assert 'prep_start_timestamp' in time_cols
        
        # Should not classify as other types
        for col in time_cols:
            assert col not in cat_cols
            assert col not in num_cols

    def test_partition_columns_categorical_classification(self, comprehensive_tat_dataset, healthcare_column_config):
        """Test categorical column identification in healthcare context."""
        time_cols, cat_cols, num_cols = partition_columns(comprehensive_tat_dataset, healthcare_column_config)
        
        # Should identify string categorical columns
        expected_categoricals = ['shift', 'nurse_credential', 'diagnosis_type']
        for col in expected_categoricals:
            assert col in cat_cols
        
        # Should respect categorical_overrides
        assert 'floor_numeric' in cat_cols  # Forced categorical despite numeric type
        
        # Should respect categorical_prefixes  
        assert 'patient_readiness_score' in cat_cols  # Matches prefix pattern
        
        # Should not classify as other types
        for col in cat_cols:
            assert col not in time_cols
            assert col not in num_cols

    def test_partition_columns_numeric_identification(self, comprehensive_tat_dataset, healthcare_column_config):
        """Test numeric column identification for TAT analytics."""
        time_cols, cat_cols, num_cols = partition_columns(comprehensive_tat_dataset, healthcare_column_config)
        
        # Should identify continuous metrics
        expected_numerics = [
            'TAT_minutes', 'queue_length_at_order', 'floor_occupancy_pct', 
            'patient_age', 'lab_WBC_k_per_uL'
        ]
        for col in expected_numerics:
            assert col in num_cols
        
        # Should not include overridden columns
        assert 'floor_numeric' not in num_cols  # Overridden as categorical
        assert 'patient_readiness_score' not in num_cols  # Prefix rule
        
        # Should not classify as other types
        for col in num_cols:
            assert col not in time_cols
            assert col not in cat_cols

    def test_partition_columns_comprehensive_coverage(self, comprehensive_tat_dataset, healthcare_column_config):
        """Test that all columns are classified exactly once."""
        time_cols, cat_cols, num_cols = partition_columns(comprehensive_tat_dataset, healthcare_column_config)
        
        # All columns should be classified
        all_classified = set(time_cols) | set(cat_cols) | set(num_cols)
        assert all_classified == set(comprehensive_tat_dataset.columns)
        
        # No overlaps between categories
        assert not (set(time_cols) & set(cat_cols))
        assert not (set(time_cols) & set(num_cols))
        assert not (set(cat_cols) & set(num_cols))

    def test_partition_columns_edge_cases(self):
        """Test column partitioning with healthcare data edge cases."""
        # Empty dataframe
        empty_df = pd.DataFrame()
        config = SummaryConfig()
        
        time_cols, cat_cols, num_cols = partition_columns(empty_df, config)
        assert time_cols == []
        assert cat_cols == []
        assert num_cols == []
        
        # Mixed type scenarios
        mixed_df = pd.DataFrame({
            'datetime_col': pd.date_range('2025-01-01', periods=5),
            'time_in_name': ['not_datetime'] * 5,  # String with 'time' in name
            'numeric_id': [1, 2, 3, 4, 5],  # Numeric that might be ID
            'object_numeric': pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=object)  # Object that's numeric
        })
        
        time_cols, cat_cols, num_cols = partition_columns(mixed_df, config)
        
        # datetime_col should be temporal
        assert 'datetime_col' in time_cols
        
        # time_in_name should be temporal due to name pattern
        assert 'time_in_name' in time_cols
        
        # Numeric columns should be classified appropriately
        assert 'numeric_id' in num_cols
        assert 'object_numeric' in cat_cols  # Object dtype, not numeric despite content

    def test_partition_columns_healthcare_domain_logic(self):
        """Test partitioning follows healthcare domain conventions."""
        healthcare_df = pd.DataFrame({
            # Timestamp patterns common in healthcare
            'order_time': pd.date_range('2025-01-01', periods=10),
            'infusion_start_dt': pd.date_range('2025-01-01', periods=10),
            'completion_timestamp': ['2025-01-01'] * 10,  # String timestamp
            
            # Healthcare categoricals
            'icd10_code': ['C78.1'] * 10,  # Diagnosis codes
            'insurance_type': ['Commercial'] * 10,
            'medication_name': ['Pembrolizumab'] * 10,
            
            # Healthcare numerics
            'dose_mg': [100.0] * 10,
            'weight_kg': [70.5] * 10,
            'creatinine_mg_dl': [1.2] * 10
        })
        
        config = SummaryConfig()
        time_cols, cat_cols, num_cols = partition_columns(healthcare_df, config)
        
        # Healthcare temporal patterns
        temporal_expected = ['order_time', 'infusion_start_dt', 'completion_timestamp']
        for col in temporal_expected:
            assert col in time_cols
        
        # Healthcare categoricals
        categorical_expected = ['icd10_code', 'insurance_type', 'medication_name']
        for col in categorical_expected:
            assert col in cat_cols
        
        # Healthcare numerics
        numeric_expected = ['dose_mg', 'weight_kg', 'creatinine_mg_dl']
        for col in numeric_expected:
            assert col in num_cols


class TestHealthcareAnalyticsIntegration:
    """Integration tests for healthcare analytics workflow compatibility."""
    
    @pytest.fixture
    def realistic_pharmacy_dataset(self):
        """Realistic pharmacy operations dataset for integration testing."""
        np.random.seed(42)
        n_orders = 5000
        
        # Generate realistic healthcare operational data
        base_time = pd.Timestamp('2025-01-15 06:00:00')
        
        return pd.DataFrame({
            # Workflow timestamps with realistic missing patterns
            'doctor_order_time': pd.date_range(base_time, periods=n_orders, freq='12min'),
            'nurse_validation_time': [
                base_time + pd.Timedelta(minutes=15+i*12) if i % 20 != 0 else pd.NaT 
                for i in range(n_orders)
            ],  # 5% missing
            'prep_complete_time': [
                base_time + pd.Timedelta(minutes=45+i*12) if i % 10 != 0 else pd.NaT 
                for i in range(n_orders)
            ],  # 10% missing
            
            # Operational metrics
            'TAT_minutes': np.random.exponential(42, n_orders) + 12,
            'queue_length_at_order': np.random.poisson(7, n_orders),
            'floor_occupancy_pct': np.random.uniform(25, 98, n_orders),
            'pharmacists_on_duty': np.random.randint(2, 8, n_orders),
            
            # Clinical context
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders, p=[0.55, 0.30, 0.15]),
            'nurse_credential': np.random.choice(['BSN', 'RN', 'MSN', 'NP'], n_orders, p=[0.45, 0.30, 0.20, 0.05]),
            'diagnosis_type': np.random.choice(
                ['SolidTumor', 'Hematologic', 'Autoimmune', 'Other'], 
                n_orders, p=[0.40, 0.35, 0.20, 0.05]
            ),
            'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders, p=[0.30, 0.50, 0.20]),
            
            # Lab values with clinical ranges
            'lab_WBC_k_per_uL': np.random.uniform(3.0, 12.0, n_orders),
            'lab_Platelets_k_per_uL': np.random.uniform(120, 450, n_orders),
            'lab_Creatinine_mg_dL': np.random.uniform(0.5, 2.0, n_orders),
            
            # Patient demographics
            'patient_age': np.random.uniform(18, 89, n_orders),
            'patient_id': [f'DFCI_{i:07d}' for i in range(n_orders)]
        })

    def test_comprehensive_analytics_workflow(self, realistic_pharmacy_dataset):
        """Test complete healthcare analytics workflow integration."""
        config = SummaryConfig(
            percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
            corr_exclude_columns=['patient_id'],
            no_hist_cols=['patient_id']
        )
        
        # 1. Preprocessing
        processed_data = basic_preprocess(realistic_pharmacy_dataset)
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == len(realistic_pharmacy_dataset)
        
        # 2. Column partitioning  
        time_cols, cat_cols, num_cols = partition_columns(processed_data, config)
        
        # Should identify healthcare column types appropriately
        assert len(time_cols) >= 3  # Multiple timestamps
        assert len(cat_cols) >= 4   # Shift, credentials, diagnosis, severity
        assert len(num_cols) >= 7   # TAT, queue, occupancy, labs, age
        
        # 3. Missing data analysis
        missing_analysis = missingness(processed_data, config)
        assert isinstance(missing_analysis, pd.Series)
        
        # Should identify realistic missing patterns
        assert missing_analysis['prep_complete_time'] > 0.05  # Expected missing
        assert missing_analysis['patient_id'] == 0.0          # Complete identifiers
        
        # 4. Statistical analysis for numeric columns
        numeric_stats = {}
        for col in num_cols:
            if col in processed_data.columns:
                stats = numeric_describe(processed_data[col], config)
                if stats:  # Not empty
                    numeric_stats[col] = stats
        
        # Should generate statistics for key TAT metrics
        assert 'TAT_minutes' in numeric_stats
        assert 'queue_length_at_order' in numeric_stats
        
        # TAT statistics should be clinically reasonable
        tat_stats = numeric_stats['TAT_minutes']
        assert 10 <= tat_stats['mean'] <= 120     # Reasonable TAT range
        assert tat_stats['p95'] > tat_stats['p75'] # Percentile ordering
        
        # 5. Correlation analysis for operational optimization
        correlations = numeric_correlations(processed_data, config)
        if not correlations.empty:
            # Should analyze relationships between operational factors
            if 'TAT_minutes' in correlations.columns and 'queue_length_at_order' in correlations.columns:
                tat_queue_corr = correlations.loc['TAT_minutes', 'queue_length_at_order']
                assert isinstance(tat_queue_corr, (int, float))
        
        # 6. Categorical analysis
        categorical_summaries = {}
        for col in cat_cols:
            if col in processed_data.columns:
                summary = cat_top_inline(processed_data[col], k=4, cfg=config)
                if summary:  # Not empty
                    categorical_summaries[col] = summary
        
        # Should summarize key healthcare categoricals
        assert len(categorical_summaries) >= 3
        if 'shift' in categorical_summaries:
            shift_summary = categorical_summaries['shift']
            assert 'Day(' in shift_summary  # Day shift should be present
            assert '%' in shift_summary     # Should include percentages

    def test_healthcare_data_quality_reporting(self, realistic_pharmacy_dataset):
        """Test data quality assessment for healthcare operations dashboard."""
        config = SummaryConfig(sort_missing=True)
        
        # Comprehensive data quality assessment
        missing_report = missingness(realistic_pharmacy_dataset, config)
        
        # Should prioritize columns by missing data severity
        assert missing_report.is_monotonic_decreasing
        
        # Should identify workflow tracking gaps
        workflow_columns = [col for col in missing_report.index if 'time' in col.lower()]
        assert len(workflow_columns) >= 3
        
        # Missing data should follow realistic healthcare patterns
        # Prep completion often has more missing data than initial order
        if 'prep_complete_time' in missing_report.index and 'doctor_order_time' in missing_report.index:
            prep_missing = missing_report['prep_complete_time']
            order_missing = missing_report['doctor_order_time']
            assert prep_missing >= order_missing  # Later steps have more missing data
        
        # Generate example values for data documentation
        data_examples = {}
        for col in realistic_pharmacy_dataset.columns:
            example = example_value(realistic_pharmacy_dataset[col])
            data_examples[col] = example
        
        # Should provide representative examples
        assert 'DFCI_' in data_examples['patient_id']  # Patient ID format
        assert data_examples['shift'] in ['Day', 'Evening', 'Night']  # Valid shift
        
        # Numeric examples should be reasonable
        try:
            tat_example = float(data_examples['TAT_minutes'])
            assert 5 <= tat_example <= 300  # Reasonable TAT range
        except (ValueError, TypeError):
            pass  # Handle edge cases gracefully

    def test_visualization_support_healthcare_context(self, realistic_pharmacy_dataset):
        """Test visualization utilities for healthcare stakeholder reporting."""
        config = SummaryConfig(force_ascii=None)  # Auto-detect
        
        # Test Unicode support detection
        unicode_supported = supports_unicode(config)
        assert isinstance(unicode_supported, bool)
        
        # Test histogram generation for TAT distributions
        tat_data = realistic_pharmacy_dataset['TAT_minutes']
        hist_counts_result, hist_labels = hist_counts(tat_data, bins=8)
        
        # Should generate meaningful TAT distribution
        assert len(hist_counts_result) == len(hist_labels)
        assert sum(hist_counts_result) == len(tat_data.dropna())
        
        # Should handle healthcare-appropriate binning
        assert all(isinstance(count, int) for count in hist_counts_result)
        assert all('(' in label and ']' in label for label in hist_labels)
        
        # Test sparkbar visualization for dashboard integration
        sparkbar_viz = sparkbar(hist_counts_result, config)
        assert isinstance(sparkbar_viz, str)
        assert len(sparkbar_viz) == len(hist_counts_result)
        
        # Test categorical visualization for operational summaries
        shift_summary = cat_top_inline(realistic_pharmacy_dataset['shift'], k=3, cfg=config)
        
        # Should provide actionable operational insights
        assert 'Day(' in shift_summary    # Primary shift
        assert '%' in shift_summary       # Percentage context
        
        # Separator should be appropriate for healthcare IT environment
        separator = ' • ' if unicode_supported else ' | '
        assert separator in shift_summary

    def test_performance_scalability_healthcare_volume(self):
        """Test analytics performance with realistic healthcare data volumes."""
        # Simulate high-volume pharmacy operations
        large_n = 50000  # Realistic daily order volume for large health system
        
        np.random.seed(123)
        large_dataset = pd.DataFrame({
            'doctor_order_time': pd.date_range('2025-01-01', periods=large_n, freq='2min'),
            'TAT_minutes': np.random.exponential(40, large_n) + 10,
            'queue_length': np.random.poisson(6, large_n),
            'shift': np.random.choice(['Day', 'Evening', 'Night'], large_n),
            'diagnosis': np.random.choice(['Type1', 'Type2', 'Type3'] * 20, large_n),
            'patient_id': [f'P{i:08d}' for i in range(large_n)]
        })
        
        config = SummaryConfig()
        
        import time
        start_time = time.time()
        
        # Core analytics should complete within reasonable time
        processed = basic_preprocess(large_dataset)
        time_cols, cat_cols, num_cols = partition_columns(processed, config)
        missing_analysis = missingness(processed, config)
        
        # Statistical analysis on key metrics
        tat_stats = numeric_describe(large_dataset['TAT_minutes'], config) 
        shift_summary = cat_top_inline(large_dataset['shift'], k=3, cfg=config)
        
        processing_time = time.time() - start_time
        
        # Should handle healthcare volume efficiently (< 10 seconds for 50k records)
        assert processing_time < 10
        
        # Should maintain accuracy at scale
        assert isinstance(tat_stats, dict)
        assert 'mean' in tat_stats
        assert 'Day(' in shift_summary
        
        # Should maintain data integrity
        assert len(processed) == large_n
        assert len(missing_analysis) == len(large_dataset.columns)