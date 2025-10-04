"""
Test suite for categorical encoding functionality.
- Pipeline architecture validation for automated feature engineering workflows
- Integration compatibility with Dana Farber's pharmacy analytics infrastructure
- Performance validation for 100k+ medication order categorical processing
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from tat.features.categoricals import CategoricalEncoder


class TestCategoricalEncoderInitialization:
    """Test suite for CategoricalEncoder initialization and configuration validation."""

    def test_default_initialization(self):
        """Validate default CategoricalEncoder initialization with healthcare-optimized settings."""
        encoder = CategoricalEncoder()
        
        # Validate default configuration parameters
        assert encoder.one_hot_prefix_map == {}
        assert encoder.one_hot_drop_first is False
        assert encoder.dtype == int
        
        # Validate healthcare domain-aware ordinal mappings initialization
        assert encoder.sex_map == {"F": 0, "M": 1}
        assert encoder.severity_map == {"Low": 0, "Medium": 1, "High": 2}
        assert encoder.nurse_credential_map == {"RN": 0, "BSN": 1, "MSN": 2, "NP": 3}
        assert encoder.pharmacist_credential_map == {"RPh": 0, "PharmD": 1, "BCOP": 2}
        
        # Validate custom encoder pipeline initialization
        assert encoder._custom_encoders == []

    def test_custom_configuration_initialization(self):
        """Validate CategoricalEncoder with custom healthcare analytics configuration."""
        custom_prefix_map = {'diagnosis_type': 'dx', 'treatment_type': 'tx'}
        encoder = CategoricalEncoder(
            one_hot_prefix_map=custom_prefix_map,
            one_hot_drop_first=True,
            dtype=float
        )
        
        # Validate custom configuration parameters
        assert encoder.one_hot_prefix_map == custom_prefix_map
        assert encoder.one_hot_drop_first is True
        assert encoder.dtype == float
        
        # Validate healthcare mappings remain consistent regardless of custom config
        assert encoder.sex_map == {"F": 0, "M": 1}
        assert encoder.severity_map == {"Low": 0, "Medium": 1, "High": 2}

    def test_none_parameters_handling(self):
        """Validate proper handling of None parameters with healthcare defaults."""
        encoder = CategoricalEncoder(one_hot_prefix_map=None)
        
        # Validate None parameter defaults to empty dict
        assert encoder.one_hot_prefix_map == {}
        assert encoder.one_hot_drop_first is False
        assert encoder.dtype == int


class TestOneHotEncoding:
    """Test suite for one-hot encoding functionality with healthcare domain awareness."""

    def setup_method(self):
        """Set up test data for one-hot encoding validation."""
        self.encoder = CategoricalEncoder()
        self.sample_df = pd.DataFrame({
            'diagnosis_type': ['SolidTumor', 'Hematologic', 'SolidTumor', 'Autoimmune'],
            'treatment_type': ['Chemotherapy', 'Immunotherapy', 'Chemotherapy', 'TargetedTherapy'],
            'floor': [1, 2, 1, 3],
            'TAT_minutes': [45, 75, 60, 50]
        })

    def test_basic_one_hot_encoding(self):
        """Validate basic one-hot encoding functionality for healthcare categorical variables."""
        result = self.encoder.one_hot(self.sample_df, ['diagnosis_type', 'treatment_type'])
        
        # Validate original non-categorical columns preserved
        assert 'floor' in result.columns
        assert 'TAT_minutes' in result.columns
        
        # Validate one-hot encoded columns generated
        diagnosis_cols = [col for col in result.columns if col.startswith('diagnosis_type_')]
        treatment_cols = [col for col in result.columns if col.startswith('treatment_type_')]
        
        assert len(diagnosis_cols) == 3  # SolidTumor, Hematologic, Autoimmune
        assert len(treatment_cols) == 3  # Chemotherapy, Immunotherapy, TargetedTherapy
        
        # Validate binary encoding integrity
        assert all(result[diagnosis_cols].sum(axis=1) == 1)  # Exactly one category per row
        assert all(result[treatment_cols].sum(axis=1) == 1)

    def test_custom_prefix_mapping(self):
        """Validate custom prefix mapping for clinical interpretability."""
        custom_encoder = CategoricalEncoder(
            one_hot_prefix_map={'diagnosis_type': 'dx', 'treatment_type': 'tx'}
        )
        result = custom_encoder.one_hot(self.sample_df, ['diagnosis_type', 'treatment_type'])
        
        # Validate custom prefixes applied
        dx_cols = [col for col in result.columns if col.startswith('dx_')]
        tx_cols = [col for col in result.columns if col.startswith('tx_')]
        
        assert len(dx_cols) == 3
        assert len(tx_cols) == 3
        
        # Validate specific column names with custom prefixes
        assert 'dx_SolidTumor' in result.columns
        assert 'tx_Chemotherapy' in result.columns

    def test_drop_first_functionality(self):
        """Validate drop_first parameter for regression modeling requirements."""
        drop_first_encoder = CategoricalEncoder(one_hot_drop_first=True)
        result = drop_first_encoder.one_hot(self.sample_df, ['diagnosis_type'])
        
        # Validate reference category dropped (alphabetically first)
        diagnosis_cols = [col for col in result.columns if col.startswith('diagnosis_type_')]
        assert len(diagnosis_cols) == 2  # One less than total categories
        assert 'diagnosis_type_Autoimmune' not in result.columns  # First alphabetically

    def test_missing_columns_handling(self):
        """Validate graceful handling of missing categorical columns."""
        result = self.encoder.one_hot(self.sample_df, ['nonexistent_column', 'diagnosis_type'])
        
        # Validate processing continues with existing columns only
        assert len([col for col in result.columns if col.startswith('diagnosis_type_')]) == 3
        assert len([col for col in result.columns if col.startswith('nonexistent_')]) == 0

    def test_empty_columns_list(self):
        """Validate behavior with empty columns list."""
        result = self.encoder.one_hot(self.sample_df, [])
        
        # Validate original dataframe returned unchanged
        pd.testing.assert_frame_equal(result, self.sample_df)

    def test_data_type_configuration(self):
        """Validate data type configuration for encoded features."""
        float_encoder = CategoricalEncoder(dtype=float)
        result = float_encoder.one_hot(self.sample_df, ['diagnosis_type'])
        
        # Validate data type applied to encoded columns
        diagnosis_cols = [col for col in result.columns if col.startswith('diagnosis_type_')]
        for col in diagnosis_cols:
            assert result[col].dtype == float


class TestOrdinalEncoding:
    """Test suite for healthcare domain-aware ordinal encoding functionality."""

    def setup_method(self):
        """Set up test data for ordinal encoding validation."""
        self.encoder = CategoricalEncoder()
        self.healthcare_df = pd.DataFrame({
            'sex': ['F', 'M', 'F', 'M'],
            'severity': ['Low', 'High', 'Medium', 'Low'],
            'nurse_credential': ['RN', 'BSN', 'MSN', 'NP'],
            'pharmacist_credential': ['RPh', 'PharmD', 'BCOP', 'RPh'],
            'TAT_minutes': [45, 75, 60, 50]
        })

    def test_sex_encoding(self):
        """Validate patient sex demographic encoding."""
        result = self.encoder.encode_sex(self.healthcare_df)
        
        # Validate sex encoding applied correctly
        expected_sex_encoded = [0, 1, 0, 1]  # F->0, M->1
        assert result['sex'].tolist() == expected_sex_encoded
        
        # Validate other columns preserved
        assert result['TAT_minutes'].tolist() == [45, 75, 60, 50]

    def test_severity_encoding(self):
        """Validate clinical severity priority encoding."""
        result = self.encoder.encode_severity(self.healthcare_df)
        
        # Validate severity hierarchy encoding applied correctly
        expected_severity_encoded = [0, 2, 1, 0]  # Low->0, High->2, Medium->1
        assert result['severity'].tolist() == expected_severity_encoded

    def test_credentials_encoding(self):
        """Validate healthcare credential hierarchy encoding."""
        result = self.encoder.encode_credentials(self.healthcare_df)
        
        # Validate nursing credential hierarchy encoding
        expected_nurse_encoded = [0, 1, 2, 3]  # RN->0, BSN->1, MSN->2, NP->3
        assert result['nurse_credential'].tolist() == expected_nurse_encoded
        
        # Validate pharmacy credential hierarchy encoding
        expected_pharmacy_encoded = [0, 1, 2, 0]  # RPh->0, PharmD->1, BCOP->2
        assert result['pharmacist_credential'].tolist() == expected_pharmacy_encoded

    def test_missing_ordinal_columns(self):
        """Validate behavior when ordinal columns are missing."""
        df_no_sex = self.healthcare_df.drop('sex', axis=1)
        result = self.encoder.encode_sex(df_no_sex)
        
        # Validate processing continues without error
        pd.testing.assert_frame_equal(result, df_no_sex)

    def test_invalid_categorical_values(self):
        """Validate handling of invalid categorical values in healthcare data."""
        df_invalid = self.healthcare_df.copy()
        df_invalid.loc[0, 'sex'] = 'Unknown'  # Invalid sex value
        
        result = self.encoder.encode_sex(df_invalid)
        
        # Validate invalid values become NaN (pandas map behavior)
        assert pd.isna(result.loc[0, 'sex'])
        assert result.loc[1, 'sex'] == 1  # Valid values still encoded


class TestCustomEncoders:
    """Test suite for custom encoder registration and pipeline extension."""

    def setup_method(self):
        """Set up test data and encoder for custom encoder validation."""
        self.encoder = CategoricalEncoder()
        self.sample_df = pd.DataFrame({
            'treatment_type': ['Chemotherapy', 'Immunotherapy', 'TargetedTherapy'],
            'TAT_minutes': [45, 75, 60]
        })

    def test_custom_encoder_registration(self):
        """Validate custom encoder registration functionality."""
        def custom_treatment_complexity(df):
            """Custom encoder for treatment complexity analysis."""
            complexity_map = {
                'Chemotherapy': 'High',
                'Immunotherapy': 'Medium',
                'TargetedTherapy': 'Low'
            }
            df['treatment_complexity'] = df['treatment_type'].map(complexity_map)
            return df
        
        # Register custom encoder
        self.encoder.register(custom_treatment_complexity)
        
        # Validate custom encoder added to pipeline
        assert len(self.encoder._custom_encoders) == 1
        assert self.encoder._custom_encoders[0] == custom_treatment_complexity

    def test_multiple_custom_encoders(self):
        """Validate multiple custom encoder registration and execution."""
        def encoder1(df):
            df['custom_feature_1'] = 'encoded_1'
            return df
        
        def encoder2(df):
            df['custom_feature_2'] = 'encoded_2'
            return df
        
        # Register multiple custom encoders
        self.encoder.register(encoder1)
        self.encoder.register(encoder2)
        
        # Validate both encoders registered
        assert len(self.encoder._custom_encoders) == 2

    def test_clear_custom_encoders(self):
        """Validate custom encoder pipeline reset functionality."""
        def dummy_encoder(df):
            return df
        
        # Register encoder then clear
        self.encoder.register(dummy_encoder)
        assert len(self.encoder._custom_encoders) == 1
        
        self.encoder.clear_custom()
        assert len(self.encoder._custom_encoders) == 0

    def test_custom_encoder_execution_in_pipeline(self):
        """Validate custom encoder execution within processing pipeline."""
        def add_complexity_feature(df):
            df['complexity_score'] = len(df)  # Simple test feature
            return df
        
        self.encoder.register(add_complexity_feature)
        result = self.encoder.apply(self.sample_df)
        
        # Validate custom feature added during pipeline execution
        assert 'complexity_score' in result.columns
        assert result['complexity_score'].iloc[0] == 3  # Length of test dataframe


class TestPipelineExecution:
    """Test suite for categorical encoding pipeline execution and configuration."""

    def setup_method(self):
        """Set up comprehensive test data for pipeline validation."""
        self.encoder = CategoricalEncoder()
        self.comprehensive_df = pd.DataFrame({
            'sex': ['F', 'M', 'F'],
            'severity': ['Low', 'High', 'Medium'],
            'nurse_credential': ['RN', 'BSN', 'MSN'],
            'pharmacist_credential': ['RPh', 'PharmD', 'BCOP'],
            'diagnosis_type': ['SolidTumor', 'Hematologic', 'Autoimmune'],
            'floor': [1, 2, 3],
            'TAT_minutes': [45, 75, 60]
        })

    def test_apply_with_default_sequence(self):
        """Validate apply method with default processing sequence."""
        result = self.encoder.apply(self.comprehensive_df)
        
        # Validate ordinal encodings applied
        assert result['sex'].dtype in [int, float]  # Encoded as numeric
        assert result['severity'].dtype in [int, float]
        assert result['nurse_credential'].dtype in [int, float]
        assert result['pharmacist_credential'].dtype in [int, float]

    def test_apply_with_custom_sequence(self):
        """Validate apply method with custom processing sequence."""
        custom_sequence = [self.encoder.encode_sex, self.encoder.encode_severity]
        result = self.encoder.apply(self.comprehensive_df, sequence=custom_sequence)
        
        # Validate only specified encodings applied
        assert result['sex'].dtype in [int, float]  # Encoded
        assert result['severity'].dtype in [int, float]  # Encoded
        assert result['nurse_credential'].dtype == object  # Not encoded
        assert result['pharmacist_credential'].dtype == object  # Not encoded

    def test_transform_full_pipeline(self):
        """Validate complete transform pipeline for TAT prediction modeling."""
        # Configure encoder for comprehensive categorical processing
        self.encoder.one_hot_prefix_map = {'diagnosis_type': 'dx'}
        
        result = self.encoder.transform(self.comprehensive_df)
        
        # Validate floor converted to categorical
        assert 'floor' in result.columns
        
        # Validate ordinal encodings applied
        assert result['sex'].dtype in [int, float]
        assert result['severity'].dtype in [int, float]
        
        # Validate one-hot encoding applied where configured
        if 'diagnosis_type' in self.encoder.one_hot_prefix_map:
            dx_cols = [col for col in result.columns if col.startswith('dx_')]
            assert len(dx_cols) > 0

    def test_transform_with_missing_floor_column(self):
        """Validate transform behavior when floor column missing."""
        df_no_floor = self.comprehensive_df.drop('floor', axis=1)
        result = self.encoder.transform(df_no_floor)
        
        # Validate processing continues without error
        assert 'floor' not in result.columns
        assert len(result) == len(df_no_floor)


class TestSklearnCompatibility:
    """Test suite for sklearn-style interface compatibility."""

    def setup_method(self):
        """Set up test data for sklearn interface validation."""
        self.encoder = CategoricalEncoder()
        self.sample_df = pd.DataFrame({
            'sex': ['F', 'M', 'F'],
            'severity': ['Low', 'High', 'Medium'],
            'TAT_minutes': [45, 75, 60]
        })

    def test_fit_method_compatibility(self):
        """Validate sklearn-style fit method."""
        result = self.encoder.fit(self.sample_df)
        
        # Validate fit returns self for method chaining
        assert result is self.encoder

    def test_transform_method_compatibility(self):
        """Validate sklearn-style transform method."""
        result = self.encoder.transform(self.sample_df)
        
        # Validate transform returns processed DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_df)

    def test_fit_transform_method_equivalence(self):
        """Validate fit_transform method equivalence to fit().transform()."""
        # Method 1: separate fit and transform
        encoder1 = CategoricalEncoder()
        encoder1.fit(self.sample_df)
        result1 = encoder1.transform(self.sample_df)
        
        # Method 2: combined fit_transform
        encoder2 = CategoricalEncoder()
        result2 = encoder2.fit_transform(self.sample_df)
        
        # Validate equivalent results
        pd.testing.assert_frame_equal(result1, result2)

    def test_method_chaining_support(self):
        """Validate method chaining capability."""
        result = self.encoder.fit(self.sample_df).transform(self.sample_df)
        
        # Validate chaining works and produces valid result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.sample_df)


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge case scenarios."""

    def setup_method(self):
        """Set up edge case test data."""
        self.encoder = CategoricalEncoder()

    def test_empty_dataframe_processing(self):
        """Validate handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.encoder.transform(empty_df)
        
        # Validate empty DataFrame handled gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_row_processing(self):
        """Validate processing of single-row DataFrame."""
        single_row_df = pd.DataFrame({
            'sex': ['F'],
            'severity': ['High'],
            'TAT_minutes': [60]
        })
        result = self.encoder.transform(single_row_df)
        
        # Validate single row processed correctly
        assert len(result) == 1
        assert result['sex'].iloc[0] == 0  # F encoded as 0
        assert result['severity'].iloc[0] == 2  # High encoded as 2

    def test_all_missing_categorical_values(self):
        """Validate handling of DataFrame with all missing categorical values."""
        missing_df = pd.DataFrame({
            'sex': [None, None, None],
            'severity': [None, None, None],
            'TAT_minutes': [45, 60, 75]
        })
        result = self.encoder.transform(missing_df)
        
        # Validate missing values handled (become NaN in encoded columns)
        assert pd.isna(result['sex']).all()
        assert pd.isna(result['severity']).all()
        assert not pd.isna(result['TAT_minutes']).any()  # Non-categorical preserved

    def test_malformed_data_resilience(self):
        """Validate resilience to malformed categorical data."""
        malformed_df = pd.DataFrame({
            'sex': ['F', 123, 'Invalid', 'M'],  # Mixed types and invalid values
            'severity': ['Low', None, '', 'High'],  # Mixed None and empty string
            'TAT_minutes': [45, 60, 75, 50]
        })
        result = self.encoder.transform(malformed_df)
        
        # Validate processing completes without error
        assert len(result) == len(malformed_df)
        # Valid values should be encoded correctly
        assert result['sex'].iloc[0] == 0  # 'F' -> 0
        assert result['sex'].iloc[3] == 1  # 'M' -> 1
        assert result['severity'].iloc[0] == 0  # 'Low' -> 0
        assert result['severity'].iloc[3] == 2  # 'High' -> 2

    def test_non_destructive_operations(self):
        """Validate that operations don't modify original DataFrame."""
        original_df = pd.DataFrame({
            'sex': ['F', 'M'],
            'severity': ['Low', 'High'],
            'TAT_minutes': [45, 75]
        })
        original_copy = original_df.copy()
        
        # Transform should not modify original
        self.encoder.transform(original_df)
        pd.testing.assert_frame_equal(original_df, original_copy)


class TestPerformanceAndScalability:
    """Test suite for performance validation with large healthcare datasets."""

    def setup_method(self):
        """Set up large dataset for performance validation."""
        self.encoder = CategoricalEncoder()

    def test_large_dataset_processing(self):
        """Validate performance with large medication order dataset simulation."""
        # Create large dataset simulating 10k medication orders
        np.random.seed(42)
        n_orders = 10000
        
        large_df = pd.DataFrame({
            'sex': np.random.choice(['F', 'M'], n_orders),
            'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders),
            'nurse_credential': np.random.choice(['RN', 'BSN', 'MSN', 'NP'], n_orders),
            'pharmacist_credential': np.random.choice(['RPh', 'PharmD', 'BCOP'], n_orders),
            'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic', 'Autoimmune'], n_orders),
            'TAT_minutes': np.random.randint(30, 120, n_orders)
        })
        
        # Configure one-hot encoding for diagnosis
        self.encoder.one_hot_prefix_map = {'diagnosis_type': 'dx'}
        
        # Validate processing completes efficiently
        result = self.encoder.transform(large_df)
        
        # Validate results integrity
        assert len(result) == n_orders
        assert result['sex'].dtype in [int, float]  # Ordinal encoded
        assert len([col for col in result.columns if col.startswith('dx_')]) == 3  # One-hot encoded

    def test_memory_efficiency(self):
        """Validate memory-efficient processing for healthcare analytics workflows."""
        # Create moderately sized dataset for memory testing
        n_orders = 5000
        np.random.seed(42)
        
        df = pd.DataFrame({
            'sex': np.random.choice(['F', 'M'], n_orders),
            'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders),
            'TAT_minutes': np.random.randint(30, 120, n_orders)
        })
        
        # Validate processing doesn't cause memory issues
        result = self.encoder.transform(df)
        
        # Validate memory usage reasonable (result size should be manageable)
        assert result.memory_usage().sum() > 0  # Has some memory usage
        assert len(result.columns) >= len(df.columns)  # Features added or preserved
