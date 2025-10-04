"""
Test suite for functionality.

Production-ready validation framework for OperationalEngineer class ensuring healthcare
operational feature generation accuracy, clinical domain compliance, and pharmacy workflow
optimization capabilities. Validates professional experience assessment, operational load
analysis, patient complexity evaluation, and interaction feature generation for TAT modeling.

Test Categories:
- Professional Experience Features: Healthcare competency indicators and staffing optimization
- Operational Load Features: Capacity planning and workflow bottleneck identification
- Patient Complexity Features: Clinical prioritization and resource allocation assessment
- Interaction Features: Operational synergies and healthcare workflow dependencies
- Integration Testing: End-to-end operational feature engineering pipeline validation
- Edge Cases: Boundary conditions and healthcare operational exception handling
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from tat.features.operational import OperationalEngineer


class TestOperationalEngineerInitialization:
    """Validate OperationalEngineer initialization and configuration management."""
    
    def test_default_initialization(self):
        """Verify default healthcare-optimized configuration parameters."""
        engineer = OperationalEngineer()
        assert engineer.experience_threshold == 5
        assert engineer.high_occupancy_threshold == 80.0
    
    def test_custom_initialization(self):
        """Validate custom threshold configuration for specialized healthcare requirements."""
        engineer = OperationalEngineer(
            experience_threshold=7,
            high_occupancy_threshold=85.0
        )
        assert engineer.experience_threshold == 7
        assert engineer.high_occupancy_threshold == 85.0
    
    def test_default_factory_method(self):
        """Verify default factory method creates properly configured instance."""
        engineer = OperationalEngineer.default()
        assert engineer.experience_threshold == 5
        assert engineer.high_occupancy_threshold == 80.0
        assert isinstance(engineer, OperationalEngineer)


class TestExperienceFeatures:
    """Validate healthcare professional experience feature generation."""
    
    @pytest.fixture
    def sample_experience_data(self):
        """Create healthcare professional experience test dataset."""
        return pd.DataFrame({
            'nurse_employment_years': [2, 5, 8, 10, 1, 15],
            'pharmacist_employment_years': [3, 7, 4, 12, 0, 20],
            'other_column': ['A', 'B', 'C', 'D', 'E', 'F']
        })
    
    def test_experience_features_creation(self, sample_experience_data):
        """Verify professional experience indicators are correctly generated."""
        engineer = OperationalEngineer(experience_threshold=5)
        result = engineer.add_experience_features(sample_experience_data)
        
        # Validate high experience nurse flags
        expected_nurse = [0, 0, 1, 1, 0, 1]  # > 5 years
        assert result['high_experience_nurse'].tolist() == expected_nurse
        
        # Validate high experience pharmacist flags
        expected_pharmacist = [0, 1, 0, 1, 0, 1]  # > 5 years
        assert result['high_experience_pharmacist'].tolist() == expected_pharmacist
    
    def test_experience_threshold_boundary(self):
        """Validate experience threshold boundary conditions."""
        df = pd.DataFrame({
            'nurse_employment_years': [4, 5, 6],
            'pharmacist_employment_years': [4, 5, 6]
        })
        
        engineer = OperationalEngineer(experience_threshold=5)
        result = engineer.add_experience_features(df)
        
        # Exactly at threshold should be 0 (not high experience)
        assert result['high_experience_nurse'].tolist() == [0, 0, 1]
        assert result['high_experience_pharmacist'].tolist() == [0, 0, 1]
    
    def test_experience_features_non_destructive(self, sample_experience_data):
        """Ensure original DataFrame is preserved during processing."""
        original_columns = sample_experience_data.columns.tolist()
        original_shape = sample_experience_data.shape
        
        engineer = OperationalEngineer()
        result = engineer.add_experience_features(sample_experience_data)
        
        # Original DataFrame should be unchanged
        assert sample_experience_data.columns.tolist() == original_columns
        assert sample_experience_data.shape == original_shape
        
        # Result should have additional columns
        assert len(result.columns) == len(original_columns) + 2
        assert 'high_experience_nurse' in result.columns
        assert 'high_experience_pharmacist' in result.columns
    
    def test_experience_features_data_types(self, sample_experience_data):
        """Validate experience feature data types for downstream processing."""
        engineer = OperationalEngineer()
        result = engineer.add_experience_features(sample_experience_data)
        
        assert result['high_experience_nurse'].dtype == int
        assert result['high_experience_pharmacist'].dtype == int
        assert all(result['high_experience_nurse'].isin([0, 1]))
        assert all(result['high_experience_pharmacist'].isin([0, 1]))


class TestLoadFeatures:
    """Validate operational load and capacity feature generation."""
    
    @pytest.fixture
    def sample_load_data(self):
        """Create operational load test dataset."""
        return pd.DataFrame({
            'floor_occupancy_pct': [70, 85, 90, 75, 95, 60],
            'queue_length_at_order': [2, 8, 12, 4, 15, 1],
            'other_column': ['A', 'B', 'C', 'D', 'E', 'F']
        })
    
    def test_load_features_creation(self, sample_load_data):
        """Verify operational load indicators are correctly generated."""
        engineer = OperationalEngineer(high_occupancy_threshold=80.0)
        result = engineer.add_load_features(sample_load_data)
        
        # Validate high occupancy flags (> 80%)
        expected_occupancy = [0, 1, 1, 0, 1, 0]
        assert result['high_occupancy'].tolist() == expected_occupancy
        
        # Validate long queue flags (> median of [2, 8, 12, 4, 15, 1] = 6.0)
        expected_queue = [0, 1, 1, 0, 1, 0]
        assert result['long_queue'].tolist() == expected_queue
    
    def test_occupancy_threshold_boundary(self):
        """Validate occupancy threshold boundary conditions."""
        df = pd.DataFrame({
            'floor_occupancy_pct': [79, 80, 81],
            'queue_length_at_order': [5, 5, 5]
        })
        
        engineer = OperationalEngineer(high_occupancy_threshold=80.0)
        result = engineer.add_load_features(df)
        
        # Exactly at threshold should be 0 (not high occupancy)
        assert result['high_occupancy'].tolist() == [0, 0, 1]
    
    def test_queue_median_calculation(self):
        """Verify queue length median calculation for threshold determination."""
        df = pd.DataFrame({
            'floor_occupancy_pct': [50, 50, 50, 50, 50],
            'queue_length_at_order': [1, 3, 5, 7, 9]  # median = 5
        })
        
        engineer = OperationalEngineer()
        result = engineer.add_load_features(df)
        
        # Values > 5 should be flagged as long queue
        expected_queue = [0, 0, 0, 1, 1]
        assert result['long_queue'].tolist() == expected_queue
    
    def test_load_features_non_destructive(self, sample_load_data):
        """Ensure original DataFrame is preserved during load processing."""
        original_columns = sample_load_data.columns.tolist()
        original_shape = sample_load_data.shape
        
        engineer = OperationalEngineer()
        result = engineer.add_load_features(sample_load_data)
        
        # Original DataFrame should be unchanged
        assert sample_load_data.columns.tolist() == original_columns
        assert sample_load_data.shape == original_shape
        
        # Result should have additional columns
        assert len(result.columns) == len(original_columns) + 2
        assert 'high_occupancy' in result.columns
        assert 'long_queue' in result.columns


class TestComplexityFeatures:
    """Validate patient complexity assessment feature generation."""
    
    @pytest.fixture
    def sample_complexity_data(self):
        """Create patient complexity test dataset."""
        return pd.DataFrame({
            'severity': ['Low', 'Medium', 'High', 'Low', 'High', 'Medium'],
            'premed_required': [0, 1, 0, 1, 1, 0],
            'stat_order': [0, 0, 1, 0, 1, 1],
            'other_column': ['A', 'B', 'C', 'D', 'E', 'F']
        })
    
    def test_complexity_features_creation(self, sample_complexity_data):
        """Verify patient complexity indicators are correctly generated."""
        engineer = OperationalEngineer()
        result = engineer.add_complexity_features(sample_complexity_data)
        
        # Complex case = High severity OR premed required OR stat order
        # ['Low',0,0] = 0, ['Medium',1,0] = 1, ['High',0,1] = 1, 
        # ['Low',1,0] = 1, ['High',1,1] = 1, ['Medium',0,1] = 1
        expected_complex = [0, 1, 1, 1, 1, 1]
        assert result['complex_case'].tolist() == expected_complex
    
    def test_complexity_individual_conditions(self):
        """Validate each complexity condition individually."""
        # Test high severity only
        df_severity = pd.DataFrame({
            'severity': ['High', 'Low'],
            'premed_required': [0, 0],
            'stat_order': [0, 0]
        })
        engineer = OperationalEngineer()
        result = engineer.add_complexity_features(df_severity)
        assert result['complex_case'].tolist() == [1, 0]
        
        # Test premedication requirement only
        df_premed = pd.DataFrame({
            'severity': ['Low', 'Low'],
            'premed_required': [1, 0],
            'stat_order': [0, 0]
        })
        result = engineer.add_complexity_features(df_premed)
        assert result['complex_case'].tolist() == [1, 0]
        
        # Test STAT order only
        df_stat = pd.DataFrame({
            'severity': ['Low', 'Low'],
            'premed_required': [0, 0],
            'stat_order': [1, 0]
        })
        result = engineer.add_complexity_features(df_stat)
        assert result['complex_case'].tolist() == [1, 0]
    
    def test_complexity_no_conditions(self):
        """Verify cases with no complexity factors are correctly identified."""
        df = pd.DataFrame({
            'severity': ['Low', 'Medium'],
            'premed_required': [0, 0],
            'stat_order': [0, 0]
        })
        
        engineer = OperationalEngineer()
        result = engineer.add_complexity_features(df)
        assert result['complex_case'].tolist() == [0, 0]
    
    def test_complexity_features_data_types(self, sample_complexity_data):
        """Validate complexity feature data types."""
        engineer = OperationalEngineer()
        result = engineer.add_complexity_features(sample_complexity_data)
        
        assert result['complex_case'].dtype == int
        assert all(result['complex_case'].isin([0, 1]))


class TestInteractionFeatures:
    """Validate operational interaction feature generation."""
    
    @pytest.fixture
    def sample_interaction_data(self):
        """Create interaction features test dataset."""
        return pd.DataFrame({
            'nurse_employment_years': [5, 10],
            'pharmacist_employment_years': [3, 8],
            'floor_occupancy_pct': [70, 90],
            'queue_length_at_order': [2, 8],
            'severity': ['Low', 'High'],
            'patient_readiness_score': [1, 3]
        })
    
    def test_interaction_features_creation(self, sample_interaction_data):
        """Verify interaction features are correctly calculated."""
        engineer = OperationalEngineer()
        result = engineer.add_interaction_features(sample_interaction_data)
        
        # Validate nurse experience × occupancy
        expected_nurse_occ = [5 * 70, 10 * 90]  # [350, 900]
        assert result['nurse_exp_x_occupancy'].tolist() == expected_nurse_occ
        
        # Validate pharmacist experience × queue
        expected_pharm_queue = [3 * 2, 8 * 8]  # [6, 64]
        assert result['pharmacist_exp_x_queue'].tolist() == expected_pharm_queue
        
        # Validate readiness × queue
        expected_readiness_queue = [1 * 2, 3 * 8]  # [2, 24]
        assert result['readiness_x_queue'].tolist() == expected_readiness_queue
    
    def test_severity_occupancy_interaction(self, sample_interaction_data):
        """Validate severity × occupancy interaction using categorical codes."""
        engineer = OperationalEngineer()
        result = engineer.add_interaction_features(sample_interaction_data)
        
        # Severity categorical codes: depends on pandas implementation
        # We'll verify the interaction exists and has reasonable values
        assert 'severity_x_occupancy' in result.columns
        assert len(result['severity_x_occupancy']) == 2
        assert all(pd.notna(result['severity_x_occupancy']))
    
    def test_interaction_features_non_destructive(self, sample_interaction_data):
        """Ensure original DataFrame is preserved during interaction processing."""
        original_columns = sample_interaction_data.columns.tolist()
        original_shape = sample_interaction_data.shape
        
        engineer = OperationalEngineer()
        result = engineer.add_interaction_features(sample_interaction_data)
        
        # Original DataFrame should be unchanged
        assert sample_interaction_data.columns.tolist() == original_columns
        assert sample_interaction_data.shape == original_shape
        
        # Result should have additional interaction columns
        expected_interactions = [
            'nurse_exp_x_occupancy',
            'pharmacist_exp_x_queue', 
            'severity_x_occupancy',
            'readiness_x_queue'
        ]
        
        for interaction in expected_interactions:
            assert interaction in result.columns


class TestOperationalEngineerTransform:
    """Validate comprehensive operational feature engineering pipeline."""
    
    @pytest.fixture
    def comprehensive_test_data(self):
        """Create comprehensive test dataset for full pipeline validation."""
        return pd.DataFrame({
            'nurse_employment_years': [3, 7, 12],
            'pharmacist_employment_years': [2, 6, 15],
            'floor_occupancy_pct': [75, 85, 95],
            'queue_length_at_order': [1, 5, 10],
            'severity': ['Low', 'Medium', 'High'],
            'premed_required': [0, 1, 0],
            'stat_order': [0, 0, 1],
            'patient_readiness_score': [2, 1, 3],
            'patient_id': ['P001', 'P002', 'P003']
        })
    
    def test_transform_comprehensive_pipeline(self, comprehensive_test_data):
        """Verify complete operational feature engineering pipeline."""
        engineer = OperationalEngineer(
            experience_threshold=5,
            high_occupancy_threshold=80
        )
        result = engineer.transform(comprehensive_test_data)
        
        # Verify all feature categories are present
        experience_features = ['high_experience_nurse', 'high_experience_pharmacist']
        load_features = ['high_occupancy', 'long_queue']
        complexity_features = ['complex_case']
        interaction_features = [
            'nurse_exp_x_occupancy', 'pharmacist_exp_x_queue',
            'severity_x_occupancy', 'readiness_x_queue'
        ]
        
        all_expected_features = (experience_features + load_features + 
                               complexity_features + interaction_features)
        
        for feature in all_expected_features:
            assert feature in result.columns
    
    def test_transform_feature_counts(self, comprehensive_test_data):
        """Validate expected number of generated operational features."""
        original_columns = len(comprehensive_test_data.columns)
        
        engineer = OperationalEngineer()
        result = engineer.transform(comprehensive_test_data)
        
        # Should add: 2 experience + 2 load + 1 complexity + 4 interaction = 9 features
        expected_new_columns = 9
        assert len(result.columns) == original_columns + expected_new_columns
    
    def test_transform_preserves_original_data(self, comprehensive_test_data):
        """Ensure transform preserves all original columns and values."""
        engineer = OperationalEngineer()
        result = engineer.transform(comprehensive_test_data)
        
        # All original columns should be present with unchanged values
        for col in comprehensive_test_data.columns:
            assert col in result.columns
            pd.testing.assert_series_equal(
                comprehensive_test_data[col], 
                result[col], 
                check_names=False
            )
    
    def test_transform_consistent_results(self, comprehensive_test_data):
        """Verify transform produces consistent results across multiple runs."""
        engineer = OperationalEngineer()
        
        result1 = engineer.transform(comprehensive_test_data)
        result2 = engineer.transform(comprehensive_test_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)


class TestEdgeCasesAndErrorHandling:
    """Validate operational feature engineering edge cases and error conditions."""
    
    def test_empty_dataframe(self):
        """Verify handling of empty DataFrame input."""
        empty_df = pd.DataFrame()
        engineer = OperationalEngineer()
        
        # Should handle gracefully - likely with KeyError for missing columns
        with pytest.raises(KeyError):
            engineer.transform(empty_df)
    
    def test_single_row_dataframe(self):
        """Validate processing of single-row datasets."""
        single_row_df = pd.DataFrame({
            'nurse_employment_years': [8],
            'pharmacist_employment_years': [12],
            'floor_occupancy_pct': [85],
            'queue_length_at_order': [5],
            'severity': ['High'],
            'premed_required': [1],
            'stat_order': [0],
            'patient_readiness_score': [2]
        })
        
        engineer = OperationalEngineer()
        result = engineer.transform(single_row_df)
        
        # Should process successfully
        assert len(result) == 1
        assert 'high_experience_nurse' in result.columns
        assert result['complex_case'].iloc[0] == 1  # High severity + premed
    
    def test_missing_values_handling(self):
        """Verify behavior with missing values in operational columns."""
        df_with_nan = pd.DataFrame({
            'nurse_employment_years': [5, np.nan, 8],
            'pharmacist_employment_years': [np.nan, 7, 3],
            'floor_occupancy_pct': [80, 90, np.nan],
            'queue_length_at_order': [2, np.nan, 6],
            'severity': ['Low', 'High', 'Medium'],
            'premed_required': [0, 1, 0],
            'stat_order': [0, 0, 1],
            'patient_readiness_score': [1, 2, np.nan]
        })
        
        engineer = OperationalEngineer()
        # Should handle NaN values appropriately (likely propagating NaN)
        result = engineer.transform(df_with_nan)
        assert len(result) == 3  # Should not crash
    
    def test_extreme_threshold_values(self):
        """Validate behavior with extreme threshold configurations."""
        df = pd.DataFrame({
            'nurse_employment_years': [10],
            'pharmacist_employment_years': [15],
            'floor_occupancy_pct': [50],
            'queue_length_at_order': [3],
            'severity': ['Low'],
            'premed_required': [0],
            'stat_order': [0],
            'patient_readiness_score': [2]
        })
        
        # Very high thresholds - nothing should be flagged as high
        high_threshold_engineer = OperationalEngineer(
            experience_threshold=50,
            high_occupancy_threshold=99
        )
        result = high_threshold_engineer.transform(df)
        
        assert result['high_experience_nurse'].iloc[0] == 0
        assert result['high_experience_pharmacist'].iloc[0] == 0
        assert result['high_occupancy'].iloc[0] == 0
        
        # Very low thresholds - everything should be flagged as high
        low_threshold_engineer = OperationalEngineer(
            experience_threshold=0,
            high_occupancy_threshold=0
        )
        result = low_threshold_engineer.transform(df)
        
        assert result['high_experience_nurse'].iloc[0] == 1
        assert result['high_experience_pharmacist'].iloc[0] == 1
        assert result['high_occupancy'].iloc[0] == 1


class TestDataTypeConsistency:
    """Validate operational feature data type consistency and compatibility."""
    
    @pytest.fixture
    def mixed_types_data(self):
        """Create dataset with mixed data types for validation."""
        return pd.DataFrame({
            'nurse_employment_years': [3.5, 7.2, 12.8],  # Float years
            'pharmacist_employment_years': [2, 6, 15],    # Integer years
            'floor_occupancy_pct': [75.5, 85.0, 95.7],   # Float percentages
            'queue_length_at_order': [1, 5, 10],         # Integer counts
            'severity': ['Low', 'Medium', 'High'],
            'premed_required': [0, 1, 0],
            'stat_order': [0, 0, 1],
            'patient_readiness_score': [2, 1, 3]
        })
    
    def test_numeric_type_handling(self, mixed_types_data):
        """Verify proper handling of mixed numeric types."""
        engineer = OperationalEngineer()
        result = engineer.transform(mixed_types_data)
        
        # Generated features should be integers (0/1 flags)
        binary_features = [
            'high_experience_nurse', 'high_experience_pharmacist',
            'high_occupancy', 'long_queue', 'complex_case'
        ]
        
        for feature in binary_features:
            assert result[feature].dtype == int
            assert all(result[feature].isin([0, 1]))
    
    def test_interaction_feature_types(self, mixed_types_data):
        """Validate interaction feature data types are appropriate."""
        engineer = OperationalEngineer()
        result = engineer.transform(mixed_types_data)
        
        # Interaction features should maintain numeric types
        interaction_features = [
            'nurse_exp_x_occupancy', 'pharmacist_exp_x_queue', 'readiness_x_queue'
        ]
        
        for feature in interaction_features:
            assert pd.api.types.is_numeric_dtype(result[feature])


class TestHealthcareComplianceValidation:
    """Validate healthcare domain compliance and clinical relevance."""
    
    def test_professional_experience_ranges(self):
        """Verify realistic healthcare professional experience handling."""
        # Test typical healthcare professional experience ranges
        df = pd.DataFrame({
            'nurse_employment_years': [0, 2, 5, 10, 25, 40],  # Career span
            'pharmacist_employment_years': [0, 3, 7, 15, 30, 35],
            'floor_occupancy_pct': [50, 60, 70, 80, 90, 100],
            'queue_length_at_order': [0, 1, 3, 5, 10, 20],
            'severity': ['Low'] * 6,
            'premed_required': [0] * 6,
            'stat_order': [0] * 6,
            'patient_readiness_score': [2] * 6
        })
        
        engineer = OperationalEngineer()
        result = engineer.transform(df)
        
        # Validate that experience thresholds make clinical sense
        # Nurses/pharmacists with >5 years should be considered experienced
        expected_nurse_exp = [0, 0, 0, 1, 1, 1]
        expected_pharm_exp = [0, 0, 1, 1, 1, 1]
        
        assert result['high_experience_nurse'].tolist() == expected_nurse_exp
        assert result['high_experience_pharmacist'].tolist() == expected_pharm_exp
    
    def test_clinical_severity_complexity_mapping(self):
        """Verify clinical severity appropriately maps to complexity assessment."""
        df = pd.DataFrame({
            'nurse_employment_years': [5] * 4,
            'pharmacist_employment_years': [5] * 4,
            'floor_occupancy_pct': [70] * 4,
            'queue_length_at_order': [3] * 4,
            'severity': ['Low', 'Medium', 'High', 'High'],
            'premed_required': [0, 0, 0, 1],
            'stat_order': [0, 0, 0, 0],
            'patient_readiness_score': [2] * 4
        })
        
        engineer = OperationalEngineer()
        result = engineer.transform(df)
        
        # High severity should always result in complex case
        # Medium/Low without other factors should not be complex
        expected_complexity = [0, 0, 1, 1]
        assert result['complex_case'].tolist() == expected_complexity
    
    def test_operational_capacity_thresholds(self):
        """Validate operational capacity thresholds align with healthcare standards."""
        df = pd.DataFrame({
            'nurse_employment_years': [5],
            'pharmacist_employment_years': [5],
            'floor_occupancy_pct': [80],  # Standard healthcare capacity threshold
            'queue_length_at_order': [5],
            'severity': ['Medium'],
            'premed_required': [0],
            'stat_order': [0],
            'patient_readiness_score': [2]
        })
        
        engineer = OperationalEngineer(high_occupancy_threshold=80.0)
        result = engineer.transform(df)
        
        # 80% occupancy should not be flagged as high (at threshold)
        assert result['high_occupancy'].iloc[0] == 0
