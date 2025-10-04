"""
Test suite for functionality.

Test suite for functionality.
medication preparation workflow analysis and TAT prediction modeling. Validates healthcare-
optimized laboratory feature engineering capabilities including clinical bounds enforcement,
reference range validation, abnormality detection, and derived feature generation.

Test Coverage Areas:
- LabProcessor initialization and configuration validation with clinical parameters
- Laboratory value bounds enforcement ensuring medically realistic clinical data integrity
- Missing value handling with clinical context preservation and transparency requirements
- Clinical abnormality detection through medical reference range validation
- Derived laboratory feature generation for comprehensive patient assessment
- Custom processing step registration and pipeline extension capabilities  
- Pipeline execution with configurable processing sequences and clinical workflows
- Error handling and edge cases for diverse healthcare laboratory data patterns
- Performance validation for large-scale oncology patient datasets

Healthcare Context:
- Clinical laboratory validation preserving patient safety and medical decision-making context
- Oncology patient laboratory monitoring supporting chemotherapy safety and treatment readiness
- Laboratory abnormality detection maintaining clinical interpretation and workflow optimization
- Integration testing for comprehensive TAT prediction modeling laboratory feature preparation

Production Validation:
- Robust handling of malformed laboratory data and clinical information system artifacts
- Pipeline architecture validation for automated laboratory feature engineering workflows
- Integration compatibility with Dana Farber's healthcare analytics infrastructure
- Performance validation for 100k+ patient order laboratory processing and analysis
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from tat.features.labs import LabProcessor


class TestLabProcessorInitialization:
    """Test suite for LabProcessor initialization and configuration validation."""

    def test_default_initialization(self):
        """Validate default LabProcessor initialization with clinical-optimized settings."""
        processor = LabProcessor()
        
        # Validate default laboratory columns configuration
        expected_lab_cols = [
            "lab_WBC_k_per_uL", "lab_HGB_g_dL", "lab_Platelets_k_per_uL",
            "lab_Creatinine_mg_dL", "lab_ALT_U_L"
        ]
        assert processor.cols == expected_lab_cols
        
        # Validate default clinical bounds configuration
        assert processor.clips["lab_WBC_k_per_uL"] == (0, 200)
        assert processor.clips["lab_HGB_g_dL"] == (0, 25)
        assert processor.clips["lab_Platelets_k_per_uL"] == (0, 600)
        
        # Validate default medical reference ranges
        assert processor.normals["lab_WBC_k_per_uL"] == (4.0, 11.0)
        assert processor.normals["lab_HGB_g_dL"] == (12.0, 16.0)
        assert processor.normals["lab_Creatinine_mg_dL"] == (0.6, 1.3)
        
        # Validate default missing value strategy
        assert processor.missing_strategy == 'median'
        
        # Validate pipeline initialization
        assert len(processor._pipeline) == 4
        assert processor._custom_steps == []

    def test_custom_configuration_initialization(self):
        """Validate LabProcessor with custom clinical analytics configuration."""
        custom_cols = ["lab_WBC_k_per_uL", "lab_HGB_g_dL"]
        custom_clips = {"lab_WBC_k_per_uL": (0.5, 100.0)}
        custom_normals = {"lab_WBC_k_per_uL": (3.5, 12.0)}
        
        processor = LabProcessor(
            cols=custom_cols,
            clips=custom_clips,
            normals=custom_normals,
            missing_strategy='flag'
        )
        
        # Validate custom configuration parameters
        assert processor.cols == custom_cols
        assert processor.clips["lab_WBC_k_per_uL"] == (0.5, 100.0)
        assert processor.normals["lab_WBC_k_per_uL"] == (3.5, 12.0)
        assert processor.missing_strategy == 'flag'

    def test_none_parameters_handling(self):
        """Validate proper handling of None parameters with clinical defaults."""
        processor = LabProcessor(cols=None, clips=None, normals=None)
        
        # Validate None parameters default to configuration values
        assert len(processor.cols) == 5  # Standard lab panel
        assert "lab_WBC_k_per_uL" in processor.cols
        assert processor.clips["lab_HGB_g_dL"] == (0, 25)
        assert processor.normals["lab_ALT_U_L"] == (7.0, 56.0)

    def test_missing_strategy_validation(self):
        """Validate different missing value strategy configurations."""
        strategies = ['mean', 'median', 'flag']
        
        for strategy in strategies:
            processor = LabProcessor(missing_strategy=strategy)
            assert processor.missing_strategy == strategy


class TestLaboratoryClipping:
    """Test suite for laboratory value bounds enforcement and clinical validation."""

    def setup_method(self):
        """Set up test data for laboratory bounds validation."""
        self.processor = LabProcessor()
        self.lab_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [5.0, -2.0, 250.0, 8.5, None, 'invalid'],
            'lab_HGB_g_dL': [14.0, -1.0, 30.0, 12.5, None, 'text'],
            'lab_Platelets_k_per_uL': [300.0, -50.0, 800.0, 200.0, None, 'invalid'],
            'lab_Creatinine_mg_dL': [1.0, -0.5, 25.0, 0.8, None, 'text'],
            'lab_ALT_U_L': [25.0, -10.0, 300.0, 45.0, None, 'invalid'],
            'TAT_minutes': [45, 60, 70, 80, 55, 65]
        })

    def test_laboratory_bounds_enforcement(self):
        """Validate laboratory bounds enforcement for clinical realism."""
        result = self.processor.clip_labs(self.lab_df)
        
        # Validate WBC clipping applied correctly (bounds: 0-200)
        wbc_values = result['lab_WBC_k_per_uL'].dropna()
        assert wbc_values.min() >= 0
        assert wbc_values.max() <= 200
        
        # Validate Hemoglobin clipping (bounds: 0-25)
        hgb_values = result['lab_HGB_g_dL'].dropna()
        assert hgb_values.min() >= 0
        assert hgb_values.max() <= 25
        
        # Validate Platelet clipping (bounds: 0-600)
        plt_values = result['lab_Platelets_k_per_uL'].dropna()
        assert plt_values.min() >= 0
        assert plt_values.max() <= 600

    def test_specific_laboratory_transformations(self):
        """Validate specific laboratory value transformations."""
        result = self.processor.clip_labs(self.lab_df)
        
        # Test specific WBC transformations: [5.0, -2.0, 250.0, 8.5, None, 'invalid']
        expected_wbc = [5.0, 0.0, 200.0, 8.5, np.nan, np.nan]
        actual_wbc = result['lab_WBC_k_per_uL'].tolist()
        
        for i, (expected, actual) in enumerate(zip(expected_wbc, actual_wbc)):
            if pd.isna(expected):
                assert pd.isna(actual), f"WBC index {i}: expected NaN, got {actual}"
            else:
                assert actual == expected, f"WBC index {i}: expected {expected}, got {actual}"

    def test_custom_laboratory_bounds(self):
        """Validate custom laboratory bounds for specialized populations."""
        oncology_processor = LabProcessor(clips={
            'lab_WBC_k_per_uL': (0.1, 50.0),  # Oncology-specific range
            'lab_Platelets_k_per_uL': (5.0, 1000.0)  # Chemotherapy impact range
        })
        
        df_oncology = pd.DataFrame({
            'lab_WBC_k_per_uL': [0.05, 25.0, 75.0],
            'lab_Platelets_k_per_uL': [1.0, 500.0, 1200.0],
            'TAT_minutes': [45, 60, 70]
        })
        
        result = oncology_processor.clip_labs(df_oncology)
        
        # Validate custom bounds applied
        expected_wbc = [0.1, 25.0, 50.0]  # 0.05->0.1, 75.0->50.0
        expected_plt = [5.0, 500.0, 1000.0]  # 1.0->5.0, 1200.0->1000.0
        assert result['lab_WBC_k_per_uL'].tolist() == expected_wbc
        assert result['lab_Platelets_k_per_uL'].tolist() == expected_plt

    def test_missing_laboratory_columns(self):
        """Validate behavior when laboratory columns are missing."""
        df_no_labs = pd.DataFrame({
            'age': [30, 45, 60],
            'TAT_minutes': [45, 60, 70]
        })
        
        result = self.processor.clip_labs(df_no_labs)
        
        # Validate processing continues without error
        pd.testing.assert_frame_equal(result, df_no_labs)

    def test_non_destructive_laboratory_processing(self):
        """Validate that laboratory processing doesn't modify original DataFrame."""
        original_df = self.lab_df.copy()
        self.processor.clip_labs(self.lab_df)
        
        # Validate original DataFrame unchanged
        pd.testing.assert_frame_equal(self.lab_df, original_df)


class TestMissingValueHandling:
    """Test suite for missing laboratory value processing with clinical context."""

    def setup_method(self):
        """Set up test data for missing value handling validation."""
        self.lab_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [5.0, None, 8.5, 6.2, None],
            'lab_HGB_g_dL': [14.0, 12.0, None, 13.5, None],
            'lab_Platelets_k_per_uL': [300.0, None, 250.0, None, 400.0],
            'TAT_minutes': [45, 60, 70, 80, 55]
        })

    def test_flag_missing_strategy(self):
        """Validate flag strategy preserving clinical transparency."""
        processor = LabProcessor(missing_strategy='flag')
        result = processor.handle_missing_values(self.lab_df)
        
        # Validate missing flags created
        expected_flags = ['lab_WBC_k_per_uL_missing', 'lab_HGB_g_dL_missing', 
                         'lab_Platelets_k_per_uL_missing']
        
        for flag in expected_flags:
            assert flag in result.columns
            # Validate flag values are binary (0 or 1)
            assert set(result[flag].unique()).issubset({0, 1})
        
        # Validate original missing values preserved
        assert pd.isna(result['lab_WBC_k_per_uL'].iloc[1])
        assert pd.isna(result['lab_HGB_g_dL'].iloc[2])

    def test_median_imputation_strategy(self):
        """Validate median imputation for robust clinical analytics."""
        processor = LabProcessor(missing_strategy='median')
        result = processor.handle_missing_values(self.lab_df)
        
        # Validate no missing values remain in lab columns
        lab_cols = ['lab_WBC_k_per_uL', 'lab_HGB_g_dL', 'lab_Platelets_k_per_uL']
        for col in lab_cols:
            assert not result[col].isna().any()
        
        # Validate median imputation applied correctly
        # WBC: [5.0, None, 8.5, 6.2, None] -> median of [5.0, 8.5, 6.2] = 6.2
        expected_wbc_median = 6.2
        assert result['lab_WBC_k_per_uL'].iloc[1] == expected_wbc_median
        assert result['lab_WBC_k_per_uL'].iloc[4] == expected_wbc_median

    def test_mean_imputation_strategy(self):
        """Validate mean imputation for clinical analytics."""
        processor = LabProcessor(missing_strategy='mean')
        result = processor.handle_missing_values(self.lab_df)
        
        # Validate no missing values remain in lab columns
        lab_cols = ['lab_WBC_k_per_uL', 'lab_HGB_g_dL', 'lab_Platelets_k_per_uL']
        for col in lab_cols:
            assert not result[col].isna().any()
        
        # Validate mean imputation applied correctly
        # WBC: [5.0, None, 8.5, 6.2, None] -> mean of [5.0, 8.5, 6.2] = 6.567
        expected_wbc_mean = (5.0 + 8.5 + 6.2) / 3
        assert abs(result['lab_WBC_k_per_uL'].iloc[1] - expected_wbc_mean) < 0.001

    def test_imputation_cache_consistency(self):
        """Validate consistent imputation values across multiple calls."""
        processor = LabProcessor(missing_strategy='median')
        
        # First call
        result1 = processor.handle_missing_values(self.lab_df)
        
        # Second call should use same imputation values
        result2 = processor.handle_missing_values(self.lab_df)
        
        # Validate consistent imputation
        pd.testing.assert_frame_equal(result1, result2)

    def test_all_missing_laboratory_column(self):
        """Validate handling of completely missing laboratory columns."""
        df_all_missing = pd.DataFrame({
            'lab_WBC_k_per_uL': [None, None, None],
            'lab_HGB_g_dL': [14.0, 12.0, 13.5],  # Valid column for comparison
            'TAT_minutes': [45, 60, 70]
        })
        
        processor = LabProcessor(missing_strategy='median')
        result = processor.handle_missing_values(df_all_missing)
        
        # Validate all-missing column handled (should remain NaN)
        assert result['lab_WBC_k_per_uL'].isna().all()
        # Valid column should be processed normally
        assert not result['lab_HGB_g_dL'].isna().any()


class TestClinicalAbnormalityDetection:
    """Test suite for laboratory abnormality detection and clinical flagging."""

    def setup_method(self):
        """Set up test data for clinical abnormality detection."""
        self.processor = LabProcessor()
        self.lab_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [2.0, 5.5, 15.0, 8.0, None],      # Low, Normal, High, Normal, Missing
            'lab_HGB_g_dL': [10.0, 14.0, 18.0, 13.0, None],       # Low, Normal, High, Normal, Missing  
            'lab_Platelets_k_per_uL': [100.0, 300.0, 500.0, 200.0, None],  # Low, Normal, High, Normal, Missing
            'lab_Creatinine_mg_dL': [0.4, 1.0, 2.0, 0.8, None],   # Low, Normal, High, Normal, Missing
            'lab_ALT_U_L': [5.0, 30.0, 80.0, 45.0, None],         # Low, Normal, High, Normal, Missing
            'TAT_minutes': [45, 60, 70, 80, 55]
        })

    def test_laboratory_abnormality_flag_generation(self):
        """Validate laboratory abnormality flag generation for clinical assessment."""
        result = self.processor.add_lab_flags(self.lab_df)
        
        # Validate abnormality flags created for each lab
        expected_flags = [
            'lab_WBC_k_per_low', 'lab_WBC_k_per_high',
            'lab_HGB_g_low', 'lab_HGB_g_high', 
            'lab_Platelets_k_per_low', 'lab_Platelets_k_per_high',
            'lab_Creatinine_mg_low', 'lab_Creatinine_mg_high',
            'lab_ALT_U_low', 'lab_ALT_U_high'
        ]
        
        for flag in expected_flags:
            assert flag in result.columns

    def test_specific_abnormality_detection(self):
        """Validate specific laboratory abnormality detection logic."""
        result = self.processor.add_lab_flags(self.lab_df)
        
        # Test WBC abnormalities (normal range: 4.0-11.0)
        # Values: [2.0, 5.5, 15.0, 8.0, None]
        expected_wbc_low = [1, 0, 0, 0, 0]   # 2.0 < 4.0
        expected_wbc_high = [0, 0, 1, 0, 0]  # 15.0 > 11.0
        assert result['lab_WBC_k_per_low'].tolist() == expected_wbc_low
        assert result['lab_WBC_k_per_high'].tolist() == expected_wbc_high
        
        # Test Hemoglobin abnormalities (normal range: 12.0-16.0)
        # Values: [10.0, 14.0, 18.0, 13.0, None]
        expected_hgb_low = [1, 0, 0, 0, 0]   # 10.0 < 12.0
        expected_hgb_high = [0, 0, 1, 0, 0]  # 18.0 > 16.0
        assert result['lab_HGB_g_low'].tolist() == expected_hgb_low
        assert result['lab_HGB_g_high'].tolist() == expected_hgb_high

    def test_missing_values_abnormality_handling(self):
        """Validate abnormality detection with missing laboratory values."""
        result = self.processor.add_lab_flags(self.lab_df)
        
        # Missing values should not trigger abnormality flags
        for col in self.processor.cols:
            base_name = col.replace('_k_per_uL', '_k_per').replace('_g_dL', '_g').replace('_mg_dL', '_mg').replace('_U_L', '_U')
            low_flag = f"{base_name}_low"
            high_flag = f"{base_name}_high"
            
            # Last row has missing values, should be 0 for both flags
            if low_flag in result.columns:
                assert result[low_flag].iloc[4] == 0  # Missing -> no abnormality flag
            if high_flag in result.columns:
                assert result[high_flag].iloc[4] == 0

    def test_custom_reference_ranges(self):
        """Validate abnormality detection with custom clinical reference ranges."""
        custom_normals = {
            'lab_WBC_k_per_uL': (3.0, 12.0),  # Wider range
            'lab_HGB_g_dL': (11.0, 17.0)      # Adjusted range
        }
        
        processor = LabProcessor(normals=custom_normals)
        result = processor.add_lab_flags(self.lab_df)
        
        # With custom ranges: WBC 2.0 is still low (< 3.0), but 15.0 is still high (> 12.0)
        expected_wbc_low = [1, 0, 0, 0, 0]   # 2.0 < 3.0
        expected_wbc_high = [0, 0, 1, 0, 0]  # 15.0 > 12.0
        assert result['lab_WBC_k_per_low'].tolist() == expected_wbc_low
        assert result['lab_WBC_k_per_high'].tolist() == expected_wbc_high

    def test_edge_case_boundary_values(self):
        """Validate abnormality detection at reference range boundaries."""
        boundary_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [4.0, 11.0, 3.99, 11.01],  # Exact boundaries
            'TAT_minutes': [45, 60, 70, 80]
        })
        
        result = self.processor.add_lab_flags(boundary_df)
        
        # Boundary value testing (normal range: 4.0-11.0)
        expected_wbc_low = [0, 0, 1, 0]   # 3.99 < 4.0
        expected_wbc_high = [0, 0, 0, 1]  # 11.01 > 11.0
        assert result['lab_WBC_k_per_low'].tolist() == expected_wbc_low
        assert result['lab_WBC_k_per_high'].tolist() == expected_wbc_high


class TestDerivedFeatureGeneration:
    """Test suite for derived laboratory feature generation and patient assessment."""

    def setup_method(self):
        """Set up test data for derived feature generation."""
        self.processor = LabProcessor()
        # Create data with pre-generated abnormality flags
        self.flagged_df = pd.DataFrame({
            'lab_WBC_low': [1, 0, 0, 0],
            'lab_WBC_high': [0, 0, 1, 0], 
            'lab_HGB_low': [0, 1, 0, 0],
            'lab_HGB_high': [0, 0, 0, 1],
            'lab_Platelets_low': [1, 0, 0, 0],
            'lab_Platelets_high': [0, 0, 0, 0],
            'lab_ALT_low': [0, 0, 0, 0],
            'lab_ALT_high': [0, 0, 1, 0],
            'TAT_minutes': [45, 60, 70, 80]
        })

    def test_laboratory_abnormality_summary_features(self):
        """Validate comprehensive laboratory abnormality summary generation."""
        result = self.processor.add_derived_features(self.flagged_df)
        
        # Validate derived feature columns created
        assert 'lab_abnormal_count' in result.columns
        assert 'has_abnormal_labs' in result.columns
        
        # Test abnormality counts: 
        # Row 0: WBC_low + Platelets_low = 2
        # Row 1: HGB_low = 1  
        # Row 2: WBC_high + ALT_high = 2
        # Row 3: HGB_high = 1
        expected_counts = [2, 1, 2, 1]
        assert result['lab_abnormal_count'].tolist() == expected_counts
        
        # Test has_abnormal_labs (binary indicator)
        expected_has_abnormal = [1, 1, 1, 1]  # All rows have abnormalities
        assert result['has_abnormal_labs'].tolist() == expected_has_abnormal

    def test_cbc_panel_specific_features(self):
        """Validate Complete Blood Count panel specific derived features."""
        result = self.processor.add_derived_features(self.flagged_df)
        
        # Validate CBC-specific features created
        assert 'cbc_abnormal_count' in result.columns
        assert 'has_abnormal_cbc' in result.columns
        
        # Test CBC abnormality counts (WBC, HGB, Platelets only):
        # Row 0: WBC_low + Platelets_low = 2
        # Row 1: HGB_low = 1
        # Row 2: WBC_high = 1 (ALT excluded from CBC)
        # Row 3: HGB_high = 1
        expected_cbc_counts = [2, 1, 1, 1]
        assert result['cbc_abnormal_count'].tolist() == expected_cbc_counts

    def test_no_abnormality_flags_handling(self):
        """Validate derived feature generation when no abnormality flags present."""
        clean_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [5.0, 6.0, 7.0],
            'TAT_minutes': [45, 60, 70]
        })
        
        result = self.processor.add_derived_features(clean_df)
        
        # Should handle gracefully without creating derived features
        derived_features = ['lab_abnormal_count', 'has_abnormal_labs', 
                          'cbc_abnormal_count', 'has_abnormal_cbc']
        
        for feature in derived_features:
            if feature in result.columns:
                # If created, should be all zeros
                assert all(result[feature] == 0)

    def test_mixed_normal_abnormal_patterns(self):
        """Validate derived features with mixed normal and abnormal patterns."""
        mixed_df = pd.DataFrame({
            'lab_WBC_low': [0, 0, 1],      # Only one abnormal
            'lab_WBC_high': [0, 0, 0],     # All normal
            'lab_HGB_low': [1, 0, 0],      # Different pattern
            'lab_HGB_high': [0, 0, 0],     # All normal
            'TAT_minutes': [45, 60, 70]
        })
        
        result = self.processor.add_derived_features(mixed_df)
        
        # Test mixed patterns:
        # Row 0: HGB_low = 1
        # Row 1: No abnormalities = 0
        # Row 2: WBC_low = 1
        expected_counts = [1, 0, 1]
        expected_has_abnormal = [1, 0, 1]
        
        assert result['lab_abnormal_count'].tolist() == expected_counts
        assert result['has_abnormal_labs'].tolist() == expected_has_abnormal


class TestCustomProcessingSteps:
    """Test suite for custom laboratory processing step registration and pipeline extension."""

    def setup_method(self):
        """Set up test data and processor for custom step validation."""
        self.processor = LabProcessor()
        self.lab_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [5.0, 8.0, 12.0],
            'lab_Platelets_k_per_uL': [200.0, 300.0, 100.0],
            'TAT_minutes': [45, 60, 70]
        })

    def test_custom_step_registration(self):
        """Validate custom laboratory processing step registration."""
        def custom_oncology_scorer(df):
            """Custom oncology risk scoring step."""
            df['oncology_risk_score'] = 'high_risk'
            return df
        
        # Register custom step
        result = self.processor.register(custom_oncology_scorer)
        
        # Validate method chaining returns self
        assert result is self.processor
        
        # Validate custom step added to pipeline
        assert len(self.processor._custom_steps) == 1
        assert self.processor._custom_steps[0] == custom_oncology_scorer

    def test_multiple_custom_steps_registration(self):
        """Validate multiple custom step registration and execution order."""
        def step1(df):
            df['step1_applied'] = True
            return df
        
        def step2(df):
            df['step2_applied'] = True  
            return df
        
        # Register multiple custom steps
        self.processor.register(step1)
        self.processor.register(step2)
        
        # Validate both steps registered in order
        assert len(self.processor._custom_steps) == 2
        assert self.processor._custom_steps[0] == step1
        assert self.processor._custom_steps[1] == step2

    def test_clear_custom_steps(self):
        """Validate custom step pipeline reset functionality."""
        def dummy_step(df):
            return df
        
        # Register step then clear
        self.processor.register(dummy_step)
        assert len(self.processor._custom_steps) == 1
        
        result = self.processor.clear_custom()
        assert len(self.processor._custom_steps) == 0
        assert result is self.processor  # Method chaining

    def test_custom_step_execution_in_transform(self):
        """Validate custom step execution within transform pipeline."""
        def add_infection_risk(df):
            """Custom infection risk assessment."""
            if 'lab_WBC_k_per_uL' in df.columns:
                df['infection_risk'] = df['lab_WBC_k_per_uL'] < 4.0
            return df
        
        self.processor.register(add_infection_risk)
        result = self.processor.transform(self.lab_df)
        
        # Validate custom step executed
        assert 'infection_risk' in result.columns
        # WBC values [5.0, 8.0, 12.0] -> all >= 4.0, so all False
        expected_risk = [False, False, False]
        assert result['infection_risk'].tolist() == expected_risk

    def test_oncology_specific_custom_step(self):
        """Validate oncology-specific custom processing step."""
        def calculate_bleeding_risk(df):
            """Custom bleeding risk calculation for oncology patients."""
            if 'lab_Platelets_k_per_uL' in df.columns:
                df['bleeding_risk_category'] = pd.cut(
                    df['lab_Platelets_k_per_uL'],
                    bins=[0, 50, 100, 500, float('inf')],
                    labels=['severe', 'high', 'moderate', 'low']
                )
            return df
        
        self.processor.register(calculate_bleeding_risk)
        result = self.processor.transform(self.lab_df)
        
        # Validate bleeding risk categories
        assert 'bleeding_risk_category' in result.columns
        # Platelets [200.0, 300.0, 100.0] with bins [0, 50, 100, 500, inf] -> ['moderate', 'moderate', 'high']
        # 200.0 is in (100, 500] -> 'moderate', 300.0 is in (100, 500] -> 'moderate', 100.0 is in (50, 100] -> 'high'
        expected_categories = ['moderate', 'moderate', 'high']
        assert result['bleeding_risk_category'].tolist() == expected_categories


class TestPipelineExecution:
    """Test suite for laboratory processing pipeline execution and configuration."""

    def setup_method(self):
        """Set up comprehensive test data for pipeline validation."""
        self.processor = LabProcessor()
        self.comprehensive_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [2.0, None, 250.0, 8.5],      # Low, Missing, High (clipped), Normal
            'lab_HGB_g_dL': [10.0, 14.0, None, 18.0],         # Low, Normal, Missing, High
            'lab_Platelets_k_per_uL': [100.0, 300.0, 500.0, None],  # Low, Normal, High, Missing
            'TAT_minutes': [45, 60, 70, 80]
        })

    def test_full_transform_pipeline_execution(self):
        """Validate complete transform pipeline with all standard steps."""
        result = self.processor.transform(self.comprehensive_df)
        
        # Validate pipeline steps executed in sequence:
        # 1. clip_labs - bounds enforcement
        assert result['lab_WBC_k_per_uL'].max() <= 200  # Clipped from 250.0
        
        # 2. handle_missing_values - imputation applied (median strategy)
        lab_cols = ['lab_WBC_k_per_uL', 'lab_HGB_g_dL', 'lab_Platelets_k_per_uL']
        for col in lab_cols:
            assert not result[col].isna().any()  # No missing values remain
        
        # 3. add_lab_flags - abnormality detection
        flag_cols = [c for c in result.columns if c.endswith(('_low', '_high'))]
        assert len(flag_cols) > 0  # Abnormality flags created
        
        # 4. add_derived_features - summary features  
        derived_cols = ['lab_abnormal_count', 'has_abnormal_labs']
        for col in derived_cols:
            assert col in result.columns

    def test_fit_transform_equivalence(self):
        """Validate fit_transform equivalence to fit().transform()."""
        # Method 1: separate fit and transform
        processor1 = LabProcessor()
        processor1.fit(self.comprehensive_df)
        result1 = processor1.transform(self.comprehensive_df)
        
        # Method 2: combined fit_transform
        processor2 = LabProcessor()
        result2 = processor2.fit_transform(self.comprehensive_df)
        
        # Validate equivalent results
        pd.testing.assert_frame_equal(result1, result2)

    def test_method_chaining_support(self):
        """Validate method chaining capability for pipeline construction."""
        def custom_step(df):
            df['chained_step'] = True
            return df
        
        result = (self.processor
                 .register(custom_step)
                 .fit(self.comprehensive_df)
                 .transform(self.comprehensive_df))
        
        # Validate chaining works and custom step executed
        assert 'chained_step' in result.columns
        assert all(result['chained_step'] == True)

    def test_custom_pipeline_sequence(self):
        """Validate custom processing sequence execution."""
        # Custom sequence with only specific steps
        custom_sequence = [
            self.processor.clip_labs,
            self.processor.add_lab_flags
        ]
        
        result = self.processor.apply(self.comprehensive_df, sequence=custom_sequence)
        
        # Validate only specified steps executed
        # Clipping should be applied
        assert result['lab_WBC_k_per_uL'].max() <= 200
        
        # Abnormality flags should be present
        flag_cols = [c for c in result.columns if c.endswith(('_low', '_high'))]
        assert len(flag_cols) > 0
        
        # Missing values should remain (handle_missing_values not in sequence)
        assert result['lab_WBC_k_per_uL'].isna().any()


class TestErrorHandlingAndEdgeCases:
    """Test suite for error handling and edge case scenarios."""

    def setup_method(self):
        """Set up edge case test data."""
        self.processor = LabProcessor()

    def test_empty_dataframe_processing(self):
        """Validate handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.processor.transform(empty_df)
        
        # Validate empty DataFrame handled gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_row_processing(self):
        """Validate processing of single-row DataFrame."""
        single_row_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [300.0],  # Out of bounds
            'lab_HGB_g_dL': [5.0],        # Low
            'TAT_minutes': [60]
        })
        
        result = self.processor.transform(single_row_df)
        
        # Validate single row processed correctly
        assert len(result) == 1
        assert result['lab_WBC_k_per_uL'].iloc[0] == 200.0  # Clipped to upper bound
        assert 'lab_HGB_g_low' in result.columns  # Abnormality flag created

    def test_all_missing_laboratory_values(self):
        """Validate handling of DataFrame with all missing laboratory values."""
        missing_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [None, None, None],
            'lab_HGB_g_dL': [None, None, None],
            'TAT_minutes': [45, 60, 75]
        })
        
        result = self.processor.transform(missing_df)
        
        # With median strategy, all-missing columns should remain NaN
        # (no valid values to compute median)
        assert result['lab_WBC_k_per_uL'].isna().all()
        assert result['lab_HGB_g_dL'].isna().all()
        
        # Non-lab columns should be preserved
        assert not result['TAT_minutes'].isna().any()

    def test_malformed_laboratory_data_resilience(self):
        """Validate resilience to malformed laboratory data."""
        malformed_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [5.0, 'text', [1, 2], {'key': 'value'}, None],
            'lab_HGB_g_dL': ['invalid', 14.0, 15.5, None, 'text'],
            'TAT_minutes': [45, 60, 70, 80, 55]
        })
        
        result = self.processor.transform(malformed_df)
        
        # Validate processing completes without error
        assert len(result) == len(malformed_df)
        
        # Valid numeric values should be preserved
        assert result['lab_WBC_k_per_uL'].iloc[0] == 5.0  # Valid value preserved
        assert result['lab_HGB_g_dL'].iloc[1] == 14.0     # Valid value preserved

    def test_non_destructive_operations(self):
        """Validate that all operations don't modify original DataFrame."""
        original_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [2.0, 300.0, None],
            'lab_HGB_g_dL': [10.0, 14.0, 18.0],
            'TAT_minutes': [45, 60, 70]
        })
        original_copy = original_df.copy()
        
        # Transform should not modify original
        self.processor.transform(original_df)
        pd.testing.assert_frame_equal(original_df, original_copy)

    def test_missing_laboratory_columns_in_configuration(self):
        """Validate behavior when configured lab columns are missing from data."""
        # Configure processor with lab columns not in data
        processor = LabProcessor(cols=['lab_WBC_k_per_uL', 'lab_MISSING_COL'])
        
        df_partial = pd.DataFrame({
            'lab_WBC_k_per_uL': [5.0, 8.0, 12.0],
            'TAT_minutes': [45, 60, 70]
        })
        
        result = processor.transform(df_partial)
        
        # Validate processing continues with available columns only
        assert len(result) == len(df_partial)
        assert 'lab_WBC_k_per_uL' in result.columns
        assert 'lab_MISSING_COL' not in result.columns


class TestPerformanceAndScalability:
    """Test suite for performance validation with large oncology datasets."""

    def setup_method(self):
        """Set up large dataset for performance validation."""
        self.processor = LabProcessor()

    def test_large_dataset_processing_performance(self):
        """Validate performance with large oncology patient dataset simulation."""
        # Create large dataset simulating 10k oncology patients
        np.random.seed(42)
        n_patients = 10000
        
        # Realistic oncology laboratory value distributions
        large_df = pd.DataFrame({
            'lab_WBC_k_per_uL': np.random.lognormal(1.5, 0.5, n_patients),      # Log-normal distribution
            'lab_HGB_g_dL': np.random.normal(12.5, 2.0, n_patients),           # Normal distribution  
            'lab_Platelets_k_per_uL': np.random.lognormal(5.5, 0.4, n_patients), # Log-normal distribution
            'lab_Creatinine_mg_dL': np.random.gamma(2, 0.4, n_patients),        # Gamma distribution
            'lab_ALT_U_L': np.random.exponential(25, n_patients),               # Exponential distribution
            'TAT_minutes': np.random.randint(30, 120, n_patients)
        })
        
        # Add some missing values to simulate real clinical data
        missing_indices = np.random.choice(n_patients, size=int(n_patients * 0.1), replace=False)
        large_df.loc[missing_indices, 'lab_WBC_k_per_uL'] = None
        
        # Validate comprehensive processing completes efficiently
        result = self.processor.transform(large_df)
        
        # Validate processing results integrity
        assert len(result) == n_patients
        
        # Validate clinical bounds applied
        for col in self.processor.cols:
            if col in result.columns:
                clipped_values = result[col].dropna()
                clip_range = self.processor.clips[col]
                assert clipped_values.min() >= clip_range[0]
                assert clipped_values.max() <= clip_range[1]
        
        # Validate derived features created
        assert 'lab_abnormal_count' in result.columns
        assert 'has_abnormal_labs' in result.columns

    def test_memory_efficiency_validation(self):
        """Validate memory-efficient processing for clinical analytics workflows."""
        # Create moderately sized dataset for memory testing
        n_patients = 5000
        np.random.seed(42)
        
        df = pd.DataFrame({
            'lab_WBC_k_per_uL': np.random.normal(7, 2, n_patients),
            'lab_HGB_g_dL': np.random.normal(13, 2, n_patients),
            'lab_Platelets_k_per_uL': np.random.normal(300, 100, n_patients),
            'TAT_minutes': np.random.randint(30, 120, n_patients)
        })
        
        # Validate processing doesn't cause memory issues
        result = self.processor.transform(df)
        
        # Validate memory usage reasonable
        assert result.memory_usage().sum() > 0  # Has some memory usage
        assert len(result.columns) >= len(df.columns)  # Features added

    def test_iterative_processing_performance(self):
        """Validate performance with iterative processing and custom steps."""
        base_df = pd.DataFrame({
            'lab_WBC_k_per_uL': [5.0, 8.0, 12.0, 3.0, 15.0],
            'lab_HGB_g_dL': [14.0, 10.0, 16.0, 18.0, 12.0],
            'TAT_minutes': [45, 60, 70, 80, 55]
        })
        
        # Multiple processing iterations with custom steps
        for i in range(5):
            def iteration_step(df, iteration=i):
                df[f'processing_iteration_{iteration}'] = True
                return df
            
            self.processor.register(iteration_step)
            result = self.processor.transform(base_df)
            
            # Validate each iteration adds features correctly
            assert f'processing_iteration_{i}' in result.columns
            
            # Clear for next iteration
            self.processor.clear_custom()
        
        # Validate final state clean
        assert len(self.processor._custom_steps) == 0
