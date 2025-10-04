"""
Test suite for TAT configuration module.
"""
import pytest
from src.tat.config import (
    STEP_COLS, DELAY_PAIRS, HELPER_SUFFIXES, DELAY_PLOT_ORDER,
    SHIFT_BOUNDARIES, BUSINESS_HOURS, WEEKEND_DAYS, NON_FEATURE_COLS,
    TARGETS, ORDER_TIME_COL, CATEGORICAL_PREFIX_MAP, CATEGORICAL_MAPPINGS,
    LAB_COLS, LAB_NORMALS, LAB_CLIPS, REQUIRED_COLUMNS, CLEANING_CONFIG,
    TRAINING_CONFIGS, HEALTHCARE_THRESHOLDS, MONITORING_CONFIG, MLOPS_CONFIG
)


class TestStepConfiguration:
    """Test medication preparation workflow step configurations."""
    
    def test_step_cols_structure(self):
        """Test that step columns are properly defined."""
        assert isinstance(STEP_COLS, list)
        assert len(STEP_COLS) == 5
        
        expected_steps = [
            "nurse_validation_time",
            "prep_complete_time", 
            "second_validation_time",
            "floor_dispatch_time",
            "patient_infusion_time"
        ]
        assert STEP_COLS == expected_steps
    
    def test_delay_pairs_structure(self):
        """Test delay pair configurations for bottleneck analysis."""
        assert isinstance(DELAY_PAIRS, list)
        assert len(DELAY_PAIRS) == 5
        
        # Check structure of each delay pair
        for pair in DELAY_PAIRS:
            assert isinstance(pair, tuple)
            assert len(pair) == 3
            # First element can be None or string
            assert pair[0] is None or isinstance(pair[0], str)
            # Second and third elements must be strings
            assert isinstance(pair[1], str)
            assert isinstance(pair[2], str)
    
    def test_delay_plot_order_consistency(self):
        """Test that delay plot order matches delay pairs."""
        assert isinstance(DELAY_PLOT_ORDER, list)
        assert len(DELAY_PLOT_ORDER) == len(DELAY_PAIRS)
        
        # Extract delay names from DELAY_PAIRS for comparison
        delay_names = [pair[2] for pair in DELAY_PAIRS]
        assert set(DELAY_PLOT_ORDER) == set(delay_names)
    
    def test_helper_suffixes(self):
        """Test helper suffix configurations."""
        assert isinstance(HELPER_SUFFIXES, tuple)
        assert len(HELPER_SUFFIXES) == 2
        assert "_mins_unwrapped" in HELPER_SUFFIXES
        assert "_dt" in HELPER_SUFFIXES


class TestTemporalConfiguration:
    """Test temporal analysis configurations."""
    
    def test_shift_boundaries(self):
        """Test healthcare shift boundary definitions."""
        assert isinstance(SHIFT_BOUNDARIES, dict)
        assert len(SHIFT_BOUNDARIES) == 3
        
        required_shifts = {"Day", "Evening", "Night"}
        assert set(SHIFT_BOUNDARIES.keys()) == required_shifts
        
        # Validate shift time ranges
        for shift, (start, end) in SHIFT_BOUNDARIES.items():
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert 0 <= start <= 23
            assert 0 <= end <= 23
    
    def test_business_hours(self):
        """Test business hours configuration."""
        assert isinstance(BUSINESS_HOURS, tuple)
        assert len(BUSINESS_HOURS) == 2
        start, end = BUSINESS_HOURS
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start < end
        assert 0 <= start <= 23
        assert 0 <= end <= 23
    
    def test_weekend_days(self):
        """Test weekend day configuration."""
        assert isinstance(WEEKEND_DAYS, tuple)
        assert len(WEEKEND_DAYS) == 2
        assert all(isinstance(day, int) for day in WEEKEND_DAYS)
        assert all(0 <= day <= 6 for day in WEEKEND_DAYS)
        assert WEEKEND_DAYS == (5, 6)  # Saturday and Sunday


class TestColumnConfiguration:
    """Test column and feature configurations."""
    
    def test_non_feature_cols(self):
        """Test non-feature column exclusions."""
        assert isinstance(NON_FEATURE_COLS, list)
        assert len(NON_FEATURE_COLS) == 3
        
        expected_non_features = ["order_id", "patient_id", "ordering_physician"]
        assert set(NON_FEATURE_COLS) == set(expected_non_features)
    
    def test_targets(self):
        """Test prediction target configurations."""
        assert isinstance(TARGETS, list)
        assert len(TARGETS) == 2
        assert "TAT_minutes" in TARGETS
        assert "TAT_over_60" in TARGETS
    
    def test_order_time_col(self):
        """Test primary timestamp column."""
        assert isinstance(ORDER_TIME_COL, str)
        assert ORDER_TIME_COL == "doctor_order_time"
    
    def test_categorical_prefix_map(self):
        """Test categorical encoding prefix mappings."""
        assert isinstance(CATEGORICAL_PREFIX_MAP, dict)
        assert len(CATEGORICAL_PREFIX_MAP) >= 10
        
        # Verify all values are strings
        for key, value in CATEGORICAL_PREFIX_MAP.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert len(value) > 0


class TestCategoricalMappings:
    """Test categorical value mappings."""
    
    def test_categorical_mappings_structure(self):
        """Test categorical mappings structure."""
        assert isinstance(CATEGORICAL_MAPPINGS, dict)
        
        required_categories = {'sex', 'severity', 'nurse_credential', 'pharmacist_credential'}
        assert required_categories.issubset(set(CATEGORICAL_MAPPINGS.keys()))
    
    def test_sex_mapping(self):
        """Test gender mapping values."""
        sex_mapping = CATEGORICAL_MAPPINGS['sex']
        assert isinstance(sex_mapping, dict)
        assert sex_mapping == {"F": 0, "M": 1}
    
    def test_severity_mapping(self):
        """Test severity level mappings."""
        severity_mapping = CATEGORICAL_MAPPINGS['severity']
        assert isinstance(severity_mapping, dict)
        assert severity_mapping == {"Low": 0, "Medium": 1, "High": 2}
        
        # Test ordinal nature
        assert severity_mapping["Low"] < severity_mapping["Medium"] < severity_mapping["High"]
    
    def test_credential_mappings(self):
        """Test healthcare credential mappings."""
        nurse_mapping = CATEGORICAL_MAPPINGS['nurse_credential']
        pharm_mapping = CATEGORICAL_MAPPINGS['pharmacist_credential']
        
        # Test hierarchical ordering
        assert nurse_mapping["RN"] < nurse_mapping["BSN"] < nurse_mapping["MSN"] < nurse_mapping["NP"]
        assert pharm_mapping["RPh"] < pharm_mapping["PharmD"] < pharm_mapping["BCOP"]


class TestLaboratoryConfiguration:
    """Test laboratory test configurations."""
    
    def test_lab_cols_structure(self):
        """Test laboratory column definitions."""
        assert isinstance(LAB_COLS, list)
        assert len(LAB_COLS) == 5
        
        expected_labs = [
            "lab_WBC_k_per_uL", "lab_HGB_g_dL", "lab_Platelets_k_per_uL",
            "lab_Creatinine_mg_dL", "lab_ALT_U_L"
        ]
        assert set(LAB_COLS) == set(expected_labs)
    
    def test_lab_normals_structure(self):
        """Test laboratory normal range definitions."""
        assert isinstance(LAB_NORMALS, dict)
        assert len(LAB_NORMALS) == len(LAB_COLS)
        
        # Test all lab columns have normal ranges
        for lab_col in LAB_COLS:
            assert lab_col in LAB_NORMALS
            normal_range = LAB_NORMALS[lab_col]
            assert isinstance(normal_range, tuple)
            assert len(normal_range) == 2
            assert normal_range[0] < normal_range[1]
            assert all(isinstance(val, (int, float)) for val in normal_range)
    
    def test_lab_clips_structure(self):
        """Test laboratory clipping range definitions."""
        assert isinstance(LAB_CLIPS, dict)
        assert len(LAB_CLIPS) == len(LAB_COLS)
        
        # Test all lab columns have clipping ranges
        for lab_col in LAB_COLS:
            assert lab_col in LAB_CLIPS
            clip_range = LAB_CLIPS[lab_col]
            assert isinstance(clip_range, tuple)
            assert len(clip_range) == 2
            assert clip_range[0] < clip_range[1]
            assert all(isinstance(val, (int, float)) for val in clip_range)
    
    def test_lab_ranges_consistency(self):
        """Test that clipping ranges contain normal ranges."""
        for lab_col in LAB_COLS:
            normal_range = LAB_NORMALS[lab_col]
            clip_range = LAB_CLIPS[lab_col]
            
            # Clipping range should contain normal range
            assert clip_range[0] <= normal_range[0]
            assert normal_range[1] <= clip_range[1]


class TestRequiredColumns:
    """Test required column configurations."""
    
    def test_required_columns_structure(self):
        """Test required columns set definition."""
        assert isinstance(REQUIRED_COLUMNS, set)
        assert len(REQUIRED_COLUMNS) >= 15
        
        # Test essential columns are included
        essential_cols = [
            'age', 'sex', 'race_ethnicity', 'insurance_type', 'diagnosis_type',
            'severity', 'treatment_type', 'patient_readiness_score', 'premed_required',
            'stat_order', 'floor', 'shift', 'nurse_credential', 'pharmacist_credential'
        ]
        
        for col in essential_cols:
            assert col in REQUIRED_COLUMNS
        
        # Test ORDER_TIME_COL is included
        assert ORDER_TIME_COL in REQUIRED_COLUMNS


class TestCleaningConfiguration:
    """Test data cleaning configurations."""
    
    def test_cleaning_config_structure(self):
        """Test data cleaning configuration structure."""
        assert isinstance(CLEANING_CONFIG, dict)
        
        required_keys = {'age_bounds', 'years_bounds', 'years_cols', 'binary_cols'}
        assert required_keys.issubset(set(CLEANING_CONFIG.keys()))
    
    def test_age_bounds(self):
        """Test age boundary configurations."""
        age_bounds = CLEANING_CONFIG['age_bounds']
        assert isinstance(age_bounds, tuple)
        assert len(age_bounds) == 2
        assert age_bounds == (0, 120)
        assert age_bounds[0] < age_bounds[1]
    
    def test_years_configuration(self):
        """Test employment years configurations."""
        years_bounds = CLEANING_CONFIG['years_bounds']
        years_cols = CLEANING_CONFIG['years_cols']
        
        assert isinstance(years_bounds, tuple)
        assert len(years_bounds) == 2
        assert years_bounds[0] < years_bounds[1]
        
        assert isinstance(years_cols, list)
        assert len(years_cols) >= 2
        assert "nurse_employment_years" in years_cols
        assert "pharmacist_employment_years" in years_cols
    
    def test_binary_cols(self):
        """Test binary column configurations."""
        binary_cols = CLEANING_CONFIG['binary_cols']
        assert isinstance(binary_cols, list)
        assert "premed_required" in binary_cols
        assert "stat_order" in binary_cols


class TestTrainingConfigurations:
    """Test model training configurations."""
    
    def test_training_configs_structure(self):
        """Test training configuration structure."""
        assert isinstance(TRAINING_CONFIGS, dict)
        assert len(TRAINING_CONFIGS) >= 3
        
        expected_configs = {
            'linear_interpretable', 'tree_based_models', 'comprehensive_ensemble'
        }
        assert expected_configs.issubset(set(TRAINING_CONFIGS.keys()))
    
    def test_config_completeness(self):
        """Test each training configuration has required fields."""
        for config_name, config in TRAINING_CONFIGS.items():
            assert isinstance(config, dict)
            
            required_fields = {
                'scaling_strategy', 'description', 'focus',
                'clinical_applications', 'deployment_scenarios'
            }
            assert required_fields.issubset(set(config.keys()))
            
            # Test field types
            assert isinstance(config['scaling_strategy'], str)
            assert isinstance(config['description'], str)
            assert isinstance(config['focus'], str)
            assert isinstance(config['clinical_applications'], list)
            assert isinstance(config['deployment_scenarios'], list)


class TestHealthcareThresholds:
    """Test healthcare quality threshold configurations."""
    
    def test_healthcare_thresholds_structure(self):
        """Test healthcare threshold definitions."""
        assert isinstance(HEALTHCARE_THRESHOLDS, dict)
        
        required_thresholds = {
            'tat_target_minutes', 'tat_warning_minutes', 'tat_critical_minutes',
            'acceptable_violation_rate', 'model_accuracy_minimum', 
            'bottleneck_significance_threshold'
        }
        assert required_thresholds.issubset(set(HEALTHCARE_THRESHOLDS.keys()))
    
    def test_tat_threshold_ordering(self):
        """Test TAT threshold logical ordering."""
        warning = HEALTHCARE_THRESHOLDS['tat_warning_minutes']
        target = HEALTHCARE_THRESHOLDS['tat_target_minutes']
        critical = HEALTHCARE_THRESHOLDS['tat_critical_minutes']
        
        assert warning < target < critical
        assert all(isinstance(val, (int, float)) for val in [warning, target, critical])
    
    def test_rate_thresholds(self):
        """Test rate-based thresholds are within valid ranges."""
        violation_rate = HEALTHCARE_THRESHOLDS['acceptable_violation_rate']
        accuracy_min = HEALTHCARE_THRESHOLDS['model_accuracy_minimum']
        
        assert 0 <= violation_rate <= 1
        assert 0 <= accuracy_min <= 1


class TestMonitoringConfiguration:
    """Test production monitoring configurations."""
    
    def test_monitoring_config_structure(self):
        """Test monitoring configuration structure."""
        assert isinstance(MONITORING_CONFIG, dict)
        
        required_keys = {'performance_metrics', 'data_drift_metrics', 'alert_thresholds'}
        assert required_keys.issubset(set(MONITORING_CONFIG.keys()))
    
    def test_performance_metrics(self):
        """Test performance metric definitions."""
        metrics = MONITORING_CONFIG['performance_metrics']
        assert isinstance(metrics, list)
        assert len(metrics) >= 4
        
        expected_metrics = [
            'RMSE_minutes', 'MAE_minutes', 'threshold_compliance_rate',
            'bottleneck_detection_accuracy'
        ]
        assert set(expected_metrics).issubset(set(metrics))
    
    def test_alert_thresholds(self):
        """Test alert threshold configurations."""
        thresholds = MONITORING_CONFIG['alert_thresholds']
        assert isinstance(thresholds, dict)
        
        required_thresholds = {
            'rmse_degradation_pct', 'accuracy_drop_threshold', 'data_drift_significance'
        }
        assert required_thresholds.issubset(set(thresholds.keys()))


class TestMLOpsConfiguration:
    """Test MLOps deployment configurations."""
    
    def test_mlops_config_structure(self):
        """Test MLOps configuration structure."""
        assert isinstance(MLOPS_CONFIG, dict)
        
        required_keys = {
            'model_refresh_frequency', 'validation_split_strategy',
            'feature_selection_threshold', 'ensemble_voting_strategy',
            'production_latency_target_ms', 'rollback_criteria'
        }
        assert required_keys.issubset(set(MLOPS_CONFIG.keys()))
    
    def test_rollback_criteria(self):
        """Test rollback criteria configurations."""
        rollback = MLOPS_CONFIG['rollback_criteria']
        assert isinstance(rollback, dict)
        
        required_criteria = {
            'performance_degradation_threshold', 'error_rate_threshold', 'latency_threshold_ms'
        }
        assert required_criteria.issubset(set(rollback.keys()))
    
    def test_latency_targets(self):
        """Test latency target configurations."""
        production_latency = MLOPS_CONFIG['production_latency_target_ms']
        rollback_latency = MLOPS_CONFIG['rollback_criteria']['latency_threshold_ms']
        
        assert isinstance(production_latency, int)
        assert isinstance(rollback_latency, int)
        assert production_latency > 0
        assert rollback_latency > production_latency


class TestConfigurationIntegrity:
    """Test overall configuration integrity and consistency."""
    
    def test_categorical_mapping_consistency(self):
        """Test consistency between categorical mappings and prefix maps."""
        # Categories in CATEGORICAL_MAPPINGS should have prefixes
        for category in CATEGORICAL_MAPPINGS.keys():
            if category in CATEGORICAL_PREFIX_MAP:
                assert isinstance(CATEGORICAL_PREFIX_MAP[category], str)
    
    def test_lab_configuration_consistency(self):
        """Test laboratory configuration consistency."""
        # All lab columns should have both normals and clips
        assert set(LAB_NORMALS.keys()) == set(LAB_COLS)
        assert set(LAB_CLIPS.keys()) == set(LAB_COLS)
    
    def test_threshold_value_consistency(self):
        """Test healthcare threshold value consistency."""
        # TAT thresholds should be positive
        for key, value in HEALTHCARE_THRESHOLDS.items():
            if 'minutes' in key:
                assert value > 0
            elif 'rate' in key or 'threshold' in key:
                assert 0 <= value <= 1 or value > 0
    
    def test_no_duplicate_configurations(self):
        """Test that configurations don't have unexpected duplicates."""
        # Step columns should be unique
        assert len(STEP_COLS) == len(set(STEP_COLS))
        
        # Lab columns should be unique
        assert len(LAB_COLS) == len(set(LAB_COLS))
        
        # Non-feature columns should be unique
        assert len(NON_FEATURE_COLS) == len(set(NON_FEATURE_COLS))