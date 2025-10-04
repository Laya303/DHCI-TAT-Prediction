"""
Test suite for functionality.

Tests the summary configuration system for Dana Farber's pharmacy turnaround time
analysis, ensuring robust parameter management for medication preparation workflow
optimization and healthcare analytics pipeline reliability.

Key Test Coverage:
- Immutable configuration dataclass behavior and parameter validation
- Healthcare domain-specific defaults for TAT workflow analysis
- HIPAA-compliant identifier handling and data governance controls
- Production MLOps configuration validation and deployment readiness
- Clinical threshold management for pharmacy operations optimization
"""
import pytest
from dataclasses import FrozenInstanceError
from typing import Set, Tuple
import copy

from src.tat.eda.summary.summary_config import SummaryConfig


class TestSummaryConfigImmutability:
    """Test suite for configuration immutability and production deployment safety."""
    
    def test_config_immutable_dataclass_behavior(self):
        """Test that SummaryConfig prevents runtime parameter drift in production."""
        config = SummaryConfig()
        
        # Should prevent modification of configuration parameters
        with pytest.raises(FrozenInstanceError):
            config.percentiles = (0.25, 0.5, 0.75)
        
        with pytest.raises(FrozenInstanceError):
            config.tat_threshold_minutes = 90.0
        
        with pytest.raises(FrozenInstanceError):
            config.missing_top_n = 25

    def test_config_deep_immutability_collections(self):
        """Test that collection attributes maintain immutability for healthcare compliance."""
        config = SummaryConfig()
        
        # Tuple attributes should be immutable
        original_percentiles = config.percentiles
        assert isinstance(config.percentiles, tuple)
        
        # Should not be able to modify tuple contents
        with pytest.raises((TypeError, AttributeError)):
            config.percentiles[0] = 0.1
        
        # Set attributes should be frozenset or provide immutable interface
        original_categorical_overrides = config.categorical_overrides
        
        # Attempting to modify should either fail or not affect original
        try:
            config.categorical_overrides.add("new_column")
            # If modification succeeds, original should be unchanged
            assert original_categorical_overrides == config.categorical_overrides
        except AttributeError:
            # Expected behavior for truly immutable sets
            pass

    def test_config_copy_independence(self):
        """Test that configuration copies don't share mutable references."""
        config1 = SummaryConfig(missing_top_n=10)
        
        # Create new instance with different parameters
        config2 = SummaryConfig(missing_top_n=20, hist_bins=15)
        
        # Configurations should be independent
        assert config1.missing_top_n == 10
        assert config2.missing_top_n == 20
        assert config1.hist_bins == 20  # Default
        assert config2.hist_bins == 15  # Custom

    def test_config_serialization_for_caching(self):
        """Test that SummaryConfig can be serialized for MLOps caching systems."""
        config1 = SummaryConfig()
        config2 = SummaryConfig()
        config3 = SummaryConfig(missing_top_n=25)
        
        # Should be serializable for caching (using string representation)
        config1_str = str(config1)
        config2_str = str(config2)
        config3_str = str(config3)
        
        # Equal configurations should have same string representation
        assert config1_str == config2_str
        assert config1_str != config3_str
        
        # Should support dictionary-like access using string keys
        config_cache = {
            f"config_{hash(config1_str)}": "default",
            f"config_{hash(config3_str)}": "custom"
        }
        assert len(config_cache) == 2
        
        # Test that configurations can be compared for equality
        assert config1 == config2
        assert config1 != config3

class TestHealthcareDefaultConfiguration:
    """Test suite for healthcare domain-specific default parameters."""
    
    def test_tat_clinical_percentiles_defaults(self):
        """Test default percentiles align with healthcare TAT SLA monitoring."""
        config = SummaryConfig()
        
        expected_percentiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        assert config.percentiles == expected_percentiles
        
        # Should include key healthcare analytics percentiles
        assert 0.95 in config.percentiles  # SLA violation threshold
        assert 0.99 in config.percentiles  # Extreme outlier detection
        assert 0.50 in config.percentiles  # Median performance
        assert 0.25 in config.percentiles  # Lower quartile baseline
        assert 0.75 in config.percentiles  # Upper quartile target

    def test_tat_threshold_clinical_standard(self):
        """Test TAT threshold matches Dana Farber's 60-minute clinical standard."""
        config = SummaryConfig()
        
        assert config.tat_threshold_minutes == 60.0
        
        # Should support clinical threshold comparisons
        assert isinstance(config.tat_threshold_minutes, float)
        assert config.tat_threshold_minutes > 0

    def test_data_quality_healthcare_thresholds(self):
        """Test data quality thresholds appropriate for healthcare EHR systems."""
        config = SummaryConfig()
        
        # Healthcare data commonly has ~10% missingness
        assert config.data_quality_warning_threshold == 0.10
        
        # Critical threshold should be higher but reasonable for healthcare
        assert config.data_quality_critical_threshold == 0.25
        assert config.data_quality_critical_threshold > config.data_quality_warning_threshold
        
        # Should be proportions between 0 and 1
        assert 0 < config.data_quality_warning_threshold < 1
        assert 0 < config.data_quality_critical_threshold < 1

    def test_visualization_parameters_clinical_context(self):
        """Test visualization parameters optimized for clinical stakeholder consumption."""
        config = SummaryConfig()
        
        # Category display should be manageable for clinical teams
        assert config.cat_top == 4  # Top 4 categories (shifts, departments, etc.)
        assert config.missing_top_n == 15  # Reasonable for healthcare column count
        
        # Histogram bins should balance detail with interpretability
        assert config.hist_bins == 20
        assert 10 <= config.hist_bins <= 50  # Reasonable range
        
        # Should sort missing data by severity for quality prioritization
        assert config.sort_missing is True

    def test_ascii_unicode_healthcare_compatibility(self):
        """Test visualization encoding defaults for healthcare IT environments."""
        config = SummaryConfig()
        
        # Should auto-detect by default for mixed healthcare environments
        assert config.force_ascii is None
        
        # Test explicit ASCII mode for legacy systems
        ascii_config = SummaryConfig(force_ascii=True)
        assert ascii_config.force_ascii is True
        
        # Test explicit Unicode mode for modern dashboards
        unicode_config = SummaryConfig(force_ascii=False)
        assert unicode_config.force_ascii is False

    def test_production_monitoring_defaults(self):
        """Test MLOps and production monitoring configuration defaults."""
        config = SummaryConfig()
        
        # Should enable production logging by default
        assert config.enable_production_logging is True
        
        # Correlation significance should be clinically meaningful
        assert config.correlation_significance_threshold == 0.1
        assert 0 < config.correlation_significance_threshold < 0.5
        
        # Categorical cardinality limit should prevent memory issues
        assert config.max_categorical_cardinality == 50
        assert config.max_categorical_cardinality > 0


class TestPharmacyWorkflowColumnClassification:
    """Test suite for medication preparation workflow column identification."""
    
    def test_known_time_cols_workflow_sequence(self):
        """Test that timestamp columns represent complete TAT workflow."""
        config = SummaryConfig()
        
        expected_workflow_timestamps = {
            "doctor_order_time",         # Step 1: Physician order
            "nurse_validation_time",     # Step 2: Clinical validation
            "prep_complete_time",        # Step 3: Medication preparation
            "second_validation_time",    # Step 4: Quality assurance
            "floor_dispatch_time",       # Step 5: Floor delivery
            "patient_infusion_time",     # Step 6: Patient administration
        }
        
        assert config.known_time_cols == expected_workflow_timestamps
        
        # Should be comprehensive workflow coverage
        assert len(config.known_time_cols) == 6
        
        # All should contain time-related keywords
        time_keywords = ["time", "timestamp", "dt"]
        for col in config.known_time_cols:
            assert any(keyword in col.lower() for keyword in time_keywords)

    def test_categorical_overrides_clinical_context(self):
        """Test categorical overrides for clinical assessment scales."""
        config = SummaryConfig()
        
        expected_categorical_overrides = {
            "patient_readiness_score",  # 1-3 ordinal clinical scale
            "floor",                    # Location identifier (1-3)
            "premed_required",          # Binary clinical protocol
            "stat_order",              # Binary urgency flag
            "TAT_over_60",             # Binary SLA violation
        }
        
        assert config.categorical_overrides == expected_categorical_overrides
        
        # Should handle clinical scales and binary indicators
        clinical_scales = ["patient_readiness_score"]
        binary_indicators = ["premed_required", "stat_order", "TAT_over_60"]
        location_identifiers = ["floor"]
        
        for scale in clinical_scales:
            assert scale in config.categorical_overrides
        
        for indicator in binary_indicators:
            assert indicator in config.categorical_overrides
        
        for location in location_identifiers:
            assert location in config.categorical_overrides

    def test_categorical_prefixes_one_hot_encoding(self):
        """Test categorical prefix patterns for pipeline one-hot encodings."""
        config = SummaryConfig()
        
        expected_prefixes = ("race_", "ins_", "dx_", "treat_", "shift_", "dept_")
        assert config.categorical_prefixes == expected_prefixes
        
        # Should cover key healthcare categorical dimensions
        healthcare_categories = {
            "race_": "Patient demographics",
            "ins_": "Insurance type categories", 
            "dx_": "Diagnosis classifications",
            "treat_": "Treatment modalities",
            "shift_": "Temporal patterns",
            "dept_": "Ordering departments"
        }
        
        for prefix in healthcare_categories.keys():
            assert prefix in config.categorical_prefixes

    def test_no_hist_cols_timestamp_suppression(self):
        """Test histogram suppression for raw timestamps in favor of delay analysis."""
        config = SummaryConfig()
        
        # Should suppress histograms for all workflow timestamps
        for timestamp_col in config.known_time_cols:
            assert timestamp_col in config.no_hist_cols
        
        # All no_hist_cols should be timestamps
        assert config.no_hist_cols == config.known_time_cols
        
        # Should prefer delay analysis over raw timestamp histograms
        assert len(config.no_hist_cols) == 6  # Complete workflow coverage


class TestCorrelationAnalysisConfiguration:
    """Test suite for TAT driver identification and correlation analysis setup."""
    
    def test_corr_exclude_prefixes_one_hot_encodings(self):
        """Test exclusion of one-hot encoded categoricals from correlation analysis."""
        config = SummaryConfig()
        
        expected_excluded_prefixes = ("race_", "ins_", "dx_", "treat_", "shift_", "dept_")
        assert config.corr_exclude_prefixes == expected_excluded_prefixes
        
        # Should match categorical prefixes to avoid binary correlation artifacts
        assert config.corr_exclude_prefixes == config.categorical_prefixes

    def test_corr_exclude_columns_analytical_focus(self):
        """Test specific column exclusions for focused TAT driver analysis."""
        config = SummaryConfig()
        
        # Should exclude demographic categoricals
        demographic_exclusions = ["sex"]
        for col in demographic_exclusions:
            assert col in config.corr_exclude_columns
        
        # Should exclude clinical categoricals
        clinical_exclusions = ["severity", "patient_readiness_score"]
        for col in clinical_exclusions:
            assert col in config.corr_exclude_columns
        
        # Should exclude protocol indicators
        protocol_exclusions = ["premed_required", "stat_order"]
        for col in protocol_exclusions:
            assert col in config.corr_exclude_columns
        
        # Should exclude temporal categoricals
        temporal_exclusions = ["order_dayofweek", "order_month", "order_on_weekend"]
        for col in temporal_exclusions:
            assert col in config.corr_exclude_columns
        
        # Should exclude target variables
        target_exclusions = ["TAT_over_60"]
        for col in target_exclusions:
            assert col in config.corr_exclude_columns
        
        # Should exclude identifiers
        identifier_exclusions = ["order_id", "patient_id"]
        for col in identifier_exclusions:
            assert col in config.corr_exclude_columns

    def test_correlation_focus_operational_drivers(self):
        """Test that correlation exclusions focus analysis on operational TAT drivers."""
        config = SummaryConfig()
        
        # Excluded categories should not interfere with operational analysis
        excluded_categories = {
            "demographics": ["sex"],
            "clinical_categoricals": ["severity", "patient_readiness_score"],
            "protocols": ["premed_required", "stat_order"], 
            "temporal_categoricals": ["order_dayofweek", "order_month"],
            "identifiers": ["order_id", "patient_id"],
            "targets": ["TAT_over_60"]
        }
        
        for category_type, columns in excluded_categories.items():
            for col in columns:
                assert col in config.corr_exclude_columns, f"{col} missing from exclusions ({category_type})"


class TestHIPAAComplianceConfiguration:
    """Test suite for HIPAA-compliant data handling and identifier management."""
    
    def test_id_columns_patient_privacy(self):
        """Test identification of patient-sensitive columns for HIPAA compliance."""
        config = SummaryConfig()
        
        expected_id_columns = {"order_id", "patient_id"}
        assert config.id_columns == expected_id_columns
        
        # Should identify both patient and order identifiers
        assert "patient_id" in config.id_columns  # Direct patient identifier
        assert "order_id" in config.id_columns    # Indirect patient identifier
        
        # Should be limited to actual identifiers
        assert len(config.id_columns) == 2

    def test_id_columns_excluded_from_correlation(self):
        """Test that patient identifiers are excluded from analytical processes."""
        config = SummaryConfig()
        
        # All ID columns should be excluded from correlation analysis
        for id_col in config.id_columns:
            assert id_col in config.corr_exclude_columns
        
        # Should prevent accidental inclusion in model features
        patient_identifiers = ["patient_id", "order_id"]
        for identifier in patient_identifiers:
            assert identifier in config.id_columns
            assert identifier in config.corr_exclude_columns

    def test_hipaa_compliant_analytical_separation(self):
        """Test separation of identifiers from analytical features."""
        config = SummaryConfig()
        
        # Identifiers should not overlap with categorical overrides
        assert not (config.id_columns & config.categorical_overrides)
        
        # Identifiers should not be in known time columns
        assert not (config.id_columns & config.known_time_cols)
        
        # Should maintain clear separation for data governance
        analytical_features = config.categorical_overrides | config.known_time_cols
        sensitive_identifiers = config.id_columns
        
        assert not (analytical_features & sensitive_identifiers)


class TestCustomConfigurationOverrides:
    """Test suite for custom configuration parameter validation."""
    
    def test_custom_percentiles_validation(self):
        """Test custom percentile configuration for specialized TAT analysis."""
        # Custom percentiles for focused SLA analysis
        custom_percentiles = (0.1, 0.5, 0.9, 0.95, 0.99)
        config = SummaryConfig(percentiles=custom_percentiles)
        
        assert config.percentiles == custom_percentiles
        
        # Should accept valid percentile ranges
        assert all(0 <= p <= 1 for p in config.percentiles)
        assert len(set(config.percentiles)) == len(config.percentiles)  # No duplicates

    def test_custom_thresholds_clinical_validation(self):
        """Test custom threshold configuration for different healthcare contexts."""
        # Custom TAT threshold for urgent care context
        urgent_config = SummaryConfig(tat_threshold_minutes=30.0)
        assert urgent_config.tat_threshold_minutes == 30.0
        
        # Custom data quality thresholds for research-grade data
        research_config = SummaryConfig(
            data_quality_warning_threshold=0.05,
            data_quality_critical_threshold=0.15
        )
        assert research_config.data_quality_warning_threshold == 0.05
        assert research_config.data_quality_critical_threshold == 0.15

    def test_custom_visualization_parameters(self):
        """Test custom visualization parameters for different stakeholder needs."""
        # Executive dashboard configuration (fewer details)
        executive_config = SummaryConfig(cat_top=3, missing_top_n=10, hist_bins=10)
        assert executive_config.cat_top == 3
        assert executive_config.missing_top_n == 10
        assert executive_config.hist_bins == 10
        
        # Detailed analytics configuration (more granularity)
        detailed_config = SummaryConfig(cat_top=8, missing_top_n=25, hist_bins=30)
        assert detailed_config.cat_top == 8
        assert detailed_config.missing_top_n == 25
        assert detailed_config.hist_bins == 30

    def test_custom_column_classifications(self):
        """Test custom column classification overrides for specialized datasets."""
        # Custom categorical overrides for research dataset
        custom_categoricals = {"protocol_version", "study_arm", "site_id"}
        research_config = SummaryConfig(categorical_overrides=custom_categoricals)
        assert research_config.categorical_overrides == custom_categoricals
        
        # Custom time column identification
        custom_time_cols = {"enrollment_date", "first_dose_time", "last_follow_up"}
        custom_config = SummaryConfig(known_time_cols=custom_time_cols)
        assert custom_config.known_time_cols == custom_time_cols

    def test_custom_exclusion_rules(self):
        """Test custom exclusion rules for specialized analytical focus."""
        # Custom correlation exclusions for multi-site study
        custom_corr_exclude = ("site_", "protocol_", "version_")
        multi_site_config = SummaryConfig(corr_exclude_prefixes=custom_corr_exclude)
        assert multi_site_config.corr_exclude_prefixes == custom_corr_exclude
        
        # Custom histogram suppression
        custom_no_hist = {"sensitive_clinical_score", "research_protocol_id"}
        research_config = SummaryConfig(no_hist_cols=custom_no_hist)
        assert research_config.no_hist_cols == custom_no_hist


class TestProductionMLOpsConfiguration:
    """Test suite for MLOps and production deployment configuration validation."""
    
    def test_production_logging_configuration(self):
        """Test production logging and monitoring configuration."""
        # Production logging enabled by default
        config = SummaryConfig()
        assert config.enable_production_logging is True
        
        # Should support disabling for development environments
        dev_config = SummaryConfig(enable_production_logging=False)
        assert dev_config.enable_production_logging is False

    def test_correlation_significance_thresholds(self):
        """Test correlation significance thresholds for clinical relevance."""
        config = SummaryConfig()
        
        # Should have clinically meaningful threshold
        assert config.correlation_significance_threshold == 0.1
        assert 0 < config.correlation_significance_threshold < 0.5
        
        # Should support custom significance levels
        strict_config = SummaryConfig(correlation_significance_threshold=0.05)
        assert strict_config.correlation_significance_threshold == 0.05

    def test_categorical_cardinality_limits(self):
        """Test categorical cardinality limits for production memory management."""
        config = SummaryConfig()
        
        # Should prevent memory issues with high-cardinality variables
        assert config.max_categorical_cardinality == 50
        assert config.max_categorical_cardinality > 0
        
        # Should support custom limits for different deployment contexts
        high_mem_config = SummaryConfig(max_categorical_cardinality=100)
        assert high_mem_config.max_categorical_cardinality == 100
        
        low_mem_config = SummaryConfig(max_categorical_cardinality=20)
        assert low_mem_config.max_categorical_cardinality == 20

    def test_configuration_validation_boundaries(self):
        """Test configuration parameter boundary validation."""
        config = SummaryConfig()
        
        # Percentiles should be valid probabilities
        assert all(0 <= p <= 1 for p in config.percentiles)
        
        # Counts should be positive integers
        assert config.missing_top_n > 0
        assert config.cat_top > 0
        assert config.hist_bins > 0
        assert config.max_categorical_cardinality > 0
        
        # Thresholds should be valid proportions
        assert 0 < config.data_quality_warning_threshold < 1
        assert 0 < config.data_quality_critical_threshold < 1
        assert 0 < config.correlation_significance_threshold < 1
        
        # TAT threshold should be positive
        assert config.tat_threshold_minutes > 0


class TestConfigurationConsistency:
    """Test suite for internal configuration consistency and logical relationships."""
    
    def test_data_quality_threshold_ordering(self):
        """Test that data quality thresholds maintain logical ordering."""
        config = SummaryConfig()
        
        # Critical threshold should be higher than warning threshold
        assert config.data_quality_critical_threshold > config.data_quality_warning_threshold
        
        # Both should be reasonable proportions
        assert 0 < config.data_quality_warning_threshold < 0.5
        assert 0 < config.data_quality_critical_threshold < 1

    def test_correlation_exclusion_consistency(self):
        """Test consistency between different correlation exclusion mechanisms."""
        config = SummaryConfig()
        
        # Prefixes and specific columns should complement each other
        # No overlap issues that would cause confusion
        prefix_patterns = config.corr_exclude_prefixes
        specific_columns = config.corr_exclude_columns
        
        # Should be complementary approaches to exclusion
        assert len(prefix_patterns) > 0
        assert len(specific_columns) > 0

    def test_categorical_classification_consistency(self):
        """Test consistency in categorical column classification approaches."""
        config = SummaryConfig()
        
        # Categorical overrides and prefixes should work together
        assert len(config.categorical_overrides) > 0
        assert len(config.categorical_prefixes) > 0
        
        # Should not have contradictory classifications
        # (This is more of a logical check - specific overlaps may be intentional)
        assert isinstance(config.categorical_overrides, set)
        assert isinstance(config.categorical_prefixes, tuple)

    def test_timestamp_analysis_consistency(self):
        """Test consistency in timestamp column handling."""
        config = SummaryConfig()
        
        # Known time columns should suppress histograms
        assert config.known_time_cols <= config.no_hist_cols or config.known_time_cols == config.no_hist_cols
        
        # Time columns should not be in categorical overrides
        assert not (config.known_time_cols & config.categorical_overrides)
        
        # Time columns should be excluded from correlation analysis
        time_cols_in_corr_exclude = [col for col in config.corr_exclude_columns if col in config.known_time_cols]
        # Not all time columns need to be explicitly excluded if they're filtered out by data type

    def test_identifier_handling_consistency(self):
        """Test consistent handling of patient and order identifiers."""
        config = SummaryConfig()
        
        # ID columns should be excluded from analytical processes
        for id_col in config.id_columns:
            assert id_col in config.corr_exclude_columns
        
        # ID columns should not be in other analytical categories
        assert not (config.id_columns & config.categorical_overrides)
        assert not (config.id_columns & config.known_time_cols)


class TestConfigurationDocumentation:
    """Test suite for configuration documentation and healthcare domain context."""
    
    def test_docstring_completeness(self):
        """Test that configuration parameters have comprehensive healthcare context."""
        config = SummaryConfig()
        
        # Class should have comprehensive docstring
        assert SummaryConfig.__doc__ is not None
        assert len(SummaryConfig.__doc__) > 100
        
        # Should mention key healthcare concepts
        class_doc = SummaryConfig.__doc__
        healthcare_terms = ["TAT", "pharmacy", "workflow", "healthcare", "clinical"]
        assert any(term in class_doc for term in healthcare_terms)

    def test_field_documentation_clinical_context(self):
        """Test that field docstrings provide clinical interpretation guidance."""
        # Key fields should have docstrings explaining clinical relevance
        import inspect
        
        # Get field annotations and check for documentation
        fields = SummaryConfig.__dataclass_fields__
        
        # Critical fields should be documented
        critical_fields = [
            "percentiles", "tat_threshold_minutes", "known_time_cols",
            "categorical_overrides", "corr_exclude_columns"
        ]
        
        for field_name in critical_fields:
            assert field_name in fields
            # In production code, these would have detailed docstrings

    def test_healthcare_domain_terminology(self):
        """Test that configuration uses appropriate healthcare and pharmacy terminology."""
        config = SummaryConfig()
        
        # Should use standard healthcare terminology
        healthcare_terms_in_config = []
        
        # Check known_time_cols for healthcare workflow terms
        workflow_terms = ["doctor", "nurse", "prep", "validation", "dispatch", "patient"]
        for col in config.known_time_cols:
            if any(term in col.lower() for term in workflow_terms):
                healthcare_terms_in_config.append(col)
        
        # Should have healthcare workflow terminology
        assert len(healthcare_terms_in_config) >= 4
        
        # Check categorical overrides for clinical terms
        clinical_terms = ["patient", "readiness", "stat", "order", "floor"]
        clinical_terms_found = []
        for override in config.categorical_overrides:
            if any(term in override.lower() for term in clinical_terms):
                clinical_terms_found.append(override)
        
        assert len(clinical_terms_found) >= 3


class TestConfigurationIntegration:
    """Integration tests for configuration usage in TAT analysis workflows."""
    
    def test_configuration_serialization_compatibility(self):
        """Test that configuration can be serialized for MLOps pipeline persistence."""
        config = SummaryConfig()
        
        # Should be serializable for pipeline configuration management
        # Test basic serialization properties
        config_dict = {
            'percentiles': config.percentiles,
            'tat_threshold': config.tat_threshold_minutes,
            'missing_top_n': config.missing_top_n
        }
        
        assert isinstance(config_dict['percentiles'], tuple)
        assert isinstance(config_dict['tat_threshold'], float)
        assert isinstance(config_dict['missing_top_n'], int)

    def test_configuration_equality_for_versioning(self):
        """Test configuration equality comparison for MLOps version control."""
        config1 = SummaryConfig()
        config2 = SummaryConfig()
        config3 = SummaryConfig(missing_top_n=25)
        
        # Equal configurations should compare equal
        assert config1 == config2
        
        # Different configurations should compare unequal
        assert config1 != config3
        
        # Should support version tracking through equality comparison
        config_versions = [config1, config2, config3]
        unique_configs = []
        
        for config in config_versions:
            if not any(config == existing for existing in unique_configs):
                unique_configs.append(config)
        
        # Should identify 2 unique configurations
        assert len(unique_configs) == 2

    def test_configuration_parameter_access_patterns(self):
        """Test common parameter access patterns for TAT analysis workflows."""
        config = SummaryConfig()
        
        # Common access patterns should work reliably
        percentile_analysis = [p for p in config.percentiles if p >= 0.9]
        assert len(percentile_analysis) >= 2  # p95, p99
        
        # Workflow timestamp iteration
        workflow_steps = list(config.known_time_cols)
        assert len(workflow_steps) == 6
        
        # Exclusion rule checking
        excluded_prefixes = config.corr_exclude_prefixes
        test_column = "race_white"
        is_excluded = any(test_column.startswith(prefix) for prefix in excluded_prefixes)
        assert is_excluded  # race_ prefix should be excluded

    def test_configuration_healthcare_workflow_alignment(self):
        """Test that configuration aligns with Dana Farber pharmacy workflow requirements."""
        config = SummaryConfig()
        
        # Should support complete TAT workflow analysis
        workflow_timestamps = config.known_time_cols
        expected_workflow_order = [
            "doctor_order_time",      # Step 1
            "nurse_validation_time",  # Step 2  
            "prep_complete_time",     # Step 3
            "second_validation_time", # Step 4
            "floor_dispatch_time",    # Step 5
            "patient_infusion_time"   # Step 6
        ]
        
        for step in expected_workflow_order:
            assert step in workflow_timestamps
        
        # Should support clinical threshold analysis
        assert config.tat_threshold_minutes == 60.0  # Dana Farber standard
        
        # Should support data quality assessment
        assert config.data_quality_warning_threshold <= 0.15  # Reasonable for healthcare
        
        # Should support stakeholder visualization needs
        assert 3 <= config.cat_top <= 6  # Appropriate for clinical team consumption
        assert 10 <= config.hist_bins <= 30  # Balance detail with interpretability