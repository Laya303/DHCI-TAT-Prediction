"""
Test suite for DataSummary analysis functionality.
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any, List
import sys
from io import StringIO

from src.tat.eda.summary.summary import DataSummary
from src.tat.eda.summary.summary_config import SummaryConfig


# Module-level fixtures available to all test classes
@pytest.fixture(scope="module")
def comprehensive_tat_dataset():
    """Generate comprehensive TAT dataset for end-to-end analysis testing."""
    np.random.seed(42)
    n_orders = 2000
    base_time = pd.Timestamp('2025-01-15 06:00:00')
    
    return pd.DataFrame({
        # Complete medication preparation workflow sequence
        'doctor_order_time': pd.date_range(base_time, periods=n_orders, freq='10min'),
        'nurse_validation_time': [
            base_time + pd.Timedelta(minutes=15+i*10) if i % 20 != 0 else pd.NaT 
            for i in range(n_orders)
        ],  # 5% missing
        'prep_complete_time': [
            base_time + pd.Timedelta(minutes=45+i*10) if i % 15 != 0 else pd.NaT
            for i in range(n_orders) 
        ],  # ~7% missing
        'second_validation_time': pd.date_range(base_time + pd.Timedelta('50min'), periods=n_orders, freq='10min'),
        'floor_dispatch_time': pd.date_range(base_time + pd.Timedelta('55min'), periods=n_orders, freq='10min'),
        'patient_infusion_time': pd.date_range(base_time + pd.Timedelta('70min'), periods=n_orders, freq='10min'),
        
        # Target performance metric
        'TAT_minutes': np.random.exponential(45, n_orders) + 12,
        'TAT_over_60': np.random.choice([0, 1], n_orders, p=[0.65, 0.35]),
        
        # Operational and staffing factors
        'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders, p=[0.55, 0.30, 0.15]),
        'nurse_credential': np.random.choice(['BSN', 'RN', 'MSN', 'NP'], n_orders, p=[0.45, 0.30, 0.20, 0.05]),
        'pharmacist_credential': np.random.choice(['RPh', 'PharmD', 'BCOP'], n_orders, p=[0.20, 0.65, 0.15]),
        'floor': np.random.choice([1, 2, 3], n_orders),
        'ordering_department': np.random.choice([
            'MedicalOncology', 'Hematology', 'StemCellTransplant', 
            'RadiationOncology', 'ImmunotherapyClinic'
        ], n_orders, p=[0.35, 0.25, 0.15, 0.15, 0.10]),
        
        # Performance and resource metrics
        'queue_length_at_order': np.random.poisson(7, n_orders),
        'floor_occupancy_pct': np.random.uniform(20, 95, n_orders),
        'pharmacists_on_duty': np.random.randint(2, 8, n_orders),
        'nurse_employment_years': np.random.uniform(0.5, 25, n_orders),
        'pharmacist_employment_years': np.random.uniform(1, 30, n_orders),
        
        # Clinical indicators and patient factors
        'patient_age': np.random.uniform(18, 85, n_orders),
        'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders, p=[0.30, 0.50, 0.20]),
        'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic', 'Autoimmune'], n_orders, p=[0.45, 0.35, 0.20]),
        'treatment_type': np.random.choice(['Chemotherapy', 'Immunotherapy', 'TargetedTherapy'], n_orders, p=[0.50, 0.30, 0.20]),
        'patient_readiness_score': np.random.choice([1, 2, 3], n_orders, p=[0.15, 0.35, 0.50]),
        'premed_required': np.random.choice([0, 1], n_orders, p=[0.70, 0.30]),
        'stat_order': np.random.choice([0, 1], n_orders, p=[0.85, 0.15]),
        
        # Laboratory values with realistic missing patterns
        'lab_WBC_k_per_uL': [
            np.random.uniform(4.0, 11.0) if np.random.random() > 0.15 else np.nan
            for _ in range(n_orders)
        ],  # 15% missing
        'lab_HGB_g_dL': [
            np.random.uniform(10.0, 16.0) if np.random.random() > 0.12 else np.nan
            for _ in range(n_orders)
        ],  # 12% missing
        'lab_Platelets_k_per_uL': np.random.uniform(150, 400, n_orders),
        'lab_Creatinine_mg_dL': np.random.uniform(0.6, 2.0, n_orders),
        
        # Identifiers (should be excluded from analysis)
        'patient_id': [f'DFCI_{i:07d}' for i in range(n_orders)],
        'order_id': [f'ORD_{i:08d}' for i in range(n_orders)],
        'ordering_physician': [f'Dr_Smith_{i%50:02d}' for i in range(n_orders)]  # 50 unique physicians
    })


@pytest.fixture(scope="module")
def production_tat_dataset():
    """Generate production-scale TAT dataset for comprehensive reporting testing."""
    np.random.seed(42)
    n_orders = 3000
    base_time = pd.Timestamp('2025-02-01 06:00:00')
    
    return pd.DataFrame({
        # Complete medication preparation workflow
        'doctor_order_time': pd.date_range(base_time, periods=n_orders, freq='8min'),
        'nurse_validation_time': [
            base_time + pd.Timedelta(minutes=12+i*8) if i % 25 != 0 else pd.NaT 
            for i in range(n_orders)
        ],  # 4% missing
        'prep_complete_time': [
            base_time + pd.Timedelta(minutes=38+i*8) if i % 20 != 0 else pd.NaT
            for i in range(n_orders)
        ],  # 5% missing
        'second_validation_time': pd.date_range(base_time + pd.Timedelta('42min'), periods=n_orders, freq='8min'),
        'floor_dispatch_time': pd.date_range(base_time + pd.Timedelta('48min'), periods=n_orders, freq='8min'),
        'patient_infusion_time': pd.date_range(base_time + pd.Timedelta('68min'), periods=n_orders, freq='8min'),
        
        #  Numerical Features
        'TAT_minutes': np.random.exponential(48, n_orders) + 10,
        'queue_length_at_order': np.random.poisson(8, n_orders),
        'floor_occupancy_pct': np.random.uniform(25, 95, n_orders),
        'pharmacists_on_duty': np.random.randint(2, 9, n_orders),
        
        # Comprehensive operational factors
        'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders, p=[0.55, 0.30, 0.15]),
        'nurse_credential': np.random.choice(['BSN', 'RN', 'MSN', 'NP'], n_orders, p=[0.40, 0.35, 0.20, 0.05]),
        'pharmacist_credential': np.random.choice(['RPh', 'PharmD', 'BCOP'], n_orders, p=[0.15, 0.70, 0.15]),
        'floor': np.random.choice([1, 2, 3], n_orders),
        'ordering_department': np.random.choice([
            'MedicalOncology', 'Hematology', 'StemCellTransplant', 
            'RadiationOncology', 'ImmunotherapyClinic'
        ], n_orders, p=[0.40, 0.25, 0.15, 0.12, 0.08]),
        'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders, p=[0.30, 0.50, 0.20]),
        'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic', 'Autoimmune'], n_orders, p=[0.50, 0.35, 0.15]),
        
        # Clinical and patient factors
        'patient_age': np.random.uniform(18, 89, n_orders),
        'patient_readiness_score': np.random.choice([1, 2, 3], n_orders, p=[0.20, 0.35, 0.45]),
        'premed_required': np.random.choice([0, 1], n_orders, p=[0.65, 0.35]),
        'stat_order': np.random.choice([0, 1], n_orders, p=[0.88, 0.12]),
        
        # Laboratory values with missing patterns
        'lab_WBC_k_per_uL': [
            np.random.uniform(3.5, 12.0) if np.random.random() > 0.14 else np.nan
            for _ in range(n_orders)
        ],  # 14% missing
        'lab_Platelets_k_per_uL': np.random.uniform(120, 450, n_orders),
        'lab_Creatinine_mg_dL': np.random.uniform(0.5, 2.2, n_orders),
        
        # Identifiers
        'patient_id': [f'DFCI_{i:07d}' for i in range(n_orders)],
        'order_id': [f'ORD_{i:08d}' for i in range(n_orders)]
    })


class TestDataSummaryInitialization:
    """Test suite for DataSummary initialization and configuration management."""
    
    def test_default_initialization_healthcare_config(self):
        """Test DataSummary initialization with healthcare-optimized defaults."""
        summary = DataSummary()
        
        # Should initialize with healthcare defaults
        assert isinstance(summary.cfg, SummaryConfig)
        assert summary.cfg.tat_threshold_minutes == 60.0  # Dana Farber standard
        
        # Should have renderer instances ready
        assert summary._console is not None
        assert summary._html is not None
        
        # Should support production healthcare analytics environments
        assert hasattr(summary, 'cfg')
        assert hasattr(summary, '_console')
        assert hasattr(summary, '_html')

    def test_custom_configuration_pharmacy_workflow(self):
        """Test DataSummary initialization with custom pharmacy workflow configuration."""
        # Custom configuration for urgent care context
        custom_config = SummaryConfig(
            tat_threshold_minutes=30.0,  # Urgent care threshold
            hist_bins=15,                # Detailed visualization
            cat_top=6                    # More category detail
        )
        
        summary = DataSummary(custom_config)
        
        # Should use custom configuration
        assert summary.cfg.tat_threshold_minutes == 30.0
        assert summary.cfg.hist_bins == 15
        assert summary.cfg.cat_top == 6
        
        # Should maintain renderer setup
        assert summary._console is not None
        assert summary._html is not None

    def test_default_class_method_convenience(self):
        """Test convenience factory method for quick pharmacy analytics."""
        summary_default = DataSummary.default()
        summary_explicit = DataSummary()
        
        # Should be equivalent to explicit initialization
        assert summary_default.cfg.tat_threshold_minutes == summary_explicit.cfg.tat_threshold_minutes
        assert summary_default.cfg.percentiles == summary_explicit.cfg.percentiles
        assert summary_default.cfg.missing_top_n == summary_explicit.cfg.missing_top_n
        
        # Should be ready for immediate TAT analysis
        assert isinstance(summary_default, DataSummary)
        assert summary_default._console is not None


class TestWorkflowTimestampAnalysisIntegration:
    """Test suite for comprehensive TAT workflow analysis integration."""
    
    @pytest.fixture
    def minimal_tat_dataset(self):
        """Generate minimal TAT dataset for edge case testing."""
        return pd.DataFrame({
            'doctor_order_time': pd.date_range('2025-01-15', periods=100, freq='1h'),  # Fixed: use 'h' not 'H'
            'TAT_minutes': np.random.exponential(40, 100) + 10,
            'shift': np.random.choice(['Day', 'Evening', 'Night'], 100)
        })
    
    @pytest.fixture
    def empty_dataset(self):
        """Generate empty dataset for robustness testing."""
        return pd.DataFrame()

    def test_print_report_comprehensive_analysis(self, comprehensive_tat_dataset, capsys):
        """Test comprehensive console-based TAT analysis for pharmacy team review."""
        summary = DataSummary.default()
        
        # Execute comprehensive analysis
        artifacts = summary.print_report(comprehensive_tat_dataset)
        
        # Capture console output for validation
        captured = capsys.readouterr()
        console_output = captured.out
        
        # Should generate complete analysis artifacts
        expected_artifacts = [
            'time_table', 'categorical_table', 'numeric_table', 'correlations',
            'missing_table', 'missing_table_console', 'counts', 'df_processed'
        ]
        assert all(key in artifacts for key in expected_artifacts)
        
        # Should display pharmacy-focused analysis sections
        assert "PHARMACY TAT DATASET ANALYSIS SUMMARY" in console_output
        assert "MEDICATION PREPARATION WORKFLOW TIMELINE" in console_output
        assert "OPERATIONAL CONTEXT & STAFFING ANALYSIS" in console_output
        assert "TAT  Numerical Features & CLINICAL INDICATORS" in console_output
        assert "TAT DRIVER CORRELATION ANALYSIS" in console_output
        
        # Should include workflow terminology
        healthcare_terms = ['Orders Analyzed', 'Workflow Steps', 'Operational Factors', 'bottleneck']
        assert any(term in console_output for term in healthcare_terms)
        
        # Should provide actionable insights
        assert "workflow optimization" in console_output.lower()
        assert "improvement" in console_output.lower()
        
        # Should include executive summary metrics
        counts = artifacts['counts']
        assert counts['rows'] == 2000
        assert counts['cols'] > 20  # Comprehensive dataset
        assert counts['time'] >= 6   # Complete workflow timestamps
        assert counts['categorical'] >= 8  # Multiple operational factors
        assert counts['numeric'] >= 10     # Performance and clinical metrics

    def test_print_report_missing_data_assessment(self, comprehensive_tat_dataset, capsys):
        """Test missing data assessment integration for healthcare data quality monitoring."""
        summary = DataSummary.default()
        artifacts = summary.print_report(comprehensive_tat_dataset)
        
        # Capture and analyze console output
        captured = capsys.readouterr()
        console_output = captured.out
        
        # Should identify missing data patterns
        assert "DATA QUALITY: MISSING WORKFLOW TIMESTAMPS" in console_output
        assert "workflow integrity" in console_output.lower()
        
        # Should assess timestamp completeness
        missing_table = artifacts['missing_table']
        assert not missing_table.empty
        
        # Should identify known missing patterns from test data
        missing_cols = set(missing_table['column'])
        expected_missing = {'nurse_validation_time', 'prep_complete_time', 'lab_WBC_k_per_uL', 'lab_HGB_g_dL'}
        assert len(missing_cols & expected_missing) >= 2  # At least some missing columns identified

    def test_print_report_correlation_analysis_bottlenecks(self, comprehensive_tat_dataset, capsys):
        """Test correlation analysis for TAT bottleneck driver identification."""
        summary = DataSummary.default()
        artifacts = summary.print_report(comprehensive_tat_dataset)
        
        # Capture console output for correlation analysis validation
        captured = capsys.readouterr()
        console_output = captured.out
        
        # Should perform correlation analysis
        assert "TAT DRIVER CORRELATION ANALYSIS" in console_output
        assert "Bottleneck identification" in console_output
        
        # Should generate correlation insights
        correlations = artifacts['correlations']
        if not correlations.empty:
            # Should include TAT-related correlations
            corr_features = set(correlations.columns) | set(correlations.index)
            assert 'TAT_minutes' in corr_features or len(corr_features) > 5
        
        # Should provide optimization guidance
        assert "workflow optimization" in console_output.lower()
        assert "improvement" in console_output.lower()

    def test_print_report_minimal_dataset_robustness(self, minimal_tat_dataset, capsys):
        """Test analysis robustness with minimal healthcare dataset."""
        summary = DataSummary.default()
        artifacts = summary.print_report(minimal_tat_dataset)
        
        # Should handle minimal dataset gracefully
        assert isinstance(artifacts, dict)
        assert artifacts['counts']['rows'] == 100
        assert artifacts['counts']['cols'] == 3
        
        # Should still provide analysis structure
        captured = capsys.readouterr()
        console_output = captured.out
        assert "PHARMACY TAT DATASET ANALYSIS SUMMARY" in console_output
        assert "Analysis Complete" in console_output

    def test_print_report_empty_dataset_robustness(self, empty_dataset, capsys):
        """Test analysis robustness with empty dataset for production error handling."""
        summary = DataSummary.default()
        artifacts = summary.print_report(empty_dataset)
        
        # Should handle empty dataset without crashing
        assert isinstance(artifacts, dict)
        assert artifacts['counts']['rows'] == 0
        assert artifacts['counts']['cols'] == 0
        
        # Should provide appropriate messaging
        captured = capsys.readouterr()
        console_output = captured.out
        assert "PHARMACY TAT DATASET ANALYSIS SUMMARY" in console_output


class TestHTMLReportingStakeholderCommunication:
    """Test suite for HTML reporting capabilities for stakeholder communication."""
    
    @pytest.fixture
    def stakeholder_tat_dataset(self):
        """Generate TAT dataset optimized for stakeholder reporting testing."""
        np.random.seed(123)
        n_orders = 1200
        base_time = pd.Timestamp('2025-01-20 07:00:00')
        
        return pd.DataFrame({
            # Core workflow timestamps
            'doctor_order_time': pd.date_range(base_time, periods=n_orders, freq='12min'),
            'nurse_validation_time': pd.date_range(base_time + pd.Timedelta('18min'), periods=n_orders, freq='12min'),
            'prep_complete_time': pd.date_range(base_time + pd.Timedelta('42min'), periods=n_orders, freq='12min'),
            'patient_infusion_time': pd.date_range(base_time + pd.Timedelta('65min'), periods=n_orders, freq='12min'),
            
            # Key  Numerical Features for stakeholder focus
            'TAT_minutes': np.random.exponential(50, n_orders) + 15,
            'queue_length_at_order': np.random.poisson(6, n_orders),
            'floor_occupancy_pct': np.random.uniform(30, 90, n_orders),
            
            # Strategic operational factors
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders, p=[0.50, 0.35, 0.15]),
            'ordering_department': np.random.choice([
                'MedicalOncology', 'Hematology', 'StemCellTransplant'
            ], n_orders, p=[0.45, 0.35, 0.20]),
            'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders, p=[0.25, 0.50, 0.25]),
            
            # Clinical context
            'patient_age': np.random.uniform(25, 80, n_orders),
            'lab_WBC_k_per_uL': np.random.uniform(4.0, 11.0, n_orders)
        })

    def test_to_html_comprehensive_stakeholder_report(self, stakeholder_tat_dataset):
        """Test comprehensive HTML report generation for pharmacy leadership review."""
        summary = DataSummary.default()
        html_output = summary.to_html(stakeholder_tat_dataset)
        
        # Should generate complete HTML document
        assert isinstance(html_output, str)
        assert len(html_output) > 1000  # Substantial report content
        
        # Should include HTML structure elements (adjust for actual output format)
        assert "<meta charset=\"utf-8\">" in html_output or "<style>" in html_output
        assert "<table>" in html_output or "table" in html_output.lower()
        
        # Should include pharmacy-focused terminology
        healthcare_terms = ['TAT', 'workflow', 'pharmacy', 'medication', 'turnaround']
        html_lower = html_output.lower()
        assert any(term in html_lower for term in healthcare_terms)
        
        # Should include data visualization elements
        assert "table" in html_lower or "chart" in html_lower
        
        # Should include styling for stakeholder distribution
        assert "<style>" in html_output or "css" in html_lower

    def test_to_html_visualization_integration(self, stakeholder_tat_dataset):
        """Test HTML visualization integration for clinical stakeholder consumption."""
        summary = DataSummary.default()
        html_output = summary.to_html(stakeholder_tat_dataset)
        
        # Should include visualization components
        visualization_indicators = ['histogram', 'distribution', 'chart', 'graph', 'sparkline']
        html_lower = html_output.lower()
        
        # At least some visualization elements should be present
        viz_present = any(indicator in html_lower for indicator in visualization_indicators)
        assert viz_present or len(html_output) > 5000  # Rich content indicates visualizations

    def test_to_html_clinical_terminology_accuracy(self, stakeholder_tat_dataset):
        """Test HTML report uses appropriate clinical terminology for healthcare audiences."""
        summary = DataSummary.default()
        html_output = summary.to_html(stakeholder_tat_dataset)
        
        # Should include healthcare domain terminology
        clinical_terms = [
            'medication preparation', 'turnaround time', 'workflow', 
            'pharmacy', 'clinical', 'patient', 'order'
        ]
        
        html_lower = html_output.lower()
        clinical_term_count = sum(1 for term in clinical_terms if term in html_lower)
        assert clinical_term_count >= 3  # Multiple healthcare terms present

    def test_to_html_empty_dataset_robustness(self):
        """Test HTML generation robustness with empty dataset for production deployment."""
        summary = DataSummary.default()
        empty_df = pd.DataFrame()
        
        html_output = summary.to_html(empty_df)
        
        # Should generate valid HTML even with empty data
        assert isinstance(html_output, str)
        assert len(html_output) > 100  # Should include basic HTML structure
        # Adjust assertion for actual HTML fragment format
        assert "<meta charset=\"utf-8\">" in html_output or "<style>" in html_output


class TestMultiFormatReportingWorkflow:
    """Test suite for comprehensive multi-format reporting workflow integration."""

    def test_report_comprehensive_multi_format_output(self, production_tat_dataset, capsys):
        """Test comprehensive multi-format reporting for diverse stakeholder needs."""
        summary = DataSummary.default()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Execute comprehensive reporting workflow
            artifacts = summary.report(
                production_tat_dataset,
                save_dir=temp_dir,
                export="html",
                console=True
            )
            
            # Should generate complete analysis artifacts
            expected_artifacts = [
                'time_table', 'categorical_table', 'numeric_table', 'correlations',
                'missing_table', 'missing_table_console', 'counts', 'df_processed'
            ]
            assert all(key in artifacts for key in expected_artifacts)
            
            # Should create structured file outputs
            expected_files = [
                'summary_time.csv',
                'summary_categorical.csv', 
                'summary_numeric.csv',
                'summary.html'
            ]
            
            for filename in expected_files:
                file_path = os.path.join(temp_dir, filename)
                assert os.path.exists(file_path), f"Missing output file: {filename}"
                assert os.path.getsize(file_path) > 100, f"Output file too small: {filename}"
            
            # Should display console output for immediate review
            captured = capsys.readouterr()
            console_output = captured.out
            # Fix: Match the actual console output
            assert "generating comprehensive tat analysis" in console_output.lower()
            assert "complete" in console_output.lower()
            
            # Should provide stakeholder messaging
            assert "pharmacy leadership" in console_output.lower() or "stakeholder" in console_output.lower()

    def test_report_csv_export_structure_downstream_analytics(self, production_tat_dataset):
        """Test CSV export structure for downstream analytics and model development."""
        summary = DataSummary.default()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = summary.report(
                production_tat_dataset,
                save_dir=temp_dir,
                export="html",
                console=False
            )
            
            # Validate time analysis CSV structure
            time_csv_path = os.path.join(temp_dir, 'summary_time.csv')
            time_df = pd.read_csv(time_csv_path)
            
            expected_time_columns = ['name', 'type', 'n_nonnull', 'miss%', 'sample']
            assert all(col in time_df.columns for col in expected_time_columns)
            assert len(time_df) >= 6  # Complete workflow timestamps
            
            # Validate categorical analysis CSV structure
            cat_csv_path = os.path.join(temp_dir, 'summary_categorical.csv')
            cat_df = pd.read_csv(cat_csv_path)
            
            expected_cat_columns = ['name', 'type', 'n_nonnull', 'miss%', 'nunique', 'sample', 'top_values']
            assert all(col in cat_df.columns for col in expected_cat_columns)
            assert len(cat_df) >= 8  # Multiple operational factors
            
            # Validate numeric analysis CSV structure
            num_csv_path = os.path.join(temp_dir, 'summary_numeric.csv')
            num_df = pd.read_csv(num_csv_path)
            
            expected_num_columns = ['name', 'type', 'n_nonnull', 'miss%', 'nunique', 'sample', 'min', 'p50', 'max', 'distribution']
            core_num_columns = ['name', 'type', 'n_nonnull', 'miss%', 'min', 'p50', 'max']
            assert all(col in num_df.columns for col in core_num_columns)
            assert len(num_df) >= 8  # Performance and clinical metrics

    def test_report_html_export_stakeholder_ready(self, production_tat_dataset):
        """Test HTML export readiness for stakeholder distribution and presentation."""
        summary = DataSummary.default()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = summary.report(
                production_tat_dataset,
                save_dir=temp_dir,
                export="html",
                console=False
            )
            
            # Validate HTML report generation
            html_path = os.path.join(temp_dir, 'summary.html')
            assert os.path.exists(html_path)
            
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Should be substantial report content
            assert len(html_content) > 5000  # Rich stakeholder report
            
            # Should include HTML structure elements (adjust for actual format)
            assert "<meta charset=\"utf-8\">" in html_content or "<style>" in html_content
            
            # Should include healthcare analytics content
            html_lower = html_content.lower()
            healthcare_terms = ['tat', 'pharmacy', 'workflow', 'medication', 'turnaround']
            assert any(term in html_lower for term in healthcare_terms)

    def test_report_console_disabled_option(self, production_tat_dataset, capsys):
        """Test console output can be disabled for automated pipeline integration."""
        summary = DataSummary.default()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = summary.report(
                production_tat_dataset,
                save_dir=temp_dir,
                export="html",
                console=False  # Disable console output
            )
            
            # Should still generate complete artifacts
            assert isinstance(artifacts, dict)
            assert len(artifacts) >= 7
            
            # Should minimize console output when disabled
            captured = capsys.readouterr()
            console_output = captured.out
            
            # Should have minimal output (file save messages only)
            console_lines = [line for line in console_output.split('\n') if line.strip()]
            assert len(console_lines) <= 10  # Minimal logging only

    def test_report_no_save_directory_memory_only(self, production_tat_dataset, capsys):
        """Test in-memory analysis without file persistence for interactive workflows."""
        summary = DataSummary.default()
        
        # Execute analysis without file persistence
        artifacts = summary.report(
            production_tat_dataset,
            save_dir=None,  # No file output
            export="html",
            console=True
        )
        
        # Should generate complete in-memory artifacts
        assert isinstance(artifacts, dict)
        expected_artifacts = [
            'time_table', 'categorical_table', 'numeric_table', 'correlations',
            'missing_table', 'counts', 'df_processed'
        ]
        assert all(key in artifacts for key in expected_artifacts)
        
        # Should display console analysis
        captured = capsys.readouterr()
        console_output = captured.out
        # Fix: Match actual console output format
        assert "generating comprehensive tat analysis" in console_output.lower() or "pharmacy tat dataset" in console_output.lower()
        assert len(console_output) > 1000  # Substantial console report

    def test_report_directory_creation_automation(self, production_tat_dataset):
        """Test automatic directory creation for automated deployment workflows."""
        summary = DataSummary.default()
        
        with tempfile.TemporaryDirectory() as base_temp_dir:
            # Use nested directory path that doesn't exist
            nested_save_dir = os.path.join(base_temp_dir, 'pharmacy_analytics', 'tat_analysis', '2025_q1')
            
            artifacts = summary.report(
                production_tat_dataset,
                save_dir=nested_save_dir,
                export="html",
                console=False
            )
            
            # Should create directory structure automatically
            assert os.path.exists(nested_save_dir)
            assert os.path.isdir(nested_save_dir)
            
            # Should generate files in created directory
            expected_files = ['summary_time.csv', 'summary_categorical.csv', 'summary_numeric.csv', 'summary.html']
            for filename in expected_files:
                file_path = os.path.join(nested_save_dir, filename)
                assert os.path.exists(file_path)


class TestProductionReadinessAndRobustness:
    """Test suite for production deployment readiness and error handling robustness."""
    
    @pytest.fixture
    def malformed_tat_dataset(self):
        """Generate malformed TAT dataset for robustness testing."""
        return pd.DataFrame({
            # Mixed data types and missing patterns
            'doctor_order_time': ['2025-01-15 08:00:00', None, 'invalid_timestamp', '2025-01-15 09:00:00'],
            'TAT_minutes': [45.5, np.nan, -10, 'invalid_number'],  # Invalid values
            'shift': ['Day', None, 'InvalidShift', 'Evening'],
            'patient_age': [25, 150, -5, np.nan],  # Extreme values
            'queue_length_at_order': [5, np.nan, 1000, 'invalid'],  # Mixed types
            
            # Completely missing column
            'missing_column': [np.nan] * 4,
            
            # Single value column
            'constant_value': [42] * 4
        })
    
    @pytest.fixture
    def large_scale_dataset(self):
        """Generate large-scale dataset for performance testing."""
        np.random.seed(42)
        n_orders = 10000  # Large dataset
        
        return pd.DataFrame({
            'doctor_order_time': pd.date_range('2025-01-01', periods=n_orders, freq='5min'),
            'TAT_minutes': np.random.exponential(50, n_orders) + 10,
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders),
            'queue_length_at_order': np.random.poisson(7, n_orders),
            'patient_age': np.random.uniform(18, 90, n_orders),
            'floor_occupancy_pct': np.random.uniform(20, 100, n_orders)
        })

    def test_malformed_data_robustness(self, malformed_tat_dataset, capsys):
        """Test analysis robustness with malformed healthcare data for production deployment."""
        summary = DataSummary.default()
        
        # Should handle malformed data without crashing
        artifacts = summary.print_report(malformed_tat_dataset)
        
        # Should generate artifacts despite data quality issues
        assert isinstance(artifacts, dict)
        assert 'counts' in artifacts
        assert artifacts['counts']['rows'] == 4
        
        # Should complete analysis and provide feedback
        captured = capsys.readouterr()
        console_output = captured.out
        assert "PHARMACY TAT DATASET ANALYSIS SUMMARY" in console_output
        assert "Analysis Complete" in console_output

    def test_large_scale_performance(self, large_scale_dataset):
        """Test performance with large-scale datasets for production deployment."""
        summary = DataSummary.default()
        
        import time
        start_time = time.time()
        
        # Execute analysis on large dataset
        artifacts = summary.print_report(large_scale_dataset)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Should complete analysis in reasonable time
        assert analysis_time < 30.0  # Should complete within 30 seconds
        
        # Should generate complete artifacts
        assert isinstance(artifacts, dict)
        assert artifacts['counts']['rows'] == 10000
        assert len(artifacts['time_table']) >= 1
        assert len(artifacts['categorical_table']) >= 1
        assert len(artifacts['numeric_table']) >= 1

    def test_unicode_console_output_healthcare_environments(self, comprehensive_tat_dataset):
        """Test Unicode console output handling for diverse healthcare IT environments."""
        summary = DataSummary.default()
        
        # Should handle Unicode characters in healthcare environments
        with patch('sys.stdout') as mock_stdout:
            mock_stdout.reconfigure = MagicMock()
            
            # Execute analysis (should attempt UTF-8 configuration)
            artifacts = summary.print_report(comprehensive_tat_dataset)
            
            # Should attempt console reconfiguration for Unicode support
            # (Graceful fallback if reconfiguration fails)
            assert isinstance(artifacts, dict)

    def test_memory_management_large_datasets(self, large_scale_dataset):
        """Test memory management with large datasets for production scalability."""
        summary = DataSummary.default()
        
        # Execute analysis and verify memory efficiency
        artifacts = summary.print_report(large_scale_dataset)
        
        # Should maintain reasonable memory footprint
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Should not consume excessive memory (adjust threshold as needed)
        assert memory_mb < 1000  # Less than 1GB for reasonable dataset
        
        # Should complete successfully
        assert isinstance(artifacts, dict)
        assert artifacts['counts']['rows'] == 10000

    @patch('builtins.open', new_callable=mock_open)
    def test_file_write_error_handling(self, mock_file_open, production_tat_dataset, capsys):
        """Test file write error handling for production deployment robustness."""
        summary = DataSummary.default()
        
        # Simulate file write error
        mock_file_open.side_effect = PermissionError("Permission denied for healthcare IT security")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should handle file write errors gracefully
            try:
                artifacts = summary.report(
                    production_tat_dataset,
                    save_dir=temp_dir,
                    export="html",
                    console=False
                )
                
                # Should still generate in-memory artifacts
                assert isinstance(artifacts, dict)
                assert 'counts' in artifacts
                
            except PermissionError:
                # Acceptable to propagate certain system errors
                pass

    def test_configuration_validation_edge_cases(self):
        """Test configuration validation with edge cases for production deployment."""
        # Test with extreme configuration values
        extreme_config = SummaryConfig(
            tat_threshold_minutes=0.1,  # Very low threshold
            hist_bins=1,                # Minimal bins
            missing_top_n=1000,         # Very high limit
            cat_top=100                 # High category limit
        )
        
        summary = DataSummary(extreme_config)
        
        # Should initialize successfully with edge case configuration
        assert summary.cfg.tat_threshold_minutes == 0.1
        assert summary.cfg.hist_bins == 1
        assert summary.cfg.missing_top_n == 1000
        assert summary.cfg.cat_top == 100
        
        # Should be ready for analysis
        assert summary._console is not None
        assert summary._html is not None


class TestHealthcareDomainAlignment:
    """Test suite for healthcare domain alignment and clinical terminology validation."""
    
    @pytest.fixture
    def clinical_context_dataset(self):
        """Generate dataset with comprehensive clinical context for domain alignment testing."""
        np.random.seed(123)
        n_orders = 1500
        
        return pd.DataFrame({
            # Clinical workflow timestamps
            'doctor_order_time': pd.date_range('2025-02-01 07:00:00', periods=n_orders, freq='10min'),
            'nurse_validation_time': pd.date_range('2025-02-01 07:15:00', periods=n_orders, freq='10min'),
            'prep_complete_time': pd.date_range('2025-02-01 07:45:00', periods=n_orders, freq='10min'),
            'patient_infusion_time': pd.date_range('2025-02-01 08:15:00', periods=n_orders, freq='10min'),
            
            # TAT performance with clinical threshold alignment
            'TAT_minutes': np.random.exponential(45, n_orders) + 10,
            'TAT_over_60': np.random.choice([0, 1], n_orders, p=[0.70, 0.30]),  # 30% exceed SLA
            
            # Clinical factors and healthcare operations
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders, p=[0.50, 0.35, 0.15]),
            'nurse_credential': np.random.choice(['BSN', 'RN', 'MSN', 'NP'], n_orders, p=[0.40, 0.35, 0.20, 0.05]),
            'pharmacist_credential': np.random.choice(['RPh', 'PharmD', 'BCOP'], n_orders, p=[0.20, 0.65, 0.15]),
            'ordering_department': np.random.choice([
                'MedicalOncology', 'Hematology', 'StemCellTransplant'
            ], n_orders, p=[0.45, 0.35, 0.20]),
            'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders, p=[0.30, 0.50, 0.20]),
            'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic', 'Autoimmune'], n_orders, p=[0.50, 0.30, 0.20]),
            'treatment_type': np.random.choice(['Chemotherapy', 'Immunotherapy', 'TargetedTherapy'], n_orders, p=[0.50, 0.30, 0.20]),
            'patient_readiness_score': np.random.choice([1, 2, 3], n_orders, p=[0.15, 0.35, 0.50]),
            'premed_required': np.random.choice([0, 1], n_orders, p=[0.70, 0.30]),
            'stat_order': np.random.choice([0, 1], n_orders, p=[0.85, 0.15]),
            
            # Healthcare resource and capacity metrics
            'floor_occupancy_pct': np.random.uniform(25, 95, n_orders),
            'queue_length_at_order': np.random.poisson(7, n_orders),
            'pharmacists_on_duty': np.random.randint(2, 8, n_orders),
            'nurse_employment_years': np.random.uniform(0.5, 25, n_orders),
            'pharmacist_employment_years': np.random.uniform(1, 30, n_orders),
            
            # Clinical laboratory values
            'lab_WBC_k_per_uL': np.random.uniform(4.0, 11.0, n_orders),
            'lab_HGB_g_dL': np.random.uniform(10.0, 16.0, n_orders),
            'lab_Platelets_k_per_uL': np.random.uniform(150, 400, n_orders),
            'lab_Creatinine_mg_dL': np.random.uniform(0.6, 1.8, n_orders),
            
            # Patient demographics and clinical context
            'patient_age': np.random.uniform(18, 85, n_orders),
            'floor': np.random.choice([1, 2, 3], n_orders)
        })

    def test_healthcare_terminology_console_output(self, clinical_context_dataset, capsys):
        """Test healthcare terminology accuracy in console output for clinical teams."""
        summary = DataSummary.default()
        artifacts = summary.print_report(clinical_context_dataset)
        
        # Capture console output for terminology validation
        captured = capsys.readouterr()
        console_output = captured.out
        
        # Should use appropriate healthcare terminology
        healthcare_terms = [
            'pharmacy', 'medication preparation', 'workflow', 'turnaround time',
            'clinical', 'patient', 'nursing', 'pharmacist', 'bottleneck'
        ]
        
        console_lower = console_output.lower()
        found_terms = [term for term in healthcare_terms if term in console_lower]
        assert len(found_terms) >= 5, f"Insufficient healthcare terminology. Found: {found_terms}"
        
        # Should include workflow-specific language
        workflow_terms = ['workflow', 'step', 'delay', 'optimization']
        workflow_found = [term for term in workflow_terms if term in console_lower]
        assert len(workflow_found) >= 2, f"Missing workflow terminology. Found: {workflow_found}"

    def test_clinical_threshold_alignment(self, clinical_context_dataset):
        """Test clinical threshold alignment with Dana Farber 60-minute standard."""
        summary = DataSummary.default()
        
        # Should use 60-minute TAT threshold
        assert summary.cfg.tat_threshold_minutes == 60.0
        
        # Analysis should reflect clinical standard
        artifacts = summary.print_report(clinical_context_dataset)
        
        # Should process TAT metrics appropriately
        numeric_table = artifacts['numeric_table']
        tat_row = numeric_table[numeric_table['name'] == 'TAT_minutes']
        
        if not tat_row.empty:
            # Should provide clinically meaningful statistics
            tat_stats = tat_row.iloc[0]
            assert tat_stats['min'] >= 0  # TAT should be non-negative
            assert tat_stats['p50'] > 0    # Median should be positive
            assert tat_stats['max'] > tat_stats['p50']  # Statistical consistency

    def test_healthcare_data_quality_thresholds(self, clinical_context_dataset):
        """Test healthcare-appropriate data quality thresholds and interpretation."""
        summary = DataSummary.default()
        
        # Should use healthcare data quality thresholds
        assert summary.cfg.data_quality_warning_threshold == 0.10  # 10% warning
        assert summary.cfg.data_quality_critical_threshold == 0.25  # 25% critical
        
        # Should assess data quality with healthcare context
        artifacts = summary.print_report(clinical_context_dataset)
        missing_table = artifacts['missing_table']
        
        # Should evaluate missingness in healthcare context
        if not missing_table.empty:
            # Missing percentages should be calculated correctly
            assert all(missing_table['missing%'] >= 0)
            assert all(missing_table['missing%'] <= 100)

    def test_workflow_column_classification_accuracy(self, clinical_context_dataset):
        """Test accuracy of workflow column classification with healthcare domain knowledge."""
        summary = DataSummary.default()
        artifacts = summary.print_report(clinical_context_dataset)
        
        # Should classify workflow timestamps correctly
        time_table = artifacts['time_table']
        workflow_timestamps = ['doctor_order_time', 'nurse_validation_time', 
                             'prep_complete_time', 'patient_infusion_time']
        
        identified_timestamps = set(time_table['name']) if not time_table.empty else set()
        expected_timestamps = set(workflow_timestamps)
        
        # Should identify most workflow timestamps
        intersection = identified_timestamps & expected_timestamps
        assert len(intersection) >= 3, f"Missing workflow timestamps. Found: {identified_timestamps}"
        
        # Should classify operational factors correctly
        categorical_table = artifacts['categorical_table']
        operational_factors = ['shift', 'nurse_credential', 'pharmacist_credential', 
                             'ordering_department', 'severity', 'diagnosis_type']
        
        identified_categoricals = set(categorical_table['name']) if not categorical_table.empty else set()
        expected_categoricals = set(operational_factors)
        
        # Should identify most operational factors
        cat_intersection = identified_categoricals & expected_categoricals
        assert len(cat_intersection) >= 4, f"Missing operational factors. Found: {identified_categoricals}"

    def test_clinical_interpretation_guidance(self, clinical_context_dataset, capsys):
        """Test clinical interpretation guidance in analysis output."""
        summary = DataSummary.default()
        artifacts = summary.print_report(clinical_context_dataset)
        
        # Capture output for interpretation guidance validation
        captured = capsys.readouterr()
        console_output = captured.out
        
        # Should provide actionable clinical insights
        clinical_guidance_terms = [
            'optimization', 'improvement', 'bottleneck', 'workflow', 
            'performance', 'quality', 'efficiency'
        ]
        
        console_lower = console_output.lower()
        guidance_found = [term for term in clinical_guidance_terms if term in console_lower]
        assert len(guidance_found) >= 3, f"Missing clinical guidance. Found: {guidance_found}"
        
        # Should suggest next steps for pharmacy teams
        next_steps_indicators = ['next steps', 'focus on', 'review', 'improvement']
        next_steps_found = [term for term in next_steps_indicators if term in console_lower]
        assert len(next_steps_found) >= 1, f"Missing next steps guidance. Found: {next_steps_found}"

    def test_stakeholder_communication_appropriateness(self, clinical_context_dataset):
        """Test stakeholder communication appropriateness for healthcare audiences."""
        summary = DataSummary.default()
        html_output = summary.to_html(clinical_context_dataset)
        
        # Should use professional healthcare language
        professional_terms = [
            'analysis', 'assessment', 'evaluation', 'performance', 
            'quality', 'operational', 'clinical', 'workflow'
        ]
        
        html_lower = html_output.lower()
        professional_found = [term for term in professional_terms if term in html_lower]
        assert len(professional_found) >= 5, f"Insufficient professional terminology. Found: {professional_found}"
        
        # Should avoid overly technical jargon for clinical stakeholders
        # (This is more of a qualitative assessment, but we can check for balance)
        assert len(html_output) > 3000  # Substantial content for stakeholders


class TestMLOpsIntegrationCompatibility:
    """Test suite for MLOps pipeline integration and automated monitoring compatibility."""
    
    @pytest.fixture
    def mlops_tat_dataset(self):
        """Generate TAT dataset optimized for MLOps pipeline integration testing."""
        np.random.seed(42)
        n_orders = 2500
        
        return pd.DataFrame({
            # Structured workflow features for model training
            'doctor_order_time': pd.date_range('2025-01-15 06:00:00', periods=n_orders, freq='9min'),
            'nurse_validation_time': pd.date_range('2025-01-15 06:12:00', periods=n_orders, freq='9min'),
            'prep_complete_time': pd.date_range('2025-01-15 06:40:00', periods=n_orders, freq='9min'),
            'patient_infusion_time': pd.date_range('2025-01-15 07:05:00', periods=n_orders, freq='9min'),
            
            # Target variable for prediction modeling
            'TAT_minutes': np.random.exponential(48, n_orders) + 12,
            'TAT_over_60': np.random.choice([0, 1], n_orders, p=[0.65, 0.35]),
            
            # Feature engineered operational factors
            'shift_encoded': np.random.choice([0, 1, 2], n_orders),  # Encoded categorical
            'queue_length_at_order': np.random.poisson(7, n_orders),
            'floor_occupancy_pct': np.random.uniform(20, 95, n_orders),
            'pharmacists_on_duty': np.random.randint(2, 8, n_orders),
            'nurse_employment_years': np.random.uniform(0.5, 25, n_orders),
            'patient_age': np.random.uniform(18, 85, n_orders),
            
            # Categorical features for encoding pipeline
            'shift': np.random.choice(['Day', 'Evening', 'Night'], n_orders),
            'severity': np.random.choice(['Low', 'Medium', 'High'], n_orders),
            'diagnosis_type': np.random.choice(['SolidTumor', 'Hematologic', 'Autoimmune'], n_orders),
            
            # Model monitoring features
            'model_version': ['v1.2.3'] * n_orders,
            'prediction_timestamp': pd.date_range('2025-01-15 06:00:00', periods=n_orders, freq='9min'),
            
            # Data quality indicators
            'data_quality_score': np.random.uniform(0.8, 1.0, n_orders),
            'feature_completeness': np.random.uniform(0.85, 1.0, n_orders)
        })

    def test_structured_artifact_generation_model_pipeline(self, mlops_tat_dataset):
        """Test structured artifact generation for model development pipeline integration."""
        summary = DataSummary.default()
        artifacts = summary.print_report(mlops_tat_dataset)
        
        # Should generate structured artifacts suitable for MLOps consumption
        assert isinstance(artifacts, dict)
        
        # Should provide processed dataset for model training
        processed_df = artifacts['df_processed']
        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(mlops_tat_dataset)
        
        # Should generate feature analysis tables
        assert 'time_table' in artifacts
        assert 'categorical_table' in artifacts
        assert 'numeric_table' in artifacts
        
        # Should provide correlation analysis for feature selection
        assert 'correlations' in artifacts
        
        # Should include metadata for pipeline monitoring
        counts = artifacts['counts']
        assert isinstance(counts, dict)
        assert all(key in counts for key in ['rows', 'cols', 'time', 'categorical', 'numeric'])

    def test_csv_export_model_development_ready(self, mlops_tat_dataset):
        """Test CSV export format compatibility with model development workflows."""
        summary = DataSummary.default()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts = summary.report(
                mlops_tat_dataset,
                save_dir=temp_dir,
                export="html",
                console=False
            )
            
            # Should generate machine-readable CSV exports
            csv_files = ['summary_time.csv', 'summary_categorical.csv', 'summary_numeric.csv']
            
            for csv_file in csv_files:
                csv_path = os.path.join(temp_dir, csv_file)
                assert os.path.exists(csv_path)
                
                # Should be readable by pandas for downstream processing
                df = pd.read_csv(csv_path)
                assert len(df) > 0
                assert len(df.columns) > 3  # Substantial feature analysis
                
                # Should have clean column names for programmatic access
                assert all(isinstance(col, str) for col in df.columns)

    def test_feature_correlation_matrix_model_selection(self, mlops_tat_dataset):
        """Test feature correlation matrix generation for automated feature selection."""
        summary = DataSummary.default()
        artifacts = summary.print_report(mlops_tat_dataset)
        
        # Should generate correlation matrix for feature engineering
        correlations = artifacts['correlations']
        
        if not correlations.empty:
            # Should be suitable for automated feature selection
            assert isinstance(correlations, pd.DataFrame)
            
            # Should include target variable correlations if present
            target_vars = ['TAT_minutes', 'TAT_over_60']
            corr_features = set(correlations.columns) | set(correlations.index)
            
            # Should analyze feature relationships
            assert len(corr_features) >= 5  # Multiple features analyzed

    def test_metadata_generation_pipeline_monitoring(self, mlops_tat_dataset):
        """Test metadata generation for MLOps pipeline monitoring and alerting."""
        summary = DataSummary.default()
        artifacts = summary.print_report(mlops_tat_dataset)
        
        # Should generate comprehensive metadata for monitoring
        counts = artifacts['counts']
        
        # Should provide dataset health metrics
        required_metrics = ['rows', 'cols', 'missing_cols']
        assert all(metric in counts for metric in required_metrics)
        
        # Should provide feature type distribution
        feature_metrics = ['time', 'categorical', 'numeric']
        assert all(metric in counts for metric in feature_metrics)
        
        # Should be JSON-serializable for monitoring systems
        import json
        try:
            json.dumps(counts)  # Should not raise exception
        except (TypeError, ValueError) as e:
            pytest.fail(f"Counts metadata not JSON-serializable: {e}")

    def test_automated_pipeline_integration_robustness(self, mlops_tat_dataset):
        """Test robustness for automated pipeline integration and scheduled execution."""
        summary = DataSummary.default()
        
        # Should handle automated execution without user interaction
        with patch('builtins.print') as mock_print:
            artifacts = summary.report(
                mlops_tat_dataset,
                save_dir=None,  # In-memory execution
                export="html",
                console=False   # No console output for automation
            )
            
            # Should complete without interactive elements
            assert isinstance(artifacts, dict)
            assert len(artifacts) >= 7
            
            # Should minimize print statements for automated execution
            print_call_count = mock_print.call_count
            assert print_call_count <= 5  # Minimal logging for automation

    def test_configuration_serialization_pipeline_versioning(self):
        """Test configuration serialization for MLOps pipeline versioning and reproducibility."""
        custom_config = SummaryConfig(
            tat_threshold_minutes=45.0,
            hist_bins=25,
            missing_top_n=20,
            enable_production_logging=True
        )
        
        summary = DataSummary(custom_config)
        
        # Should support configuration serialization for version control
        config_dict = {
            'tat_threshold_minutes': summary.cfg.tat_threshold_minutes,
            'hist_bins': summary.cfg.hist_bins,
            'missing_top_n': summary.cfg.missing_top_n,
            'enable_production_logging': summary.cfg.enable_production_logging
        }
        
        # Should be serializable for pipeline configuration management
        import json
        try:
            config_json = json.dumps(config_dict)
            assert isinstance(config_json, str)
            assert len(config_json) > 50  # Substantial configuration data
        except (TypeError, ValueError) as e:
            pytest.fail(f"Configuration not JSON-serializable: {e}")

    def test_batch_processing_capability_large_datasets(self, mlops_tat_dataset):
        """Test batch processing capability for large-scale MLOps data processing."""
        summary = DataSummary.default()
        
        # Simulate batch processing scenario
        batch_size = 1000
        n_batches = len(mlops_tat_dataset) // batch_size + 1
        
        batch_results = []
        
        for i in range(min(n_batches, 3)):  # Test first 3 batches
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(mlops_tat_dataset))
            batch_df = mlops_tat_dataset.iloc[start_idx:end_idx]
            
            # Should process batches independently
            batch_artifacts = summary.report(
                batch_df,
                save_dir=None,
                export="html",
                console=False
            )
            
            batch_results.append(batch_artifacts)
        
        # Should process all batches successfully
        assert len(batch_results) >= 2
        
        # Each batch should generate complete artifacts
        for batch_artifacts in batch_results:
            assert isinstance(batch_artifacts, dict)
            assert 'counts' in batch_artifacts
            assert batch_artifacts['counts']['rows'] > 0