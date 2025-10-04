"""
Test suite for DataIO class - simplified and accurate version
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
import tempfile
import os
from src.tat.data_io import DataIO


class TestDataIOBasic:
    """Basic functionality tests for DataIO class."""
    
    def test_dataio_initialization(self):
        """Test DataIO initialization with default parameters."""
        io = DataIO()
        assert io.sort_column == "order_id"
        assert io.sort_ascending is True
        assert io.validate_healthcare_data is True
        
    def test_dataio_initialization_custom(self):
        """Test DataIO initialization with custom parameters."""
        io = DataIO(sort_column="timestamp", sort_ascending=False, validate_healthcare_data=False)
        assert io.sort_column == "timestamp"
        assert io.sort_ascending is False
        assert io.validate_healthcare_data is False

    @patch("pandas.read_csv")
    @patch("os.path.exists")
    def test_read_csv_basic(self, mock_exists, mock_read_csv):
        """Test basic CSV reading functionality."""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({'order_id': [1, 2, 3], 'TAT_minutes': [30, 45, 60]})
        mock_read_csv.return_value = mock_df
        
        io = DataIO(validate_healthcare_data=False)  # Disable validation for simple test
        result = io.read_csv("test.csv")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        mock_read_csv.assert_called_once_with("test.csv")

    def test_read_csv_file_not_found(self):
        """Test FileNotFoundError for non-existent files."""
        io = DataIO()
        with pytest.raises(FileNotFoundError, match="Healthcare dataset not found"):
            io.read_csv("nonexistent.csv")

    def test_write_csv_basic(self):
        """Test basic CSV writing functionality."""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
            
        try:
            io = DataIO()
            io.write_csv(df, temp_path)
            
            # Verify file was created and has content
            assert os.path.exists(temp_path)
            result_df = pd.read_csv(temp_path)
            assert len(result_df) == 3
            assert list(result_df.columns) == ['col1', 'col2']
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDataIOValidation:
    """Test healthcare data validation functionality."""
    
    def test_validate_healthcare_dataset_basic(self):
        """Test basic healthcare dataset validation."""
        data = pd.DataFrame({
            'order_id': [1, 2, 3],
            'TAT_minutes': [30, 45, 60],
            'age': [25, 35, 45]
        })
        
        io = DataIO()
        result = io._validate_healthcare_dataset(data, "/test/path.csv")
        
        assert isinstance(result, dict)
        assert 'validation_status' in result
        assert 'dataset_shape' in result
        assert 'source_path' in result
        assert result['source_path'] == "/test/path.csv"

    def test_validate_healthcare_dataset_empty(self):
        """Test validation with empty dataset."""
        data = pd.DataFrame()
        
        io = DataIO()
        result = io._validate_healthcare_dataset(data, "/test/empty.csv")
        
        assert isinstance(result, dict)
        assert result['dataset_shape'] == (0, 0)


class TestDataIOAdvanced:
    """Advanced functionality tests for DataIO class methods."""
    
    def test_validate_tat_data_complete(self):
        """Test TAT data validation with complete dataset."""
        data = pd.DataFrame({
            'TAT_minutes': [30, 45, 60, 75, 90],
            'doctor_order_time': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'patient_infusion_time': pd.date_range('2024-01-01 01:00', periods=5, freq='1h'),
            'age': [25, 35, 45, 55, 65],
            'severity': ['Low', 'Medium', 'High', 'Medium', 'Low']
        })
        
        io = DataIO()
        result = io.validate_tat_data(data)
        
        assert isinstance(result, dict)
        assert 'tat_statistics' in result
        assert 'clinical_priority' in result
        assert 'missing_tat_rate' in result
        
    def test_validate_tat_data_missing_columns(self):
        """Test TAT validation with missing columns."""
        data = pd.DataFrame({
            'some_column': [1, 2, 3]
        })
        
        io = DataIO()
        result = io.validate_tat_data(data)
        
        assert isinstance(result, dict)
        assert 'validation_status' in result
        assert 'workflow_completeness' in result
        
    def test_get_data_summary_basic(self):
        """Test data summary generation."""
        data = pd.DataFrame({
            'TAT_minutes': [30, 45, 60, 75, 90],
            'age': [25, 35, 45, 55, 65],
            'severity': ['Low', 'Medium', 'High', 'Medium', 'Low']
        })
        
        io = DataIO()
        summary = io.get_data_summary(data)
        
        assert isinstance(summary, dict)
        assert 'dataset_overview' in summary
        assert 'tat_analysis' in summary
        assert 'missing_data_analysis' in summary
        
    def test_get_data_summary_empty_dataframe(self):
        """Test data summary with empty DataFrame."""
        data = pd.DataFrame()
        
        io = DataIO()
        summary = io.get_data_summary(data)
        
        assert isinstance(summary, dict)
        assert 'dataset_overview' in summary
        
    def test_get_temporal_coverage_with_timestamps(self):
        """Test temporal coverage analysis with timestamp columns."""
        data = pd.DataFrame({
            'doctor_order_time': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'patient_infusion_time': pd.date_range('2024-01-01 01:00', periods=5, freq='1h'),
            'TAT_minutes': [30, 45, 60, 75, 90]
        })
        
        io = DataIO()
        coverage = io._get_temporal_coverage(data)
        
        if coverage is not None:
            assert isinstance(coverage, dict)
            
    def test_write_csv_with_index(self):
        """Test CSV writing with index parameter."""
        data = pd.DataFrame({
            'TAT_minutes': [30, 45, 60],
            'patient_id': ['P001', 'P002', 'P003']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
            
        try:
            io = DataIO()
            io.write_csv(data, temp_path, index=True)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Read back and verify
            result = pd.read_csv(temp_path)
            assert len(result) == 3
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDataIOIntegration:
    """Integration tests for DataIO operations."""
    
    def test_read_write_roundtrip(self):
        """Test reading and writing data maintains integrity."""
        original_df = pd.DataFrame({
            'order_id': [1, 2, 3, 4, 5],
            'TAT_minutes': [30.5, 45.2, 60.0, 75.8, 90.1],
            'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005']
        })
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
            
        try:
            io = DataIO(validate_healthcare_data=False)  # Disable validation for simple test
            
            # Write data
            io.write_csv(original_df, temp_path)
            
            # Read it back
            result_df = pd.read_csv(temp_path)
            
            # Verify data integrity
            assert len(result_df) == len(original_df)
            assert list(result_df.columns) == list(original_df.columns)
            pd.testing.assert_frame_equal(result_df, original_df)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_full_pipeline_with_validation(self):
        """Test complete data pipeline with healthcare validation."""
        # Create healthcare-like data
        healthcare_data = pd.DataFrame({
            'order_id': [1, 2, 3, 4, 5],
            'TAT_minutes': [30, 45, 60, 75, 90],
            'age': [25, 35, 45, 55, 65],
            'sex': ['M', 'F', 'M', 'F', 'M'],
            'diagnosis_type': ['Type1', 'Type2', 'Type1', 'Type2', 'Type1'],
            'severity': ['Low', 'Medium', 'High', 'Medium', 'Low'],
            'doctor_order_time': pd.date_range('2024-01-01', periods=5, freq='1h'),
            'patient_infusion_time': pd.date_range('2024-01-01 01:00', periods=5, freq='1h')
        })
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            temp_path = temp_file.name
            
        try:
            io = DataIO(validate_healthcare_data=True)
            
            # Write healthcare data
            io.write_csv(healthcare_data, temp_path)
            
            # Read with validation
            result_df = io.read_csv(temp_path)
            
            # Verify data integrity and structure
            assert len(result_df) == 5
            assert 'TAT_minutes' in result_df.columns
            assert 'age' in result_df.columns
            
            # Test validation methods
            tat_validation = io.validate_tat_data(result_df)
            assert isinstance(tat_validation, dict)
            
            data_summary = io.get_data_summary(result_df)
            assert isinstance(data_summary, dict)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestDataIOErrorHandling:
    """Error handling and edge case tests for DataIO."""
    
    def test_validate_healthcare_dataset_with_missing_data(self):
        """Test validation with datasets containing missing values."""
        data = pd.DataFrame({
            'TAT_minutes': [30, None, 60, 75, None],
            'age': [25, 35, None, 55, 65],
            'severity': ['Low', None, 'High', 'Medium', 'Low']
        })
        
        io = DataIO()
        result = io._validate_healthcare_dataset(data, "/test/missing_data.csv")
        
        assert isinstance(result, dict)
        assert 'validation_status' in result
        
    def test_get_data_summary_with_mixed_types(self):
        """Test data summary with mixed data types."""
        data = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'string_col': ['A', 'B', 'C', 'D', 'E'],
            'bool_col': [True, False, True, False, True],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
        
        io = DataIO()
        summary = io.get_data_summary(data)
        
        assert isinstance(summary, dict)
        assert 'dataset_overview' in summary
        assert 'categorical_distribution' in summary