""" 
Dataset preparation script for TAT models.

Processes raw data through feature engineering pipeline
to create model-ready datasets.taset Preparation Script for Medication Preparation TAT Prediction Models

Command-line interface for creating production-ready datasets supporting medication
preparation turnaround time analysis and pharmacy workflow optimization. Provides
comprehensive dataset preparation capabilities designed specifically for healthcare
analytics environments and clinical operations excellence initiatives.

Key Features:
- F0 Dataset Creation: Real-time inference features without future information leakage
- Diagnostics Dataset Creation: Comprehensive analysis features including step-to-step delays
- Automated artifact saving: Scalers, metadata, and feature removal information preservation
- Production-ready processing: Scalable pipeline suitable for large healthcare datasets
- Quality assurance: Comprehensive validation ensuring clinical data integrity throughout processing

Technical Features:
- Automated F0 dataset creation preventing future information leakage in production deployment
- Comprehensive diagnostics dataset with delay features for bottleneck identification analysis
- Artifact management supporting model deployment and production monitoring requirements
- Error handling and validation ensuring robust operation in clinical environments
- Consistent processing pipeline integration with comprehensive TAT analysis framework

Usage Examples:
    # Standard dataset preparation for TAT prediction modeling
    python prepare_dataset.py
    
    # Creates F0 and diagnostics datasets with comprehensive feature engineering
    # Saves scalers, metadata, and removal information for production deployment

"""
import logging
from pathlib import Path
import joblib
from typing import Dict, Any

import pandas as pd
from tat.data_io import DataIO
from tat.pipelines.make_dataset import make_f0, make_diagnostics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths for healthcare analytics data pipeline infrastructure
PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

def main() -> int:
    """
    Execute comprehensive dataset preparation for medication preparation TAT prediction modeling.

    Creates production-ready F0 and diagnostics datasets through healthcare-optimized
    feature engineering pipelines supporting both real-time inference and analytical
    research. Essential for healthcare analytics enabling accurate TAT prediction
    modeling and pharmacy workflow optimization in clinical environments.

    Dataset Creation Workflow:
    - Raw data loading: Validates and loads medication preparation TAT datasets
    - F0 processing: Creates real-time inference features preventing information leakage
    - Diagnostics processing: Generates comprehensive analysis features with delay calculations
    - Artifact preservation: Saves scalers, metadata, and processing information for deployment
    - Quality validation: Ensures clinical data integrity throughout preparation workflows

    Production Deployment Support:
    - F0 dataset: Real-time prediction features safe for production inference deployment
    - Diagnostics dataset: Comprehensive analytical features supporting workflow research
    - Scaler artifacts: Production-ready transformation objects for consistent feature scaling
    - Metadata preservation: Feature removal and processing information for model monitoring

    Returns:
        int: Exit code indicating preparation success (0) or specific failure modes for
        healthcare operations monitoring and automated pipeline integration requirements.

    Exit Codes:
        0: Successful dataset preparation and artifact saving for production deployment
        1: Input file access issues preventing healthcare data processing and analysis
        2: Data loading failures compromising dataset preparation and feature engineering
        3: Processing failures affecting dataset creation and production artifact generation

    Example:
        For comprehensive healthcare analytics dataset preparation:
        ```bash
        # Execute dataset preparation pipeline
        python prepare_dataset.py
        
        # Results in production-ready datasets and artifacts:
        # - data/processed/f0.csv (real-time inference features)
        # - data/processed/diagnostics.csv (analytical features with delays)
        # - data/models/scalers and metadata for deployment
        ```
    """
    
    logger.info("🏥 Starting comprehensive TAT dataset preparation pipeline for healthcare analytics")
    logger.info("   Supporting medication preparation workflow optimization and clinical operations excellence")
    
    # Ensure healthcare analytics directory structure exists for organized data management
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Healthcare analytics directory structure validated:")
    logger.info(f"   • Processed data: {PROCESSED_DIR}")
    logger.info(f"   • Model artifacts: {MODEL_DIR}")
    
    # Load raw healthcare dataset with comprehensive validation and error handling
    io = DataIO()
    input_path = RAW_DIR / "DFCI_TAT_Dataset_100k.csv"
    if not input_path.exists():
        logger.error(f"❌ Healthcare dataset not found: {input_path}")
        logger.error("   Please ensure TAT dataset is available for processing")
        return 1
        
    try:
        logger.info(f"📊 Loading raw medication preparation TAT dataset: {input_path.name}")
        df_raw = io.read_csv(input_path)
        logger.info(f"✅ Healthcare dataset loaded successfully: {df_raw.shape[0]:,} medication orders, {df_raw.shape[1]} variables")
        
        # Provide dataset overview for clinical context and quality assessment
        logger.info(f"   • Data shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        logger.info(f"   • Memory usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Identify key healthcare variable categories for processing insight
        time_cols = [col for col in df_raw.columns if 'time' in col.lower()]
        lab_cols = [col for col in df_raw.columns if col.startswith('lab_')]
        clinical_cols = [col for col in df_raw.columns if any(term in col.lower() 
                        for term in ['age', 'diagnosis', 'severity', 'treatment'])]
        
        logger.info(f"   • Timestamp variables: {len(time_cols)} (workflow timing)")
        logger.info(f"   • Laboratory values: {len(lab_cols)} (clinical assessment)")
        logger.info(f"   • Clinical variables: {len(clinical_cols)} (patient context)")
        
    except Exception as e:
        logger.error(f"❌ Failed to load healthcare dataset: {e}")
        logger.error("   Please verify CSV format and data integrity for TAT analysis")
        return 2
    
    try:
        logger.info("\n" + "="*80)
        logger.info("CREATING PRODUCTION-READY DATASETS FOR HEALTHCARE ANALYTICS")
        logger.info("="*80)
        
        # Create F0 dataset for real-time TAT prediction deployment
        logger.info("\n🚀 Creating F0 dataset for real-time TAT prediction deployment...")
        logger.info("   • Real-time inference features without future information leakage")
        logger.info("   • Production-safe feature set enabling live pharmacy workflow optimization")
        logger.info("   • Healthcare-optimized scaling preserving clinical interpretability")
        
        X_f0, y_reg_f0, y_clf_f0, scaler_f0, f0_removal_info = make_f0(df_raw)
        
        logger.info(f"✅ F0 dataset creation complete:")
        logger.info(f"   • Features: {X_f0.shape[1]} real-time inference variables")
        logger.info(f"   • Samples: {X_f0.shape[0]:,} medication orders processed")
        logger.info(f"   • Targets: Regression (TAT_minutes) + Classification (TAT_over_60)")
        
        # Create diagnostics dataset for comprehensive workflow analysis
        logger.info("\n🔬 Creating diagnostics dataset for comprehensive workflow analysis...")
        logger.info("   • Analytical features including step-to-step delays for bottleneck identification")
        logger.info("   • Detailed workflow analysis supporting pharmacy operations research")
        logger.info("   • Process insight capabilities enabling medication preparation optimization")
        
        X_diag, y_reg_diag, y_clf_diag, scaler_diag, diag_removal_info = make_diagnostics(df_raw)
        
        logger.info(f"✅ Diagnostics dataset creation complete:")
        logger.info(f"   • Features: {X_diag.shape[1]} comprehensive analytical variables")
        logger.info(f"   • Samples: {X_diag.shape[0]:,} medication orders processed")
        logger.info(f"   • Delay features: {len([col for col in X_diag.columns if 'delay_' in col])} bottleneck indicators")
        
        # Save comprehensive datasets for production deployment and analysis
        logger.info(f"\n💾 Saving production-ready datasets and deployment artifacts...")
        
        # Combine and save F0 dataset for real-time inference
        f0_df = pd.concat([
            X_f0, 
            y_reg_f0.rename('TAT_minutes'), 
            y_clf_f0.rename('TAT_over_60')
        ], axis=1)
        f0_path = PROCESSED_DIR / "f0.csv"
        f0_df.to_csv(f0_path, index=False)
        logger.info(f"   ✅ F0 dataset saved: {f0_path} ({f0_df.shape[0]:,} × {f0_df.shape[1]})")
        
        # Combine and save diagnostics dataset for analytical research
        diag_df = pd.concat([
            X_diag, 
            y_reg_diag.rename('TAT_minutes'), 
            y_clf_diag.rename('TAT_over_60')
        ], axis=1)
        diag_path = PROCESSED_DIR / "diagnostics.csv"
        diag_df.to_csv(diag_path, index=False)
        logger.info(f"   ✅ Diagnostics dataset saved: {diag_path} ({diag_df.shape[0]:,} × {diag_df.shape[1]})")
        
        # Save production deployment artifacts for model consistency
        logger.info(f"   💼 Saving production deployment artifacts...")
        
        # Save scaling artifacts for consistent feature transformation
        f0_scaler_path = MODEL_DIR / "f0_scaler.joblib"
        diag_scaler_path = MODEL_DIR / "diagnostics_scaler.joblib"
        joblib.dump(scaler_f0, f0_scaler_path)
        joblib.dump(scaler_diag, diag_scaler_path)
        logger.info(f"     • F0 scaler: {f0_scaler_path}")
        logger.info(f"     • Diagnostics scaler: {diag_scaler_path}")
        
        # Save feature removal metadata for model monitoring and validation
        metadata_path = MODEL_DIR / "feature_removal_metadata.joblib"
        feature_metadata = {
            'f0': f0_removal_info, 
            'diagnostics': diag_removal_info,
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'raw_data_shape': df_raw.shape,
            'f0_final_shape': f0_df.shape,
            'diagnostics_final_shape': diag_df.shape
        }
        joblib.dump(feature_metadata, metadata_path)
        logger.info(f"     • Feature metadata: {metadata_path}")
        
        # Provide comprehensive completion summary for healthcare operations
        logger.info(f"\n" + "="*80)
        logger.info("HEALTHCARE ANALYTICS DATASET PREPARATION COMPLETE")
        logger.info("="*80)
        logger.info(f"📊 Dataset Summary:")
        logger.info(f"   • Input: {df_raw.shape[0]:,} medication orders, {df_raw.shape[1]} raw variables")
        logger.info(f"   • F0 Output: {f0_df.shape[0]:,} orders, {f0_df.shape[1]} real-time features")
        logger.info(f"   • Diagnostics Output: {diag_df.shape[0]:,} orders, {diag_df.shape[1]} analytical features")
        
        logger.info(f"\n🎯 Production Deployment Ready:")
        logger.info(f"   • Real-time TAT prediction: {f0_path}")
        logger.info(f"   • Workflow bottleneck analysis: {diag_path}")
        logger.info(f"   • Scaling artifacts: {MODEL_DIR}")
        logger.info(f"   • Feature metadata: {metadata_path}")
        
        logger.info(f"\n📋 Next Steps for Healthcare Analytics:")
        logger.info(f"   1. Train TAT prediction models using F0 dataset for production deployment")
        logger.info(f"   2. Conduct bottleneck analysis using diagnostics dataset for workflow optimization")
        logger.info(f"   3. Deploy real-time prediction system using F0 features and scaling artifacts")
        logger.info(f"   4. Monitor model performance using feature metadata and validation frameworks")
        
        logger.info(f"\n🎉 Healthcare analytics pipeline ready: {PROCESSED_DIR}")
        logger.info("   Supporting pharmacy workflow optimization and clinical operations excellence")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Healthcare dataset preparation failed: {e}", exc_info=True)
        logger.error("   Please verify data integrity and processing requirements")
        logger.error("   Check available disk space and file permissions for artifact saving")
        return 3

if __name__ == "__main__":
    """
    Execute comprehensive dataset preparation for medication preparation TAT prediction modeling.
    
    Provides command-line interface for healthcare analytics supporting pharmacy
    workflow optimization and clinical operations excellence through production-ready
    dataset creation and comprehensive artifact management capabilities.
    """
    raise SystemExit(main())