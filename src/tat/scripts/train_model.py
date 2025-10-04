"""
Model training script for TAT prediction.

Command-line interface for training multiple model configurations
and generating performance reports for medication preparation analysis.
- Patient care throughput enhancement via predictive resource allocation and scheduling

Healthcare Analytics Integration:
- Compatible with existing pharmacy information systems and healthcare IT infrastructure
- Clinical data processing ensuring medically relevant model development and validation
- Production-ready models suitable for real-time TAT prediction and workflow optimization
- Quality assurance features maintaining clinical data integrity throughout training workflows

Technical Features:
- Multi-model ensemble training optimized for diverse healthcare operational scenarios
- Healthcare-specific  Numerical Features aligned with clinical quality standards
- Feature importance analysis with clinical interpretation and workflow optimization insights
- Comprehensive artifact management supporting production deployment and model monitoring
- Error handling and validation ensuring robust operation in healthcare environments

Usage Examples:
    # Standard model training for TAT prediction deployment
    python train_model.py
    
    # Comprehensive training across all configurations for optimal model selection
    # Generates production-ready models with healthcare-optimized evaluation
"""
import logging
from pathlib import Path
import pandas as pd
import joblib

from tat.config import TRAINING_CONFIGS
from tat.data_io import DataIO
from tat.pipelines.make_dataset import make_f0
from tat.models.factory import TATTrainingOrchestrator

# Configure logging for healthcare model training
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = DATA_DIR / "models"

def train_tat_models():
    """
    Execute comprehensive TAT prediction model training for pharmacy workflow optimization.
    
    Orchestrates multi-configuration model training across diverse scaling strategies
    optimized for medication preparation TAT analysis and healthcare workflow optimization.
    Essential for developing accurate prediction capabilities supporting pharmacy operations
    and patient care throughput enhancement through evidence-based machine learning models.
    
    Training Workflow Components:
    - Multi-configuration dataset preparation with healthcare-optimized feature engineering
    - Ensemble model training across diverse algorithms suited for clinical applications
    - Comprehensive performance evaluation focused on 60-minute TAT threshold compliance
    - Clinical feature importance analysis supporting pharmacy workflow understanding
    - Production deployment artifact generation with MLOps-ready model management
    
    Healthcare Quality Metrics:
    - RMSE and MAE evaluation for precise TAT prediction accuracy assessment
    - Threshold compliance analysis supporting clinical quality standards monitoring
    - Workflow optimization scoring aligned with pharmacy operations improvement objectives
    - Feature importance ranking enabling evidence-based bottleneck identification
    - Clinical insight generation supporting healthcare stakeholder decision-making
    
    Returns:
        dict: Comprehensive training results across all configurations including
        model  Numerical Features, feature importance analysis, and deployment
        artifacts supporting healthcare analytics and pharmacy workflow optimization.
    
    Example:
        For comprehensive TAT prediction model development:
        ```python
        # Execute complete training pipeline
        results = train_tat_models()
        
        # Access best performing model for production deployment
        best_config = min(results.keys(), 
                         key=lambda x: results[x]['training_results']['xgboost_regression']['metrics']['RMSE'])
        ```
    """
    print("🏥 TAT PREDICTION MODEL TRAINING SYSTEM")
    print("=" * 70)
    print("Advanced Analytics for Pharmacy Workflow Optimization")
    print("Supporting 60-Minute TAT Threshold Compliance & Patient Care Enhancement")
    print("=" * 70)
    
    # Load comprehensive medication preparation TAT dataset
    logger.info("📊 Loading medication preparation TAT dataset for model training...")
    io = DataIO()
    dataset_path = DATA_DIR / 'raw' / 'DFCI_TAT_Dataset_100k.csv'
    
    if not dataset_path.exists():
        logger.error(f"❌ Healthcare dataset not found: {dataset_path}")
        raise FileNotFoundError(f"Required dataset not available: {dataset_path}")
    
    df_raw = io.read_csv(dataset_path)
    logger.info(f"✅ Healthcare dataset loaded: {df_raw.shape[0]:,} medication orders, {df_raw.shape[1]} variables")
    
    # Provide dataset overview for clinical context and training preparation
    logger.info(f"   • Data shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    logger.info(f"   • Memory usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Shuffle dataset to ensure robust model training and validation
    df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info("✅ Dataset shuffled for robust training distribution and validation")
    
    # Initialize comprehensive training results storage
    all_training_results = {}
    
    # Execute model training across all healthcare-optimized configurations
    logger.info(f"\n🚀 Starting multi-configuration model training pipeline...")
    logger.info(f"   • Training configurations: {len(TRAINING_CONFIGS)}")
    logger.info(f"   • Healthcare focus: Pharmacy workflow optimization and TAT prediction")
    
    # Train models for each configuration optimized for healthcare workflows
    for config_name, config in TRAINING_CONFIGS.items():
        print(f"\n📊 {config_name.upper().replace('_', ' ')} CONFIGURATION")
        print(f"Description: {config['description']}")
        print(f"Clinical Focus: {config['focus']}")
        print(f"Scaling Strategy: {config['scaling_strategy']}")
        print("-" * 50)
        
        logger.info(f"🔧 Preparing dataset with {config['scaling_strategy']} scaling strategy...")
        
        # Prepare dataset with configuration-specific healthcare-optimized scaling
        X, y_reg, y_clf, scaler_info, removal_info = make_f0(
            df_raw,
            scaling_strategy=config['scaling_strategy']
        )
        
        logger.info(f"✅ Dataset preparation complete: {X.shape[0]:,} samples, {X.shape[1]} features")
        logger.info(f"   • Features ready for {config_name} model training")
        logger.info(f"   • Scaling strategy: {config['scaling_strategy']} (healthcare-optimized)")
        
        # Initialize training orchestrator for comprehensive model development
        logger.info(f"🤖 Initializing training orchestrator for {config_name}...")
        orchestrator = TATTrainingOrchestrator(
            scaling_strategy=config['scaling_strategy'],
            random_state=42  # Reproducible training for healthcare validation
        )
        
        # Execute comprehensive model training suite
        logger.info(f"🎯 Training model ensemble for TAT prediction...")
        training_results = orchestrator.train_all_models(X, y_reg, y_clf)
        
        # Save configuration-specific models and artifacts for deployment
        config_output_dir = MODEL_DIR / config_name
        config_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving {config_name} training artifacts...")
        orchestrator.save_all_models(config_output_dir)
        
        # Save preprocessing artifacts essential for production deployment
        joblib.dump(scaler_info, config_output_dir / "scaler_info.joblib")
        joblib.dump(removal_info, config_output_dir / "removal_info.joblib")
        
        logger.info(f"✅ Production artifacts saved: {config_output_dir}")
        logger.info(f"   • Models: Trained ensemble ready for deployment")
        logger.info(f"   • Scalers: Preprocessing artifacts for consistent inference")
        logger.info(f"   • Metadata: Feature removal and processing information")
        
        # Store comprehensive results for cross-configuration analysis
        all_training_results[config_name] = {
            'training_results': training_results,
            'best_regression_model': orchestrator.get_best_model(),
            'scaling_strategy': config['scaling_strategy'],
            'dataset_shape': X.shape,
            'config_metadata': {
                'description': config['description'],
                'focus': config['focus'],
                'clinical_applications': config.get('clinical_applications', [])
            }
        }
        
        print(f"✅ {config_name} training complete - Models ready for healthcare deployment")
    
    # Generate comprehensive cross-configuration performance analysis
    print(f"\n🏆 TAT PREDICTION - COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 70)
    print("Healthcare Analytics Model Comparison & Clinical Feature Insights")
    print("=" * 70)
    
    logger.info("📊 Generating comprehensive model performance comparison...")
    generate_final_comparison_report(all_training_results)
    
    logger.info("🏥 Generating clinical feature importance analysis...")
    generate_clinical_feature_report(all_training_results)
    
    # Save comprehensive results for MLOps deployment and monitoring
    logger.info("💾 Saving comprehensive training results for production deployment...")
    save_comprehensive_results(all_training_results)
    
    # Training completion summary for healthcare stakeholders
    print(f"\n" + "="*70)
    print("✅ TAT PREDICTION MODEL TRAINING COMPLETE")
    print("="*70)
    print(f"🎯 Training Summary:")
    # Count distinct model types properly (avoiding double-counting ensemble components)
    distinct_models = set()
    for config_name, config_data in all_training_results.items():
        for model_name in config_data['training_results'].keys():
            # Map to base model type to avoid counting duplicates
            base_model = model_name.replace('_regression', '').split('_')[0]
            distinct_models.add(f"{base_model}_{config_name}")
    
    print(f"   • Configurations Trained: {len(all_training_results)}")
    print(f"   • Distinct Model Approaches: {len(distinct_models)}")
    print(f"   • Healthcare Focus: 60-minute TAT threshold compliance")
    print(f"   • Clinical Application: Pharmacy workflow optimization")
    
    print(f"\n📁 Production Deployment Assets:")
    print(f"   • Model Directory: {MODEL_DIR}")
    print(f"   • Training Results: complete_training_results.joblib")
    print(f"   • Deployment Metadata: deployment_metadata.joblib")
    print(f"   • Performance Comparison: model_comparison.csv")
    print(f"   • Feature Analysis: feature_importance_summary.json")
    
    print(f"\n📋 Next Steps for Healthcare Analytics:")
    print(f"   1. Review model performance comparison for optimal configuration selection")
    print(f"   2. Analyze clinical feature importance for pharmacy workflow insights")
    print(f"   3. Deploy selected model for real-time TAT prediction and monitoring")
    print(f"   4. Establish continuous monitoring for model performance and healthcare impact")
    
    logger.info("🎉 Comprehensive TAT prediction model training pipeline completed successfully")
    return all_training_results

def generate_final_comparison_report(all_results: dict) -> None:
    """
    Generate comprehensive model performance comparison with clinical feature insights.
    
    Creates detailed performance analysis across all training configurations supporting
    healthcare stakeholder decision-making for optimal model selection and deployment.
    Essential for pharmacy workflow optimization enabling evidence-based model selection
    and clinical feature understanding supporting healthcare operations excellence.
    
    Healthcare Performance Analysis:
    - RMSE and MAE evaluation for accurate TAT prediction assessment and validation
    - Threshold compliance analysis focusing on 60-minute clinical quality standards
    - Workflow optimization scoring aligned with pharmacy operations improvement objectives
    - Feature importance ranking enabling evidence-based bottleneck identification
    - Clinical insight generation supporting healthcare stakeholder communication
    
    Args:
        all_results: Comprehensive training results across configurations including
                    model performance, feature importance, and clinical insights for
                    healthcare analytics and pharmacy workflow optimization analysis.
    """
    
    # Collect regression model results with healthcare-focused metrics
    # Focus on distinct model approaches to avoid double-counting ensemble components
    comparison_data = []
    feature_importance_summary = {}
    
    for config_name, config_data in all_results.items():
        for model_name, model_result in config_data['training_results'].items():
            if 'RMSE' in model_result['metrics']:
                # Clean model name for proper identification
                clean_model_name = model_name.replace('_regression', '')
                
                # Filter out base learners from comprehensive ensemble - only keep stacking
                if config_name == 'comprehensive_ensemble' and clean_model_name != 'stacking':
                    continue  # Skip Ridge, XGBoost, Random Forest from ensemble config
                
                metrics = model_result['metrics']
                comparison_data.append({
                    'Configuration': config_name,
                    'Model': model_name.replace('_regression', ''),
                    'Scaling_Strategy': config_data['scaling_strategy'],
                    'RMSE_minutes': metrics['RMSE'],
                    'MAE_minutes': metrics['MAE'],
                    'Within_10min_%': metrics.get('within_10min_pct', 0),
                    'Within_30min_%': metrics.get('within_30min_pct', 0),
                    'Threshold_60min_Accuracy_%': metrics.get('threshold_60min_accuracy', 0),
                    'Workflow_Optimization_Score': metrics.get('healthcare_score', 0),
                    'Clinical_Applicability': config_data['config_metadata']['focus']
                })
                
                # Collect feature importance for clinical workflow analysis
                if 'feature_importance' in metrics:
                    top_features = metrics['feature_importance'].get('top_features', [])
                    # Include even if empty to ensure model appears in analysis
                    feature_importance_summary[f"{model_name}_{config_name}"] = {
                        'top_features': top_features[:10] if top_features else [],
                        'clinical_insights': metrics.get('clinical_insights', []),
                        'workflow_relevance': metrics['feature_importance'].get('workflow_categories', {}),
                        'has_features': len(top_features) > 0
                    }
    
    if not comparison_data:
        logger.warning("❌ No regression models found for performance comparison")
        print("❌ No regression models available for performance analysis")
        return
    
    # Create comprehensive comparison DataFrame for healthcare analysis
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('RMSE_minutes')
    
    logger.info(f"✅ Performance comparison generated: {len(comparison_df)} distinct model approaches analyzed")
    
    print("\n📈 TAT PREDICTION MODEL PERFORMANCE COMPARISON:")
    print("-" * 80)
    
    # Display all models since we now have only distinct approaches
    top_models = comparison_df.head(len(comparison_df))
    for idx, (_, row) in enumerate(top_models.iterrows(), 1):
        model_key = f"{row['Model']}_regression_{row['Configuration']}"
        
        print(f"{idx}. {row['Model'].upper()} ({row['Configuration'].replace('_', ' ').title()}):")
        print(f"   • Prediction Accuracy: {row['RMSE_minutes']:.2f} min RMSE, {row['MAE_minutes']:.2f} min MAE")
        print(f"   • Clinical Performance: {row['Within_30min_%']:.1f}% within 30-min window")
        print(f"   • TAT Threshold (60min): {row['Threshold_60min_Accuracy_%']:.1f}% accuracy")
        print(f"   • Workflow Score: {row['Workflow_Optimization_Score']:.1f}/100")
        print(f"   • Clinical Focus: {row['Clinical_Applicability']}")
        
        # Display top clinical features for pharmacy workflow optimization
        if model_key in feature_importance_summary:
            summary_data = feature_importance_summary[model_key]
            top_features = summary_data['top_features']
            if top_features and summary_data.get('has_features', True):
                print(f"   • Key Workflow Predictors:")
                for i, feature in enumerate(top_features[:3], 1):
                    feat_name = feature.get('feature', 'Unknown')
                    importance = feature.get('importance_pct', feature.get('shap_importance_pct', 0))
                    # Clean feature name for clinical interpretation
                    clean_name = feat_name.replace('_', ' ').title()
                    print(f"     {i}. {clean_name} ({importance:.1f}% importance)")
            elif clean_model_name == 'stacking':
                print(f"   • Ensemble Model: Combines Ridge, XGBoost, and Random Forest")
                print(f"   • Meta-learner: Uses base model predictions for final TAT prediction")
                print(f"   • Feature Importance: Distributed across base learners")
        print()
    
    # Highlight recommended model for production deployment
    best_model = top_models.iloc[0]
    best_model_key = f"{best_model['Model']}_regression_{best_model['Configuration']}"
    
    print("🚀 RECOMMENDED FOR PRODUCTION DEPLOYMENT:")
    print("-" * 50)
    print(f"Model Type: {best_model['Model'].upper()}")
    print(f"Configuration: {best_model['Configuration'].replace('_', ' ').title()}")
    print(f"Clinical Performance: {best_model['RMSE_minutes']:.2f}min RMSE")
    print(f"TAT Threshold Accuracy: {best_model['Threshold_60min_Accuracy_%']:.1f}%")
    print(f"Workflow Optimization Potential: {best_model['Within_30min_%']:.1f}% accurate predictions")
    print(f"Healthcare Application: {best_model['Clinical_Applicability']}")
    
    # Provide clinical insights for the recommended model
    if best_model_key in feature_importance_summary:
        clinical_insights = feature_importance_summary[best_model_key]['clinical_insights']
        if clinical_insights:
            print(f"\n🏥 KEY CLINICAL INSIGHTS FOR PHARMACY WORKFLOW OPTIMIZATION:")
            for i, insight in enumerate(clinical_insights[:4], 1):
                print(f"   {i}. {insight}")
        
        # Workflow category analysis for targeted improvements
        workflow_categories = feature_importance_summary[best_model_key].get('workflow_relevance', {})
        if workflow_categories:
            print(f"\n⚙️  WORKFLOW CATEGORY IMPACT ANALYSIS:")
            for category, impact in sorted(workflow_categories.items(), 
                                         key=lambda x: x[1], reverse=True)[:3]:
                print(f"   • {category.replace('_', ' ').title()}: {impact:.1f}% total importance")
    
    # Save comprehensive comparison results for healthcare documentation
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    logger.info(f"✅ Model comparison saved: {MODEL_DIR / 'model_comparison.csv'}")
    
    # Save detailed feature importance analysis for clinical team review
    import json
    with open(MODEL_DIR / "feature_importance_summary.json", 'w') as f:
        json.dump(feature_importance_summary, f, indent=2)
    
    print(f"\n📊 Comprehensive analysis artifacts saved:")
    print(f"   • Performance comparison: {MODEL_DIR / 'model_comparison.csv'}")
    print(f"   • Feature importance: {MODEL_DIR / 'feature_importance_summary.json'}")
    
    logger.info("✅ Comprehensive model performance comparison completed")

def generate_clinical_feature_report(all_results: dict) -> None:
    """
    Generate focused clinical feature importance analysis for pharmacy operations teams.
    
    Creates comprehensive feature importance reporting with clinical interpretation
    supporting pharmacy workflow optimization and evidence-based decision-making.
    Essential for healthcare stakeholders understanding predictive factors affecting
    medication preparation TAT and enabling targeted workflow improvement initiatives.
    
    Clinical Feature Analysis Components:
    - Top feature importance ranking across all trained models with clinical context
    - Workflow category analysis identifying operational areas affecting TAT performance
    - Clinical insight generation supporting evidence-based pharmacy optimization decisions
    - Bottleneck identification through predictive feature analysis and interpretation
    - Healthcare stakeholder communication with actionable workflow improvement recommendations
    
    Args:
        all_results: Comprehensive training results including feature importance analysis
                    and clinical insights supporting pharmacy workflow optimization and
                    healthcare operations excellence through evidence-based understanding.

    """
    logger.info("🏥 Generating clinical feature importance analysis for healthcare stakeholders...")
    
    print(f"\n🏥 TAT PREDICTION - CLINICAL FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    print("Evidence-Based Insights for Pharmacy Workflow Optimization")
    print("=" * 70)
    
    # Aggregate feature importance across all models for comprehensive analysis
    feature_frequency = {}
    feature_avg_importance = {}
    clinical_categories = {
        'Operational': [],
        'Clinical': [],
        'Temporal': [],
        'Staffing': [],
        'Patient': []
    }
    
    for config_name, config_data in all_results.items():
        for model_name, model_result in config_data['training_results'].items():
            if 'feature_importance' in model_result['metrics']:
                # Skip base learners from comprehensive ensemble - only show distinct models
                clean_model_name = model_name.replace('_regression', '')
                if config_name == 'comprehensive_ensemble' and clean_model_name != 'stacking':
                    continue  # Skip duplicated base learners
                    
                importance_data = model_result['metrics']['feature_importance']
                
                # Handle stacking model even if it has no feature importance
                if 'top_features' in importance_data or clean_model_name == 'stacking':
                    print(f"\n📊 {model_name.upper().replace('_', ' ')} ({config_name.replace('_', ' ').title()})")
                    print(f"Configuration Focus: {config_data['config_metadata']['focus']}")
                    print("-" * 50)
                    
                    # Special handling for stacking ensemble
                    if clean_model_name == 'stacking' and (not importance_data.get('top_features') or len(importance_data.get('top_features', [])) == 0):
                        print("   📋 Ensemble Architecture:")
                        print("      • Base Learners: Ridge, XGBoost, Random Forest")
                        print("      • Meta-learner: Ridge regression for optimal combination")
                        print("      • Feature Processing: Inherits from individual base models")
                        print("      • Prediction Strategy: Weighted combination of base predictions")
                        print("\n   💡 Clinical Workflow Insights:")
                        print("      • Combines linear interpretability with tree-based pattern recognition")
                        print("      • Robust predictions through algorithmic diversity and cross-validation")
                        print("      • Optimal for critical healthcare operations requiring maximum accuracy")
                        
                        # Skip the normal feature iteration for stacking without features
                        continue
                    
                    # Display top features with clinical interpretation for models with features
                    if importance_data.get('top_features'):
                        for i, feature in enumerate(importance_data['top_features'][:8], 1):
                            feat_name = feature.get('feature', 'Unknown')
                            importance = feature.get('importance_pct', feature.get('shap_importance_pct', 0))
                            
                            # Clean feature name for clinical presentation
                            clean_name = feat_name.replace('_', ' ').title()
                            if 'delay_' in feat_name.lower():
                                clean_name = feat_name.replace('delay_', '').replace('_', ' → ').title() + ' Delay'
                            elif 'lab_' in feat_name.lower():
                                clean_name = feat_name.replace('lab_', '').upper() + ' Lab Value'
                            
                            print(f"   {i:2d}. {clean_name:<35} {importance:>6.1f}%")
                            
                            # Track feature frequency and importance for aggregation
                            if feat_name not in feature_frequency:
                                feature_frequency[feat_name] = 0
                                feature_avg_importance[feat_name] = []
                            feature_frequency[feat_name] += 1
                            feature_avg_importance[feat_name].append(importance)
                            
                            # Categorize features for workflow analysis
                            if any(term in feat_name.lower() for term in ['queue', 'occupancy', 'floor', 'shift']):
                                clinical_categories['Operational'].append((feat_name, importance))
                            elif any(term in feat_name.lower() for term in ['lab_', 'diagnosis', 'severity', 'treatment']):
                                clinical_categories['Clinical'].append((feat_name, importance))
                            elif any(term in feat_name.lower() for term in ['hour', 'day', 'month', 'time']):
                                clinical_categories['Temporal'].append((feat_name, importance))
                            elif any(term in feat_name.lower() for term in ['nurse', 'pharmacist', 'credential']):
                                clinical_categories['Staffing'].append((feat_name, importance))
                            elif any(term in feat_name.lower() for term in ['age', 'sex', 'insurance', 'readiness']):
                                clinical_categories['Patient'].append((feat_name, importance))
                    
                    # Display clinical insights for workflow optimization
                    if 'clinical_insights' in model_result['metrics']:
                        insights = model_result['metrics']['clinical_insights']
                        if insights:
                            print(f"\n💡 Clinical Workflow Insights:")
                            for insight in insights[:3]:
                                print(f"      • {insight}")
    
    # Generate aggregated feature importance analysis
    print(f"\n🔍 AGGREGATED FEATURE IMPORTANCE ACROSS ALL MODELS")
    print("-" * 70)
    
    # Calculate average importance for frequently appearing features
    important_features = []
    for feat_name, importance_list in feature_avg_importance.items():
        if len(importance_list) >= 2:  # Appears in multiple models
            avg_importance = sum(importance_list) / len(importance_list)
            frequency = feature_frequency[feat_name]
            important_features.append((feat_name, avg_importance, frequency))
    
    # Sort by average importance and display top consistent predictors
    important_features.sort(key=lambda x: x[1], reverse=True)
    
    print(f"🏆 TOP CONSISTENT PREDICTORS ACROSS MODELS:")
    print(f"{'Feature':<40} | {'Avg Importance':<15} | {'Model Frequency'}")
    print(f"{'-'*40} | {'-'*15} | {'-'*14}")
    
    for feat_name, avg_importance, frequency in important_features[:10]:
        clean_name = feat_name.replace('_', ' ').title()
        if 'delay_' in feat_name.lower():
            clean_name = feat_name.replace('delay_', '').replace('_', ' → ').title() + ' Delay'
        
        print(f"{clean_name:<40} | {avg_importance:>6.1f}%        | {frequency}/{len(all_results)+1} models")
    
    # Clinical category analysis for targeted workflow improvements
    print(f"\n⚙️  WORKFLOW CATEGORY IMPACT ANALYSIS")
    print("-" * 50)
    
    for category, features in clinical_categories.items():
        if features:
            avg_category_importance = sum(importance for _, importance in features) / len(features)
            print(f"{category} Features: {avg_category_importance:.1f}% average importance ({len(features)} features)")
            
            # Show top feature in each category
            if features:
                top_feature = max(features, key=lambda x: x[1])
                clean_name = top_feature[0].replace('_', ' ').title()
                print(f"   • Key {category} Factor: {clean_name} ({top_feature[1]:.1f}%)")
    
    # Healthcare stakeholder recommendations based on feature analysis
    print(f"\n💡 EVIDENCE-BASED PHARMACY WORKFLOW RECOMMENDATIONS:")
    print("-" * 60)
    
    if important_features:
        top_feature = important_features[0]
        print(f"1. Primary Focus Area: {top_feature[0].replace('_', ' ').title()}")
        print(f"   • Consistently high importance ({top_feature[1]:.1f}%) across models")
        print(f"   • Appears in {top_feature[2]}/{len(all_results)} model configurations")
    
    # Category-specific recommendations
    operational_features = clinical_categories['Operational']
    if operational_features:
        top_operational = max(operational_features, key=lambda x: x[1])
        print(f"2. Operational Optimization: Focus on {top_operational[0].replace('_', ' ').title()}")
        print(f"   • High operational impact ({top_operational[1]:.1f}% importance)")
    
    staffing_features = clinical_categories['Staffing']
    if staffing_features:
        top_staffing = max(staffing_features, key=lambda x: x[1])
        print(f"3. Staffing Strategy: Monitor {top_staffing[0].replace('_', ' ').title()}")
        print(f"   • Significant staffing impact ({top_staffing[1]:.1f}% importance)")
    
    print(f"4. Continuous Monitoring: Track top predictors for ongoing optimization")
    print(f"5. Targeted Training: Develop staff protocols for high-impact workflow areas")
    
    logger.info("✅ Clinical feature importance analysis completed successfully")

def save_comprehensive_results(all_results: dict) -> None:
    """
    Save comprehensive training results and deployment metadata for MLOps integration.
    
    Creates production-ready artifacts supporting model deployment, monitoring, and
    maintenance in healthcare environments. Essential for MLOps workflows enabling
    automated model management and continuous improvement in pharmacy operations
    through comprehensive artifact preservation and deployment preparation.
    
    Args:
        all_results: Comprehensive training results across configurations including
                    model performance, artifacts, and metadata supporting healthcare
                    analytics deployment and pharmacy workflow optimization initiatives.
    
    Note:
        Critical for healthcare MLOps enabling production model deployment and
        continuous monitoring supporting pharmacy workflow optimization and medication
        preparation efficiency through comprehensive artifact management and deployment preparation.
    """
    logger.info("💾 Saving comprehensive training results for MLOps deployment...")
    
    # Create comprehensive deployment metadata for healthcare analytics
    deployment_metadata = {
        'project_info': {
            'name': 'TAT Prediction System',
            'objective': 'Pharmacy workflow optimization through predictive analytics',
            'healthcare_context': 'Medication preparation turnaround time prediction',
            'clinical_threshold': '60-minute TAT compliance requirement',
            'target_application': 'Real-time workflow optimization and bottleneck identification'
        },
        'training_summary': {
            'configurations_trained': list(all_results.keys()),
            'total_models_developed': sum(
                len(config['training_results']) for config in all_results.values()
            ),
            'dataset_size': next(iter(all_results.values()))['dataset_shape'],
            'training_date': pd.Timestamp.now().isoformat(),
            'healthcare_focus_areas': [
                'Medication preparation workflow optimization',
                'TAT threshold compliance monitoring',
                'Bottleneck identification and resolution',
                'Resource allocation optimization'
            ]
        },
        'deployment_specifications': {
            'primary_task': 'regression',
            'threshold_monitoring': 60,  # minutes
            'clinical_metrics': [
                'RMSE (minutes)',
                'MAE (minutes)', 
                'Threshold compliance accuracy',
                'Workflow optimization score'
            ],
            'recommended_model_refresh': 'monthly',
            'monitoring_requirements': [
                'Model performance drift detection',
                'Feature distribution monitoring',
                'Clinical threshold compliance tracking',
                'Healthcare outcome impact assessment'
            ]
        },
        'clinical_applications': {
            'real_time_prediction': 'TAT forecasting for incoming medication orders',
            'workflow_optimization': 'Bottleneck identification and resource allocation',
            'quality_monitoring': 'TAT threshold compliance assessment and reporting',
            'decision_support': 'Evidence-based pharmacy operations improvement'
        }
    }
    
    # Save comprehensive training artifacts for production deployment
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("💼 Saving production deployment artifacts...")
    joblib.dump(all_results, MODEL_DIR / "complete_training_results.joblib")
    joblib.dump(deployment_metadata, MODEL_DIR / "deployment_metadata.joblib")
    
    # Create deployment summary for healthcare stakeholders
    deployment_summary = {
        'recommended_production_model': None,
        'performance_summary': {},
        'clinical_insights': {},
        'deployment_readiness': True
    }
    
    # Identify best performing model for production recommendation
    best_rmse = float('inf')
    best_config = None
    best_model = None
    
    for config_name, config_data in all_results.items():
        for model_name, model_result in config_data['training_results'].items():
            if 'RMSE' in model_result['metrics']:
                rmse = model_result['metrics']['RMSE']
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_config = config_name
                    best_model = model_name
                    
                    deployment_summary['recommended_production_model'] = {
                        'configuration': config_name,
                        'model_type': model_name,
                        'performance': {
                            'rmse_minutes': rmse,
                            'mae_minutes': model_result['metrics']['MAE'],
                            'threshold_accuracy': model_result['metrics'].get('threshold_60min_accuracy', 0)
                        },
                        'clinical_focus': config_data['config_metadata']['focus']
                    }
    
    # Save deployment summary for healthcare operations
    joblib.dump(deployment_summary, MODEL_DIR / "deployment_summary.joblib")
    
    # Create human-readable deployment documentation
    deployment_doc = f"""
TAT PREDICTION MODEL DEPLOYMENT PACKAGE
========================================

Project: Medication Preparation TAT Prediction
Healthcare Application: Pharmacy Workflow Optimization
Clinical Threshold: 60-minute TAT Compliance

RECOMMENDED PRODUCTION MODEL:
-----------------------------
Configuration: {best_config}
Model Type: {best_model}
Performance: {best_rmse:.2f} minutes RMSE
Healthcare Focus: {deployment_summary['recommended_production_model']['clinical_focus'] if deployment_summary['recommended_production_model'] else 'N/A'}

DEPLOYMENT ARTIFACTS:
--------------------
• complete_training_results.joblib - Full training results and models
• deployment_metadata.joblib - Healthcare deployment specifications
• deployment_summary.joblib - Production model recommendations
• model_comparison.csv - Performance analysis across configurations
• feature_importance_summary.json - Clinical feature insights

NEXT STEPS:
-----------
1. Review recommended model performance and clinical applicability
2. Deploy selected model configuration for real-time TAT prediction
3. Establish monitoring for model performance and healthcare impact
4. Schedule regular model retraining based on operational data updates

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(MODEL_DIR / "DEPLOYMENT_README.txt", 'w') as f:
        f.write(deployment_doc)
    
    logger.info(f"✅ Comprehensive training artifacts saved to {MODEL_DIR}")
    logger.info(f"   • Training results: complete_training_results.joblib")
    logger.info(f"   • Deployment metadata: deployment_metadata.joblib")
    logger.info(f"   • Production summary: deployment_summary.joblib")
    logger.info(f"   • Documentation: DEPLOYMENT_README.txt")
    
    print(f"\n📦 MLOPS DEPLOYMENT PACKAGE READY:")
    print(f"   • Location: {MODEL_DIR}")
    print(f"   • Recommended Model: {best_model} ({best_config})")
    print(f"   • Performance: {best_rmse:.2f} min RMSE")
    print(f"   • Documentation: DEPLOYMENT_README.txt")

if __name__ == "__main__":
    """
    Execute comprehensive TAT prediction model training for pharmacy workflow optimization.
    
    Provides command-line interface for healthcare analytics supporting medication
    preparation TAT analysis and pharmacy operations improvement through evidence-based
    machine learning model development and deployment preparation.
    """
    results = train_tat_models()