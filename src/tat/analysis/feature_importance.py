"""
Feature importance analysis using SHAP for model interpretability.
"""
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for TAT prediction models using SHAP.
    
    Provides interpretability insights to identify key factors affecting
    medication preparation turnaround times.
    
    Parameters:
        model: Trained ML model for TAT prediction
        X_train: Training feature matrix for SHAP baseline calculation
    
    Attributes:
        model: Trained prediction model instance
        X_train: Training data for explainer initialization
        feature_names: List of feature column names for interpretation
        model_type: Detected model type for appropriate importance extraction
    """
    
    def __init__(self, model, X_train: pd.DataFrame):
        """
        Initialize analyzer with trained TAT prediction model and training data.
        
        Automatically detects model type and prepares appropriate feature importance
        extraction methodology for pharmacy workflow analysis.
        
        Args:
            model: Trained ML model (XGBoost, RandomForest, LinearRegression, etc.)
            X_train: Training feature matrix for SHAP baseline computation
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = X_train.columns.tolist()
        
        # Detect model architecture for optimal importance extraction strategy
        self.model_type = self._detect_model_type()
        
    def _detect_model_type(self) -> str:
        """
        Detect ML model architecture for appropriate importance extraction.
        
        Supports tree-based models (XGBoost, RandomForest), linear models (Ridge, Logistic),
        and ensemble methods commonly used in healthcare TAT prediction systems.
        
        Returns:
            str: Model type identifier for importance analysis strategy selection
        """
        model_class = self.model.__class__.__name__
        if 'XGB' in model_class:
            return 'xgboost'
        elif 'RandomForest' in model_class:
            return 'random_forest'
        elif 'Ridge' in model_class or 'Linear' in model_class:
            return 'linear'
        elif 'Stacking' in model_class:
            return 'ensemble'
        else:
            return 'unknown'
    
    def get_basic_importance(self) -> Dict[str, Any]:
        """
        Extract native feature importance from ML models for pharmacy workflow analysis.
        
        Provides fast, model-native importance scoring suitable for production TAT
        monitoring systems and clinical team review. Fallback method when SHAP
        analysis is unavailable or computationally intensive for real-time inference.
        
        Supports:
        - Tree-based models: feature_importances_ (Gini/entropy-based scoring)
        - Linear models: coefficient magnitude analysis with clinical interpretation
        - Ensemble models: aggregated importance from base estimators
        
        Returns:
            Dict containing:
            - method: Importance extraction methodology used
            - model_type: Detected model architecture
            - top_features: Ranked feature importance list for clinical review
            - feature_importance_available: Boolean success indicator
            - top_10_cumulative_importance: Concentration metric for workflow focus
            
        Note:
            Optimized for healthcare production environments with <100ms latency requirement.
        """
        importance_results = {
            'method': 'basic_feature_importance',
            'model_type': self.model_type,
            'top_features': [],
            'feature_importance_available': False
        }
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Tree-based models (XGBoost, RandomForest) - native importance scoring
                importances = self.model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances,
                    'importance_pct': importances / importances.sum() * 100
                }).sort_values('importance', ascending=False)
                
                # Focus on top features for clinical interpretability and workflow optimization
                top_features = importance_df.head(15).to_dict('records')
                
                importance_results.update({
                    'top_features': top_features,
                    'feature_importance_available': True,
                    'total_features': len(self.feature_names),
                    'top_10_cumulative_importance': importance_df.head(10)['importance_pct'].sum()
                })
                
                logger.info(f"Native feature importance extracted for {self.model_type} model")
                
            elif hasattr(self.model, 'coef_'):
                # Linear models (Ridge) - coefficient magnitude analysis
                coefficients = np.abs(self.model.coef_)
                if coefficients.ndim > 1:
                    coefficients = coefficients[0]  
                
                coef_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'abs_coefficient': coefficients,
                    'coefficient_pct': coefficients / coefficients.sum() * 100
                }).sort_values('abs_coefficient', ascending=False)
                
                top_features = coef_df.head(15).to_dict('records')
                
                importance_results.update({
                    'top_features': top_features,
                    'feature_importance_available': True,
                    'method': 'linear_coefficients',
                    'total_features': len(self.feature_names),
                    'top_10_cumulative_importance': coef_df.head(10)['coefficient_pct'].sum()
                })
                
                logger.info(f"Linear coefficient importance extracted for {self.model_type} model")
                
            elif hasattr(self.model, 'final_estimator_') and hasattr(self.model.final_estimator_, 'coef_'):
                # Stacking ensemble models - extract importance from meta-learner coefficients
                meta_coefs = np.abs(self.model.final_estimator_.coef_)
                if meta_coefs.ndim > 1:
                    meta_coefs = meta_coefs[0]
                
                # For stacking need to map back to original features through base models
                # Since base models output predictions that become meta-learner inputs,
                # we'll use SHAP on the full ensemble instead of trying to decompose
                try:
                    import shap
                    # Use TreeExplainer for ensemble models
                    explainer = shap.Explainer(self.model.predict, self.X_train.sample(100))
                    shap_values = explainer(self.X_train.sample(500))
                    
                    # Calculate mean absolute SHAP values for feature importance
                    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
                    
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'mean_abs_shap': mean_abs_shap,
                        'shap_importance_pct': mean_abs_shap / mean_abs_shap.sum() * 100
                    }).sort_values('mean_abs_shap', ascending=False)
                    
                    top_features = importance_df.head(15).to_dict('records')
                    
                    importance_results.update({
                        'top_features': top_features,
                        'feature_importance_available': True,
                        'method': 'ensemble_shap',
                        'total_features': len(self.feature_names),
                        'top_10_cumulative_importance': importance_df.head(10)['shap_importance_pct'].sum()
                    })
                    
                    logger.info(f"Ensemble SHAP importance extracted for {self.model_type} model")
                    
                except ImportError:
                    logger.warning("SHAP not available for ensemble model - using base estimator average")
                    # Fallback: average importance from base estimators if available
                    base_importances = []
                    for name, estimator in self.model.estimators_:
                        if hasattr(estimator, 'feature_importances_'):
                            base_importances.append(estimator.feature_importances_)
                        elif hasattr(estimator, 'coef_'):
                            base_importances.append(np.abs(estimator.coef_[0] if estimator.coef_.ndim > 1 else estimator.coef_))
                    
                    if base_importances:
                        avg_importance = np.mean(base_importances, axis=0)
                        
                        importance_df = pd.DataFrame({
                            'feature': self.feature_names,
                            'avg_importance': avg_importance,
                            'importance_pct': avg_importance / avg_importance.sum() * 100
                        }).sort_values('avg_importance', ascending=False)
                        
                        top_features = importance_df.head(15).to_dict('records')
                        
                        importance_results.update({
                            'top_features': top_features,
                            'feature_importance_available': True,
                            'method': 'ensemble_average',
                            'total_features': len(self.feature_names),
                            'top_10_cumulative_importance': importance_df.head(10)['importance_pct'].sum()
                        })
                        
                        logger.info(f"Ensemble average importance extracted for {self.model_type} model")
                
        except Exception as e:
            logger.warning(f"Could not extract native feature importance: {e}")
            
        return importance_results
    
    def shap_summary(self, X_test: pd.DataFrame, max_display: int = 15) -> Dict[str, Any]:
        """
        Generate comprehensive SHAP-based feature importance analysis for TAT prediction models.
        
        Provides detailed model interpretability analysis to help pharmacy operations teams
        understand prediction drivers and identify high-impact workflow optimization opportunities.
        Uses SHAP (SHapley Additive exPlanations) values for model-agnostic feature attribution.
        
        Key Analysis Components:
        - Individual prediction explanations for clinical case review
        - Global feature importance ranking for workflow prioritization
        - Clinical insights generation for pharmacy leadership consumption
        - Automated visualization for stakeholder reporting and dashboard integration
        
        Args:
            X_test: Test feature matrix for SHAP value computation
            max_display: Maximum features to analyze (default: 15 for clinical focus)
            
        Returns:
            Dict containing:
            - method: 'shap_analysis' methodology identifier
            - model_type: ML architecture used for TAT prediction
            - top_features: Ranked SHAP importance with clinical interpretation
            - clinical_insights: Actionable recommendations for pharmacy workflow optimization
            - shap_computation_successful: Analysis completion indicator
            - plot_saved: File path for generated SHAP visualization (dashboard integration)
            
        Note:
            Computationally intensive - recommended for batch analysis and model development.
            Production inference should use get_basic_importance() for real-time requirements.
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP library not available. Falling back to native feature importance.")
            return self.get_basic_importance()
        
        shap_results = {
            'method': 'shap_analysis',
            'model_type': self.model_type,
            'shap_available': True,
            'top_features': [],
            'clinical_insights': []
        }
        
        try:
            # Optimize computation for healthcare production constraints (memory/latency)
            sample_size = min(500, len(X_test))
            X_test_sample = X_test.sample(n=sample_size, random_state=42)
            X_train_sample = self.X_train.sample(n=min(500, len(self.X_train)), random_state=42)
            
            logger.info(f"Computing SHAP values for {self.model_type} TAT model (sample size: {sample_size})")
            
            # Initialize model-specific explainer for optimal performance
            # Suppress SHAP's internal numpy random seed warning
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "The NumPy global RNG was seeded by calling", FutureWarning)
                
                if self.model_type in ['xgboost', 'random_forest']:
                    # Tree-based models: use efficient TreeExplainer for pharmacy production systems
                    explainer = shap.TreeExplainer(self.model)
                    shap_values = explainer.shap_values(X_test_sample)
                else:
                    # Linear/ensemble models: general explainer with background dataset
                    explainer = shap.Explainer(self.model, X_train_sample)
                    shap_values = explainer(X_test_sample)
                    if hasattr(shap_values, 'values'):
                        shap_values = shap_values.values
            
            # Compute global feature importance from individual SHAP attributions
            if isinstance(shap_values, list):
                # Multi-class case: use primary class for TAT binary classification
                mean_abs_shap = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Generate feature importance ranking for clinical workflow prioritization
            shap_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'mean_abs_shap': mean_abs_shap,
                'shap_importance_pct': mean_abs_shap / mean_abs_shap.sum() * 100
            }).sort_values('mean_abs_shap', ascending=False)
            
            # Extract top features for pharmacy leadership review and intervention planning
            top_features = shap_importance_df.head(max_display).to_dict('records')
            
            # Generate actionable clinical insights for workflow optimization
            clinical_insights = self._generate_clinical_insights(shap_importance_df.head(10))
            
            shap_results.update({
                'top_features': top_features,
                'clinical_insights': clinical_insights,
                'total_features_analyzed': len(self.feature_names),
                'sample_size_used': sample_size,
                'top_10_cumulative_importance': shap_importance_df.head(10)['shap_importance_pct'].sum(),
                'shap_computation_successful': True
            })
            
            # Generate and save SHAP visualization for stakeholder reporting
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend for automated model training pipelines
                
                from pathlib import Path
                
                # Create standardized reports directory structure for healthcare analytics
                reports_dir = Path("reports/figures")
                feature_importance_dir = reports_dir
                feature_importance_dir.mkdir(parents=True, exist_ok=True)
                
                plt.figure(figsize=(12, 8))
                # Suppress SHAP's internal numpy random seed warning during plot generation
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "The NumPy global RNG was seeded by calling", FutureWarning)
                    
                    if isinstance(shap_values, list):
                        shap.summary_plot(shap_values[0], X_test_sample, 
                                        feature_names=self.feature_names,
                                        max_display=max_display, show=False)
                    else:
                        shap.summary_plot(shap_values, X_test_sample, 
                                        feature_names=self.feature_names,
                                        max_display=max_display, show=False)
                
                plt.title(f'TAT Prediction Model - Feature Importance Analysis\n'
                        f'Model: {self.model_type.upper()} | Pharmacy Workflow Optimization', 
                        fontsize=14, pad=20)
                plt.tight_layout()
                
                # Save with healthcare analytics naming convention for dashboard integration
                plot_filename = f'tat_shap_analysis_{self.model_type}.png'
                plot_path = feature_importance_dir / plot_filename
                
                plt.savefig(plot_path, dpi=150, bbox_inches='tight', 
                        facecolor='white', edgecolor='none')
                plt.close()
                
                shap_results['plot_saved'] = str(plot_path)
                shap_results['reports_directory'] = str(feature_importance_dir)
                
                logger.info(f"TAT SHAP analysis visualization saved to: {plot_path}")
                
            except Exception as plot_error:
                logger.warning(f"Could not save SHAP visualization to reports directory: {plot_error}")
            
            logger.info(f"SHAP analysis completed for TAT prediction model ({self.model_type})")
            
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}. Falling back to native importance extraction.")
            return self.get_basic_importance()
        
        return shap_results
    
    def _generate_clinical_insights(self, top_features_df: pd.DataFrame) -> List[str]:
        """
        Generate actionable clinical insights from feature importance for pharmacy operations teams.
        
        Translates ML feature importance scores into specific, actionable recommendations
        for medication preparation workflow optimization and TAT improvement initiatives.
        Designed for healthcare leadership consumption and operational decision-making.
        
        Args:
            top_features_df: Top-ranked features with importance scores for analysis
            
        Returns:
            List of clinical insights with specific operational recommendations:
            - Queue management and capacity planning insights
            - Staffing optimization recommendations by role and shift
            - Laboratory value impact analysis for clinical protocol adjustment
            - Temporal pattern identification for resource allocation optimization
            - Patient acuity workflow adjustments for complex case management
        """
        insights = []
        
        for _, row in top_features_df.head(5).iterrows():
            feature = row['feature']
            importance_pct = row['shap_importance_pct']
            
            # Generate feature-specific operational insights for pharmacy workflow optimization
            if 'queue_length' in feature.lower():
                insights.append(f"Queue management optimization critical: {feature} accounts for {importance_pct:.1f}% of TAT variation")
            elif 'floor_occupancy' in feature.lower():
                insights.append(f"Floor capacity planning impact: {feature} contributes {importance_pct:.1f}% to TAT predictions")
            elif 'pharmacist' in feature.lower():
                insights.append(f"Pharmacist staffing optimization: {feature} represents {importance_pct:.1f}% of TAT prediction drivers")
            elif 'lab_' in feature.lower():
                insights.append(f"Laboratory value integration: {feature} influences {importance_pct:.1f}% of TAT outcome predictions")
            elif 'hour' in feature.lower() or 'shift' in feature.lower():
                insights.append(f"Temporal workflow patterns: {feature} affects {importance_pct:.1f}% of TAT variation")
            elif 'severity' in feature.lower():
                insights.append(f"Patient acuity workflow impact: {feature} drives {importance_pct:.1f}% of TAT differences")
            else:
                insights.append(f"Critical operational factor: {feature} contributes {importance_pct:.1f}% to TAT prediction accuracy")
        
        # Add strategic summary insight for pharmacy leadership decision-making
        top_5_cumulative = top_features_df.head(5)['shap_importance_pct'].sum()
        insights.append(f"Top 5 operational factors explain {top_5_cumulative:.1f}% of TAT variation - prioritize these areas for maximum workflow improvement impact")
        
        return insights
    
    def analyze_importance(self, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Main entry point for feature importance analysis with automatic method selection.
        
        Provides backwards-compatible interface for existing TAT prediction pipelines.
        Automatically selects optimal analysis method based on available resources:
        - SHAP analysis (preferred): Comprehensive model interpretability with clinical insights
        - Native importance (fallback): Fast model-specific importance for production systems
        
        Args:
            X_test: Test feature matrix for importance computation
            
        Returns:
            Dict: Feature importance analysis results with clinical interpretation
        """
        if SHAP_AVAILABLE:
            return self.shap_summary(X_test)
        else:
            return self.get_basic_importance()
    
    def get_top_features_for_clinical_review(self, X_test: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Generate clinical team-focused feature importance summary for pharmacy workflow review.
        
        Formats feature importance analysis results for healthcare stakeholder consumption.
        Provides simplified, actionable summary suitable for pharmacy leadership meetings
        and operational improvement planning sessions.
        
        Args:
            X_test: Test dataset for importance analysis
            top_n: Number of top features to include in clinical summary (default: 10)
            
        Returns:
            pd.DataFrame: Formatted feature importance summary with:
            - feature: Original feature name for technical reference
            - Clinical_Category: Healthcare domain classification for workflow mapping
            - Impact_Score/Impact_Percentage: Quantified importance metrics for prioritization
            - Clinical interpretation optimized for non-technical pharmacy leadership
        """
        importance_results = self.analyze_importance(X_test)
        
        if importance_results.get('top_features'):
            top_features_df = pd.DataFrame(importance_results['top_features']).head(top_n)
            
            # Rename technical columns for clinical clarity and stakeholder understanding
            column_mapping = {
                'importance': 'Impact_Score',
                'importance_pct': 'Impact_Percentage',
                'mean_abs_shap': 'SHAP_Impact_Score',
                'shap_importance_pct': 'SHAP_Impact_Percentage'
            }
            
            top_features_df = top_features_df.rename(columns=column_mapping)
            
            # Add clinical domain categorization for workflow optimization planning
            top_features_df['Clinical_Category'] = top_features_df['feature'].apply(self._categorize_feature_clinically)
            
            return top_features_df[['feature', 'Clinical_Category'] + [col for col in top_features_df.columns if 'Impact' in col]]
        
        return pd.DataFrame()
    
    def _categorize_feature_clinically(self, feature: str) -> str:
        """
        Categorize ML features into clinical domains for pharmacy workflow optimization.
        
        Maps technical feature names to healthcare operational categories that align
        with pharmacy team responsibilities and workflow improvement initiatives.
        Supports strategic planning and resource allocation for TAT optimization.
        
        Args:
            feature: Technical feature name from ML model
            
        Returns:
            str: Clinical category for operational workflow mapping:
            - 'Operations & Staffing': Queue management, occupancy, pharmacist scheduling
            - 'Laboratory Values': Clinical indicators affecting medication preparation complexity
            - 'Temporal Patterns': Time-based workflow variations for shift planning
            - 'Patient Acuity': Severity and treatment complexity factors
            - 'Clinical Staffing': Nurse credentials and experience impact on workflow
            - 'Location & Workflow': Physical and departmental logistics factors
        """
        feature_lower = feature.lower()
        
        if any(x in feature_lower for x in ['queue', 'occupancy', 'pharmacist']):
            return 'Operations & Staffing'
        elif any(x in feature_lower for x in ['lab_', 'wbc', 'hgb', 'platelet', 'creatinine']):
            return 'Laboratory Values'
        elif any(x in feature_lower for x in ['hour', 'shift', 'day', 'month']):
            return 'Temporal Patterns'
        elif any(x in feature_lower for x in ['severity', 'diagnosis', 'treatment']):
            return 'Patient Acuity'
        elif any(x in feature_lower for x in ['nurse', 'credential', 'employment']):
            return 'Clinical Staffing'
        elif any(x in feature_lower for x in ['floor', 'department']):
            return 'Location & Workflow'
        else:
            return 'Other Clinical Factors'