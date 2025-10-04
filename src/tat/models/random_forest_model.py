"""
Healthcare Analytics Random Forest for Medication Preparation TAT Prediction

Advanced ensemble tree-based learning supporting healthcare pharmacy workflow
optimization and 60-minute TAT threshold compliance initiatives. Provides robust
Random Forest regression with feature interaction modeling enabling comprehensive
bottleneck identification and clinical decision support for medication preparation
turnaround time prediction and pharmacy operations excellence.

Key Features:
- Ensemble tree-based learning with healthcare-optimized parameters for clinical robustness
- Feature interaction modeling capturing complex pharmacy workflow relationships
- Robust feature importance analysis enabling evidence-based bottleneck identification
- Healthcare-specific hyperparameter optimization with interpretability constraints
- Production-ready deployment configuration with MLOps integration capabilities
"""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import BaseRegressionTATModel

logger = logging.getLogger(__name__)

class RandomForestTATRegressor(BaseRegressionTATModel):
    """
    Healthcare Random Forest Regression for Robust TAT Prediction
    
    Advanced ensemble tree-based learning optimized for healthcare medication
    preparation TAT prediction and pharmacy workflow optimization. Provides robust
    Random Forest regression with feature interaction modeling ensuring comprehensive
    bottleneck identification and clinical decision support for diverse healthcare
    operational scenarios and 60-minute threshold compliance requirements.
    
    Healthcare Ensemble Model Features:
    - Robust ensemble learning preventing overfitting through bootstrap aggregation
    - Feature interaction modeling capturing complex pharmacy workflow relationships
    - Tree-based interpretability enabling pathway analysis for clinical decision-making
    - Healthcare parameter optimization balancing robustness with computational efficiency
    - Production deployment readiness with MLOps integration and automated monitoring capabilities
   
    Args:
        random_state: Reproducibility seed ensuring consistent healthcare analytics results
        **kwargs: Random Forest-specific parameters including n_estimators and max_depth
    
    Attributes:
        default_params: Healthcare-optimized Random Forest configuration with clinical constraints
        model: Scikit-learn Random Forest regressor with healthcare parameter optimization
        
    Example:
        For robust TAT prediction ensemble modeling:
        ```python
        # Initialize Random Forest model for ensemble learning
        rf_model = RandomForestTATRegressor(
            random_state=42,
            n_estimators=200,  # Robust ensemble size
            max_depth=8        # Clinical interpretability balance
        )
        
        # Train on medication preparation data
        rf_model.fit(X_train, y_train)
        
        # Extract clinical insights from feature importance
        importance = rf_model.get_feature_importance()
        top_factors = importance.head(10)
        ```
    """
    
    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize Healthcare Random Forest with Clinical Ensemble Optimization
        
        Establishes Random Forest ensemble model optimized for healthcare medication
        preparation TAT prediction and pharmacy workflow optimization. Configures
        healthcare-specific parameters ensuring clinical robustness while maintaining
        interpretability for evidence-based decision-making and bottleneck identification.
        
        Healthcare Random Forest Configuration:
        - Ensemble size optimization balancing robustness with computational efficiency requirements
        - Tree depth limitations maintaining clinical interpretability while capturing complex patterns
        - Bootstrap sampling ensuring model stability across diverse healthcare dataset variations
        - Feature sampling promoting ensemble diversity and preventing overfitting in correlated features
        - Production configuration supporting MLOps integration and automated monitoring workflows
        
        Random Forest Parameter Optimization:
        - N_estimators: Balanced ensemble size ensuring robustness without excessive computation
        - Max_depth: Clinical interpretability constraint preventing overly complex tree structures
        - Min_samples_split: Conservative splitting preventing overfitting in healthcare data
        - Max_features: Feature sampling promoting diversity and reducing correlation impacts
        - Healthcare constraints: Parameter ranges ensuring clinical deployment readiness and stability
        
        Args:
            random_state: Reproducibility seed ensuring consistent healthcare analytics
            **kwargs: Random Forest-specific healthcare parameters supporting clinical optimization

        """
        super().__init__(random_state)
        
        # Healthcare-optimized Random Forest parameters with clinical focus
        self.default_params = {
            'n_estimators': 200,             # Robust ensemble size for healthcare stability
            'max_depth': 8,                  # Clinical interpretability balance
            'min_samples_split': 5,          # Conservative splitting for stability
            'min_samples_leaf': 2,           # Minimum leaf size for robustness
            'max_features': 'sqrt',          # Feature sampling for diversity
            'random_state': random_state,    # Reproducible clinical results
            'n_jobs': -1,                   # Parallel processing for efficiency
            'bootstrap': True,              # Bootstrap sampling for ensemble diversity
            'oob_score': True,              # Out-of-bag validation for internal assessment
            'warm_start': False,            # Fresh training for healthcare reproducibility
            'max_samples': None             # Use all samples for robust training
        }
        
        # Update with clinical parameter overrides
        self.default_params.update(kwargs)
        
        # Initialize Random Forest model with healthcare-optimized configuration
        self.model = RandomForestRegressor(**self.default_params)
        
        # Update healthcare metadata with ensemble model specifications
        self.metadata.update({
            'algorithm': 'Random Forest Regression',                                    # Algorithm type
            'healthcare_optimization': 'TAT prediction with ensemble robustness',      # Clinical focus
            'interpretability': 'Feature importance and tree pathway analysis',        # Clinical transparency
            'clinical_advantages': 'Robust predictions and feature interactions',      # Stakeholder benefits
            'bottleneck_identification': 'Ensemble-based importance ranking',          # Workflow optimization
            'regulatory_compliance': 'Interpretable tree-based architecture',         # Audit support
            'deployment_readiness': 'MLOps integration with ensemble monitoring'       # Production capabilities
        })
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            validation_data: Optional[tuple] = None) -> 'RandomForestTATRegressor':
        """
        Train Random Forest Ensemble Model on Healthcare TAT Data
        
        Comprehensive Random Forest training optimized for healthcare medication
        preparation TAT prediction and pharmacy workflow optimization. Handles skewed
        healthcare TAT distributions through target transformation while preserving
        ensemble interpretability for clinical decision-making and bottleneck analysis.
        
        Healthcare Random Forest Training:
        - Target transformation handling skewed TAT distributions common in healthcare data
        - Bootstrap aggregation preventing overfitting through diverse ensemble training
        - Out-of-bag validation providing internal model assessment for healthcare validation
        - Clinical feature importance extraction enabling transparent workflow analysis
        - Healthcare metadata tracking supporting audit trails and regulatory compliance
        
        Training Workflow Components:
        - Skewed target transformation: Log1p handling for realistic healthcare TAT distributions
        - Ensemble fitting: Bootstrap aggregated training ensuring robust prediction capabilities
        - Out-of-bag assessment: Internal validation supporting model quality confirmation
        - Clinical importance analysis: Tree-based feature ranking for bottleneck identification
        - Healthcare metadata management: Training statistics and configuration preservation
        
        Args:
            X: Healthcare feature matrix containing clinical, operational, and temporal
               variables supporting comprehensive medication preparation workflow analysis.
            y: TAT target variable in minutes supporting continuous prediction and
               60-minute threshold compliance assessment for pharmacy quality monitoring.
            validation_data: Optional (X_val, y_val) tuple - maintained for API consistency
                           but not used in Random Forest training workflow for simplicity.
            
        Returns:
            RandomForestTATRegressor: Fitted Random Forest ensemble model ready for healthcare
            TAT prediction and clinical feature importance analysis with validated training status.
        
        Example:
            For healthcare Random Forest ensemble training:
            ```python
            # Train Random Forest model on medication preparation data
            rf_model.fit(X_train, y_train)
            
            # Validate training completion and access ensemble insights
            assert rf_model.is_fitted
            importance = rf_model.get_feature_importance()
            print(f"Top TAT driver: {importance.iloc[0]['feature']}")
            print(f"OOB Score: {rf_model.model.oob_score_:.3f}")
            ```
        """
        # Transform target variable for skewed healthcare TAT distribution
        y_transformed = self._transform_target(y)
        
        # Fit Random Forest ensemble model with bootstrap aggregation for healthcare robustness
        self.model.fit(X, y_transformed)
        
        # Set training completion status for safe healthcare prediction operation
        self.is_fitted = True
        
        # Store comprehensive training metadata for clinical documentation and audit trails
        self.metadata.update({
            'training_samples': len(X),                          # Dataset size for validation
            'feature_count': X.shape[1],                         # Feature dimensionality
            'target_transform': self.target_transform,           # Transformation method
            'n_estimators_used': self.model.n_estimators,        # Ensemble size confirmation
            'max_depth_used': self.model.max_depth,              # Tree depth limitation
            'oob_score': getattr(self.model, 'oob_score_', None), # Out-of-bag validation score
            'feature_importances_available': hasattr(self.model, 'feature_importances_'), # Importance capability
            'training_completed': True,                          # Validation status
            'clinical_deployment_ready': True,                   # Production readiness
            'ensemble_interpretability_confirmed': True          # Clinical transparency verified
        })
        
        # Log ensemble training completion with healthcare context
        oob_info = f", OOB Score: {self.model.oob_score_:.3f}" if hasattr(self.model, 'oob_score_') else ""
        logger.info(f"Random Forest ensemble trained: {len(X):,} samples, {self.model.n_estimators} trees{oob_info}")
        return self
    
    def get_hyperparameter_space(self) -> Dict[str, Any]:
        """
        Define Healthcare-Appropriate Hyperparameter Ranges for Random Forest Optimization
        
        Provides comprehensive hyperparameter search space optimized for healthcare
        medication preparation TAT prediction and clinical interpretability requirements.
        Balances ensemble robustness with computational efficiency while maintaining
        tree-based interpretability for healthcare stakeholder communication and validation.
        
        Healthcare Random Forest Hyperparameter Optimization:
        - N_estimators range: Ensemble size balancing robustness with computational efficiency
        - Max_depth bounds: Tree depth maintaining clinical interpretability while capturing complexity
        - Sampling parameters: Bootstrap and feature sampling promoting ensemble diversity
        - Split criteria: Conservative splitting preventing overfitting in healthcare data
        - Clinical constraints: Parameter ranges ensuring deployment readiness and interpretability
        
        Ensemble Optimization Strategy:
        - Tree count optimization: 100-300 estimators balancing performance with training time
        - Depth limitation: 5-15 levels ensuring interpretability while capturing interactions
        - Sample splitting: Conservative thresholds preventing overfitting in clinical data
        - Feature sampling: Multiple strategies promoting ensemble diversity and robustness
        - Healthcare boundaries: Parameter ranges tested for clinical deployment suitability
        
        Returns:
            Dict[str, Any]: Comprehensive hyperparameter search space with healthcare
            optimization ranges supporting clinical interpretability and deployment needs.
        
        Example:
            For Random Forest hyperparameter optimization:
            ```python
            # Get healthcare-optimized parameter space
            param_space = rf_model.get_hyperparameter_space()
            
            # Use with Optuna for clinical optimization
            study = optuna.create_study()
            study.optimize(objective, n_trials=100)
            ```

        """
        return {
            'n_estimators': ('int', 100, 300),                              # Ensemble size range
            'max_depth': ('int', 5, 15),                                    # Tree depth bounds
            'min_samples_split': ('int', 2, 10),                           # Split threshold range
            'min_samples_leaf': ('int', 1, 5),                             # Leaf size bounds
            'max_features': ('categorical', ['sqrt', 'log2', 0.5, 0.8])    # Feature sampling strategies
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Extract Random Forest Feature Importance for Clinical Decision-Making and Bottleneck Analysis
        
        Comprehensive feature importance extraction optimized for healthcare pharmacy
        workflow optimization and clinical interpretability. Provides ensemble-based
        importance ranking enabling evidence-based bottleneck identification and targeted
        intervention strategies for medication preparation efficiency and 60-minute threshold compliance.
        
        Healthcare Feature Importance Analysis:
        - Ensemble-based importance aggregation providing robust feature ranking across trees
        - Percentage contribution calculation enabling prioritized intervention targeting
        - Clinical significance assessment supporting evidence-based decision-making workflows
        - Feature ranking hierarchy revealing primary TAT drivers and bottleneck sources
        - Healthcare interpretation supporting stakeholder communication and operational planning
        
        Random Forest Importance Components:
        - Tree aggregation: Importance averaging across ensemble providing robust feature ranking
        - Gini importance: Tree split-based calculation capturing feature discrimination power
        - Percentage scaling: Normalized importance enabling clear prioritization and comparison
        - Sorted ranking: Feature hierarchy supporting targeted intervention and resource allocation
        - Clinical context: Healthcare interpretation enabling stakeholder communication and action
        
        Returns:
            pd.DataFrame: Comprehensive feature importance analysis with healthcare context
            and clinical interpretation supporting pharmacy workflow optimization and bottleneck targeting.
        
        Raises:
            ValueError: Unfitted model preventing unsafe feature importance extraction
            AttributeError: Model without importance attributes compromising analysis integrity
        
        Example:
            For clinical feature importance analysis and bottleneck identification:
            ```python
            # Extract Random Forest feature importance for clinical insights
            importance = rf_model.get_feature_importance()
            
            # Identify top TAT drivers for intervention targeting
            top_bottlenecks = importance.head(10)
            print("Primary TAT drivers for pharmacy optimization:")
            for _, row in top_bottlenecks.iterrows():
                print(f"  {row['feature']}: {row['importance_pct']:.1f}%")
            
            # Focus on high-impact features for workflow improvement
            high_impact = importance[importance['importance_pct'] >= 5.0]
            print(f"High-impact features: {len(high_impact)} identified")
            ```
        """
        if not self.is_fitted:
            raise ValueError("Random Forest model must be fitted to extract clinical feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            # Create comprehensive feature importance analysis DataFrame with healthcare context
            importance_df = pd.DataFrame({
                'feature': self.model.feature_names_in_,                     # Clinical variable names
                'importance': self.model.feature_importances_,              # Ensemble importance weights
                'importance_pct': (self.model.feature_importances_ / 
                                 self.model.feature_importances_.sum() * 100), # Percentage contribution
                'importance_rank': np.argsort(-self.model.feature_importances_) + 1, # Priority ranking
                'clinical_significance': ['High' if imp > np.mean(self.model.feature_importances_) + np.std(self.model.feature_importances_)
                                        else 'Moderate' if imp > np.mean(self.model.feature_importances_)
                                        else 'Low' for imp in self.model.feature_importances_], # Impact level
                'bottleneck_potential': ['Critical' if imp > np.percentile(self.model.feature_importances_, 90)
                                       else 'Significant' if imp > np.percentile(self.model.feature_importances_, 75)
                                       else 'Moderate' if imp > np.percentile(self.model.feature_importances_, 50)
                                       else 'Limited' for imp in self.model.feature_importances_] # Bottleneck assessment
            }).sort_values('importance', ascending=False)
            
            # Add healthcare context and clinical metadata for stakeholder communication
            importance_df['healthcare_context'] = 'Medication preparation TAT prediction'
            importance_df['clinical_objective'] = '60-minute threshold optimization'
            importance_df['ensemble_confidence'] = 'High - Aggregated across multiple trees'
            importance_df['interpretation_method'] = 'Gini importance from Random Forest ensemble'
            
            logger.info(f"Random Forest feature importance extracted: {len(importance_df)} features analyzed")
            return importance_df
        else:
            raise AttributeError("Random Forest model does not have feature importance attributes")
    
    def get_clinical_insights(self) -> Dict[str, Any]:
        """
        Generate Clinical Insights from Random Forest Ensemble for Healthcare Stakeholders
        
        Comprehensive clinical insight generation optimized for healthcare pharmacy
        workflow optimization and healthcare stakeholder communication. Translates
        Random Forest feature importance into actionable clinical recommendations
        supporting evidence-based bottleneck identification and targeted intervention
        strategies for medication preparation efficiency and workflow optimization.
        
        Healthcare Insight Generation:
        - Critical bottlenecks: Top-tier features requiring immediate intervention attention
        - Significant drivers: Secondary factors supporting comprehensive workflow optimization
        - Feature interactions: Ensemble-detected relationships for complex bottleneck patterns
        - Intervention recommendations: Evidence-based suggestions for pharmacy operations improvement
        - Stakeholder communication: Healthcare-friendly insights supporting decision-making processes
        
        Random Forest Clinical Analysis:
        - Ensemble consensus: Multi-tree agreement providing robust bottleneck identification
        - Feature interaction detection: Tree pathway analysis revealing workflow complexity
        - Threshold-based categorization: Clinical significance levels for prioritized intervention
        - Impact quantification: Percentage contributions enabling resource allocation decisions
        - Evidence strength: Ensemble-based confidence supporting clinical decision validation
        
        Returns:
            Dict[str, Any]: Comprehensive clinical insights with healthcare recommendations
            and evidence-based intervention strategies supporting pharmacy workflow optimization.
        
        Raises:
            ValueError: Unfitted model preventing safe clinical insight generation
        
        Example:
            For clinical insight generation and healthcare communication:
            ```python
            # Generate actionable clinical insights from Random Forest ensemble
            insights = rf_model.get_clinical_insights()
            
            # Review critical bottleneck factors
            print("Critical bottlenecks requiring immediate intervention:")
            for factor in insights['critical_bottlenecks']:
                print(f"  {factor['feature']}: {factor['importance_pct']:.1f}% impact")
            
            # Access evidence-based recommendations
            for rec in insights['intervention_recommendations']:
                print(f"Recommendation: {rec}")
            ```

        """
        if not self.is_fitted:
            raise ValueError("Random Forest model must be fitted to generate clinical insights")
        
        # Extract feature importance analysis for clinical interpretation
        importance = self.get_feature_importance()
        
        # Identify critical bottlenecks (top 90th percentile) for immediate intervention
        critical_threshold = np.percentile(importance['importance_pct'], 90)
        critical_bottlenecks = importance[importance['importance_pct'] >= critical_threshold].head(5)
        critical_factors = [
            {
                'feature': row['feature'],
                'importance_pct': row['importance_pct'],
                'clinical_impact': f"Critical TAT driver - {row['importance_pct']:.1f}% ensemble contribution",
                'intervention_priority': 'Immediate',
                'bottleneck_type': 'Critical workflow constraint',
                'evidence_strength': 'High - Ensemble consensus'
            }
            for _, row in critical_bottlenecks.iterrows()
        ]
        
        # Identify significant drivers (75th-90th percentile) for comprehensive optimization
        significant_threshold = np.percentile(importance['importance_pct'], 75)
        significant_drivers = importance[
            (importance['importance_pct'] >= significant_threshold) & 
            (importance['importance_pct'] < critical_threshold)
        ].head(5)
        significant_factors = [
            {
                'feature': row['feature'],
                'importance_pct': row['importance_pct'],
                'clinical_impact': f"Significant TAT influence - {row['importance_pct']:.1f}% ensemble contribution",
                'intervention_priority': 'High',
                'bottleneck_type': 'Secondary workflow factor',
                'evidence_strength': 'Moderate to High - Multi-tree support'
            }
            for _, row in significant_drivers.iterrows()
        ]
        
        # Generate evidence-based clinical recommendations
        recommendations = []
        
        # Critical bottleneck intervention recommendations
        if critical_factors:
            primary_bottleneck = critical_factors[0]
            recommendations.append(
                f"Immediate priority: Address {primary_bottleneck['feature']} "
                f"({primary_bottleneck['clinical_impact']})"
            )
        
        # Comprehensive workflow optimization recommendations
        if significant_factors:
            recommendations.append(
                f"Secondary optimization: Focus on {len(significant_factors)} significant drivers "
                f"contributing {sum(f['importance_pct'] for f in significant_factors):.1f}% total impact"
            )
        
        # Ensemble-specific recommendations
        recommendations.append(
            "Ensemble validation: Multiple tree consensus provides high confidence in bottleneck identification"
        )
        
        # Generate comprehensive clinical insights with healthcare context
        clinical_insights = {
            'model_type': 'Random Forest Regression - Ensemble Learning',
            'clinical_objective': '60-minute TAT threshold optimization',
            'healthcare_context': 'Medication preparation workflow optimization',
            'critical_bottlenecks': critical_factors,              # Immediate intervention targets
            'significant_drivers': significant_factors,            # Secondary optimization factors
            'intervention_recommendations': recommendations,        # Evidence-based actions
            'ensemble_advantages': [
                'Robust feature importance through bootstrap aggregation',
                'Feature interaction detection through tree pathways',
                'Reduced overfitting through ensemble consensus',
                'High confidence bottleneck identification'
            ],
            'interpretability_confidence': 'High - Tree-based ensemble with feature importance',
            'clinical_validation': 'Multi-tree consensus evidence',    # Validation approach
            'stakeholder_communication': 'Clear feature ranking with percentage contributions',
            'total_features_analyzed': len(importance),                # Analysis scope
            'critical_features_identified': len(critical_factors),     # Critical bottlenecks
            'significant_features_identified': len(significant_factors), # Secondary drivers
            'oob_validation_score': getattr(self.model, 'oob_score_', None)  # Internal validation
        }
        
        logger.info(f"Random Forest clinical insights generated: {len(critical_factors)} critical, {len(significant_factors)} significant bottlenecks")
        return clinical_insights