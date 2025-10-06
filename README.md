# Healthcare TAT Prediction System
*Predicting medication preparation turnaround times to optimize pharmacy workflows*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![Coverage](https://img.shields.io/badge/coverage-78%25-yellow)]() [![Tests](https://img.shields.io/badge/tests-900+-blue)]() [![Python](https://img.shields.io/badge/python-3.9+-blue)]()

## 📋 **Project Overview**

This system analyzes medication preparation workflows in healthcare settings to predict turnaround times (TAT) and identify bottlenecks. Built for Dana Farber's pharmacy team to improve patient care by optimizing drug preparation processes.

**Goal**: Predict if medication TAT will exceed the 60-minute threshold and identify workflow bottlenecks.

**Dataset**: 100,000 medication orders (Jan-July 2025) with ~10% missing timestamps, including patient demographics, clinical data, lab values, and operational metrics.

### **📈 Analysis Reports & Visualizations**

**Delays Plot**
<img width="3883" height="2487" alt="delays" src="https://github.com/user-attachments/assets/9315e15d-ef66-4081-917f-b7044b2ec661" />


**Interactive EDA Report**: 📊 Comprehensive Data Analysis(reports/eda/summary.html)
Also at console

**Feature Importance Plots** (More at console)
<img width="1170" height="1108" alt="tat_shap_analysis_xgboost" src="https://github.com/user-attachments/assets/604bd274-2196-4e02-a2f6-800e2f25330f" />
<img width="1182" height="1108" alt="tat_shap_analysis_random_forest" src="https://github.com/user-attachments/assets/5ba46856-a1b6-4d72-b85f-a4bc3e4fe4e3" />
<img width="1164" height="1108" alt="tat_shap_analysis_linear" src="https://github.com/user-attachments/assets/47aa0347-e596-4a04-9faa-f785d0fa1b9b" />
<img width="1168" height="1108" alt="tat_shap_analysis_ensemble" src="https://github.com/user-attachments/assets/9f49c844-12b9-41b4-ab30-24d900c542e3" />



**Bottleneck Analysis**: (More at console)
<img width="3342" height="2368" alt="medication_prep_bottleneck_heatmap" src="https://github.com/user-attachments/assets/eb869492-66dc-4a21-9d32-898b42474017" />

## 🎯 **Key Design Decisions**

### **1. 🤖 Multi-Model Ensemble Architecture**
- **Ridge Regression**: Interpretable linear baseline with clinical coefficient insights
- **XGBoost**: Captures complex non-linear interactions in workflow patterns  
- **Random Forest**: Robust ensemble with feature importance ranking
- **Stacking Meta-Learner**: Meta-learning stacking combining Ridge, XGBoost, and Random Forest

### **2. ⏰ Advanced Missing Data Imputation Strategy**
**Sequential Forward-Fill with Operational Context**
- Learns step-specific processing times from historical data
- Respects chronological workflow order (prevents future data leakage)
- Uses operational context (shift, floor, staffing) for accurate estimates
- Interpolates between valid timestamps when available
- Maintains clinical workflow sequence integrity

### **3. 🔍 SHAP-Based Explainable AI**
- **Global Feature Importance**: Identifies key drivers across all predictions
- **Local Explanations**: Individual prediction breakdowns for clinical review
- **Feature Interactions**: Discovers complex relationships between variables
- **Clinical Categorization**: Groups features by temporal, clinical, and operational factors

### **4. 📊 Comprehensive Bottleneck Analysis**
- **Step-wise Delay Computation**: Measures time between each workflow stage
- **Statistical Significance Testing**: Identifies true bottlenecks
- **Operational Stratification**: Analyzes delays by shift, floor occupancy, and staffing
- **Actionable Recommendations**: Provides specific interventions for each bottleneck

### **5. ⚡ Production-Ready Scaling Architecture**
- **Modular Design**: Independent feature engineering components with sklearn interfaces
- **Automated ML Pipeline**: CI/CD with weekly retraining and performance monitoring
- **Real-time Inference**: <50ms prediction latency for operational integration
- **Model Monitoring**: Automated drift detection and retraining triggers

## � **Key Results & Analysis**

### **Model Performance**
| Model | RMSE (min) | MAE (min) | R² Score | Threshold Accuracy |
|-------|------------|-----------|----------|--------------------|
| Ridge Regression | 13.67 | 10.82 | 0.847 | 89.2% |
| XGBoost | 13.51 | 10.65 | 0.851 | 91.7% |
| Random Forest | 13.78 | 10.91 | 0.844 | 88.9% |
| **Stacking Ensemble** | **13.51** | **10.58** | **0.852** | **92.3%** |

## 🛠 **Technical Implementation**

### **Feature Engineering (77-dimensional feature space)**
- **Temporal Features**: Cyclical encoding of time patterns, shift assignments, business hours
- **Clinical Features**: Lab value normalization, abnormality flags, organ dysfunction scores  
- **Operational Features**: Staff capacity modeling, queue utilization, floor pressure indices
- **Delay Features**: Step-wise workflow timing with missing data imputation

### **Data Pipeline**
```
Raw Data → Validation → Timestamp Reconstruction → Feature Engineering → Model Training → Predictions
```

### **Project Structure**
```
src/tat/
├── features/          # Feature engineering modules
│   ├── temporal/      # Time-based features & delay computation  
│   ├── categoricals.py # Categorical encoding
│   ├── labs.py        # Clinical lab features
│   └── operational.py # Staffing & capacity features
├── models/            # ML model implementations
├── analysis/          # Bottleneck & feature importance analysis  
├── pipelines/         # Data processing workflows
└── scripts/           # Execution scripts
```

## � **Comprehensive Testing Suite**

Test Coverage: 900+ Tests | 80% Coverage**

## � **Quality Assurance**:

### Performance Gates
- 870+ automated tests (unit, integration) ensure reliability.
- Cross-validation and Optuna optimization for robust model selection.
- RMSE and accuracy benchmarks on holdout splits.

### Data Validation
- Schema and missing value checks via custom cleaners.
- Feature encoding and range validation for all input types.

### Model Reliability
- SHAP and feature importance analysis.
- Ensemble and base models validated across seeds/splits.
- Integration tests confirm end-to-end pipeline stability.

## � **Quick Start**

### **Installation**
```bash
git clone https://github.com/Laya303/DHCI-TAT-Prediction.git
cd DHCI-TAT-Prediction

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -e .

# Run analysis
python -m tat.scripts.prepare_dataset    # Data preprocessing
python -m tat.scripts.train_model       # Model training  
python -m tat.scripts.run_bottleneck_analysis  # Bottleneck identification
```

### **Key Scripts**
- `prepare_dataset.py`: Complete data preprocessing pipeline
- `train_model.py`: Multi-model training with hyperparameter optimization and feature importance analysis
- `run_bottleneck_analysis.py`: Statistical bottleneck identification
- `get_eda.py`: Generate comprehensive EDA reports
- `get_delay_plots.py`: Visualize step-wise delay patterns

### **Testing**
```bash
# Run full test suite
python -m pytest tests/ --cov=src/tat --cov-report=html

# Run specific test categories  
python -m pytest tests/test_features/test_delays.py    # Feature engineering tests
python -m pytest tests/test_integration/test_system_integration.py        # End-to-end pipeline tests
```

---

**Built for healthcare environments with focus on interpretability, reliability, and clinical utility.**

---
