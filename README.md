# Healthcare TAT Prediction System
*Predicting medication preparation turnaround times to optimize pharmacy workflows*

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]() [![Coverage](https://img.shields.io/badge/coverage-78%25-yellow)]() [![Tests](https://img.shields.io/badge/tests-900+-blue)]() [![Python](https://img.shields.io/badge/python-3.9+-blue)]()

## 📋 **Project Overview**

This system analyzes medication preparation workflows in healthcare settings to predict turnaround times (TAT) and identify bottlenecks. Built for Dana Farber's pharmacy team to improve patient care by optimizing drug preparation processes.

**Goal**: Predict if medication TAT will exceed the 60-minute threshold and identify workflow bottlenecks.

**Dataset**: 100,000 medication orders (Jan-July 2025) with ~10% missing timestamps, including patient demographics, clinical data, lab values, and operational metrics.

## 🎯 **Key Design Innovations**

### **1. 🤖 Multi-Model Ensemble Architecture**
- **Ridge Regression**: Interpretable linear baseline with clinical coefficient insights
- **XGBoost**: Captures complex non-linear interactions in workflow patterns  
- **Random Forest**: Robust ensemble with feature importance ranking
- **Stacking Meta-Learner**: Combines all models using cross-validated predictions

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
- **Statistical Significance Testing**: Identifies true bottlenecks vs. random variation
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

### **📈 Analysis Reports & Visualizations**

**Interactive EDA Report**: [📊 Comprehensive Data Analysis](htmlcov_full/index.html)

**Key Figures & Insights**:
- **Figure 1**: Step-wise delay distribution showing prep-to-validation as primary bottleneck
- **Figure 2**: SHAP feature importance revealing queue length as top predictor  
- **Figure 3**: Bottleneck analysis by shift showing evening delays 40% higher
- **Figure 4**: Model calibration plots demonstrating prediction reliability
- **Figure 5**: Operational efficiency heatmaps by floor and time of day

### **🎯 Major Findings**
- **Primary Bottleneck**: Preparation-to-second-validation step (avg: 18.3 min)
- **Top Predictors**: Queue length, floor occupancy, treatment complexity, lab abnormalities
- **Operational Insights**: Evening shift 40% slower, Floor 3 most efficient
- **Clinical Impact**: High-severity patients average 23% longer TAT
- **Staffing Effect**: BCOP pharmacists reduce TAT by 12% vs. RPh credentials

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

### **Test Coverage: 900+ Tests | 78% Coverage**
```
tests/
├── unit/ (600+ tests)           # Individual component testing
│   ├── test_features/           # Feature engineering validation
│   ├── test_models/             # Model architecture testing  
│   └── test_pipelines/          # Data pipeline validation
├── integration/ (150+ tests)    # End-to-end workflow testing
├── performance/ (50+ tests)     # Latency and throughput benchmarks
└── clinical/ (100+ tests)       # Healthcare domain validation
```

**Quality Assurance**:
- **Performance Gates**: RMSE < 15 minutes, inference < 50ms
- **Data Validation**: Schema compliance, temporal causality checks
- **Model Reliability**: Cross-validation, calibration testing
- **Clinical Validation**: Domain expert review of predictions

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
- `train_model.py`: Multi-model training with hyperparameter optimization  
- `run_bottleneck_analysis.py`: Statistical bottleneck identification
- `get_eda.py`: Generate comprehensive EDA reports
- `get_delay_plots.py`: Visualize step-wise delay patterns

### **Testing**
```bash
# Run full test suite
python -m pytest tests/ --cov=src/tat --cov-report=html

# Run specific test categories  
python -m pytest tests/unit/test_features/    # Feature engineering tests
python -m pytest tests/integration/          # End-to-end pipeline tests
```

---

## 💡 **Key Innovations Summary**

1. **🤖 Ensemble Architecture**: Meta-learning stacking combining Ridge, XGBoost, and Random Forest
2. **⏰ Smart Imputation**: Sequential forward-fill with operational context for missing timestamps  
3. **🔍 SHAP Explainability**: Clinical feature importance with interaction analysis
4. **📊 Bottleneck Detection**: Statistical workflow analysis with actionable recommendations
5. **⚡ Production Scaling**: Modular design with automated ML pipeline and monitoring

**Built for healthcare environments with focus on interpretability, reliability, and clinical utility.**

---