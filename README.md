# Student Performance Predictor

This project implements machine learning models to predict student performance based on various academic and demographic features. The project includes both regression (GPA prediction) and classification (pass/fail prediction) models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🎯 Overview

The Student Performance Predictor uses machine learning algorithms to:
- **Predict GPA**: Regression model to predict continuous GPA values
- **Classify Pass/Fail**: Binary classification to determine if a student will pass (GPA ≥ 2.0)

The project leverages features such as study time, attendance, and other academic indicators to make accurate predictions.

## ✨ Features

- **Data Analysis**: Comprehensive exploratory data analysis with visualizations
- **Dual Modeling Approach**: 
  - Linear Regression for GPA prediction
  - Logistic Regression for pass/fail classification
- **Model Evaluation**: Complete performance metrics including RMSE, R², accuracy, precision, recall, and F1-score
- **Interactive Jupyter Notebook**: Step-by-step analysis and visualization
- **Automated Training Pipeline**: Ready-to-run Python script for model training

## 📊 Dataset

The project uses student performance data containing features such as:
- Study time per week
- Academic performance indicators
- Student demographics
- Target variables: GPA and Pass/Fail status

**Dataset Location**: local Data storage 

## 🤖 Models

### Regression Model (GPA Prediction)
- **Algorithm**: Linear Regression
- **Features**: All available features except StudentID, GPA, and PassStatus
- **Preprocessing**: StandardScaler for feature normalization
- **Evaluation Metrics**: RMSE and R-squared

### Classification Model (Pass/Fail Prediction)
- **Algorithm**: Logistic Regression
- **Target**: Binary classification (Pass: GPA ≥ 2.0, Fail: GPA < 2.0)
- **Features**: Same as regression model
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository** (or download the project files):
   ```bash
   git clone https://github.com/Himanshu-u-rai/student-performance-predictor.git
   cd student-performance-predictor
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn jupyter
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, sklearn, matplotlib, seaborn; print('All packages installed successfully!')"
   ```

## 💻 Usage

### Option 1: Run the Training Script
```bash
python train_model.py
```

This will:
- Load and preprocess the data
- Train both regression and classification models
- Display performance metrics

### Option 2: Interactive Analysis
```bash
jupyter notebook Student_Performance_Analysis.ipynb
```

This provides:
- Step-by-step data exploration
- Visualization of data distributions
- Interactive model building and evaluation

### Expected Output
```
Regression Model RMSE: 0.189
Regression Model R-squared: 0.957
Classification Model Accuracy: 0.954
Classification Model Precision: 0.960
Classification Model Recall: 0.943
Classification Model F1-score: 0.952
```

## 📁 Project Structure

```
Student-Performance-Predictor/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── train_model.py                     # Main training script
├── Student_Performance_Analysis.ipynb # Jupyter notebook for analysis
│
└── Data/
    └── Student_performance_data _.csv # Dataset (external location)
```

## 📈 Results

### Regression Model Performance
- **RMSE**: ~0.189 (Root Mean Square Error)
- **R²**: ~0.957 (97% variance explained)

### Classification Model Performance
- **Accuracy**: ~95.4%
- **Precision**: ~96.0%
- **Recall**: ~94.3%
- **F1-Score**: ~95.2%

Both models demonstrate excellent performance with high accuracy and strong predictive capabilities.

## 📦 Requirements

The project dependencies are listed in `requirements.txt`:

```
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
jupyter==1.0.0
```

**Minimum System Requirements**:
- Python 3.7+
- 4GB RAM (recommended)
- 1GB free disk space

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: Ensure the dataset path in `train_model.py` matches your local file structure before running the scripts.
