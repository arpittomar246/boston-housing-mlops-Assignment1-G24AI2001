# Boston Housing Price Prediction — MLOps Assignment 1

### Course Project: MLOps Assignment 1  
**Author:** Arpit Tomar  
**Roll No:** G24AI2001 

---

This project demonstrates a **complete MLOps pipeline** for predicting **Boston housing prices** using two classical machine learning models —  
**DecisionTreeRegressor** and **KernelRidge** — with full **CI/CD automation** through **GitHub Actions**.

The goal is to showcase:
- End-to-end workflow automation for model training & evaluation.
- Modular, reusable code structure.
- Continuous Integration (CI) to automatically validate the ML pipeline on every code push.

## Project Structure
```boston-housing-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions workflow for CI/CD
├── data/
│   └── boston.csv               # Local copy of Boston dataset (used by CI)
├── misc.py                      # Helper functions (load, preprocess, train, eval)
├── train.py                     # DecisionTreeRegressor training script
├── train2.py                    # KernelRidge training script (tunable)
├── requirements.txt             # Python dependencies
├── results_summary.csv          # Generated performance results (after runs)
├── README.md                    # Project documentation
└── .gitignore                   # Files and folders to ignore in git
```

## Model Training and Evaluation

1️-Decision Tree Model

This script:
```-Loads and scales the dataset
-Trains the Decision Tree model
-Prints evaluation metrics
-Saves the results to results_summary.csv
```

2️-Kernel Ridge Model

This script supports hyperparameter tuning using command-line arguments.


## Available parameters:

Argument	Description	Default
```
--kernel	Kernel type (linear, poly, rbf, sigmoid, cosine)	linear
--alpha	Regularization strength	1.0
--gamma	Kernel coefficient (for rbf, poly, etc.)	None
--test_size	Train-test split ratio	0.2
--show_n	Number of predictions to display	10
```

## Results Summary

After both scripts are executed (either locally or via CI), a CSV file results_summary.csv is generated:

## Model	MSE	RMSE	R²
```
DecisionTreeRegressor	10.4161	3.2274	0.8579
KernelRidge (RBF, α=0.5, γ=0.1)	15.4754	3.9339	0.7890
```

## Observation:
Decision Tree achieved better performance with lower error and higher R², indicating a stronger fit to the dataset.

## Installation and Usage Instructions

### 1 - Clone the Repository
```
git clone https://github.com/arpittomar/boston-housing-mlops-Assignment1-G24AI2001.git
cd boston-housing-mlops-Assignment1-G24AI2001
```

### 2 - Create a Virtual Environment
```
python -m venv venv
```


### Activate it:

Windows: venv\Scripts\activate
macOS/Linux: source venv/bin/activate

### 3- Install Dependencies
```
pip install -r requirements.txt
```

This installs:
1.pandas
2.numpy
3.scikit-learn
4.matplotlib

### 4️- Run the Models
Decision Tree:
```
python train.py
```

Kernel Ridge:
```
python train2.py --kernel rbf --alpha 0.5 --gamma 0.1
```

### 5️- View Results
```
cat results_summary.csv
```

Example Output:
```
model,mse,rmse,r2
DecisionTreeRegressor,10.4161,3.2274,0.8579
KernelRidge (RBF, α=0.5, γ=0.1),15.4754,3.9339,0.7890
```


### 6️- GitHub Actions (CI/CD)

This project automates testing and model evaluation on every push to:
1.dtree
2.kernelridge
3.main

The workflow file .github/workflows/ci.yml performs:
1.Dependency installation
2.Running both training scripts
3.Outputting model performance logs


## Project Workflow Summary

### Local Development
```
-Implement helper functions in misc.py
-Train and evaluate models locally (train.py, train2.py)
-Validate performance and save metrics
```


### Git & Branch Management
```
-dtree branch → DecisionTreeRegressor implementation
-kernelridge branch → KernelRidge + CI/CD setup
-main branch → Final merged branch
```

### Continuous Integration (CI)
```
-Every push triggers automated runs on GitHub Actions
-CI validates model scripts, dependency installation, and prints metrics
-Ensures reproducibility and stable workflow execution
```


# Author Information

Name: Arpit Tomar
Roll No: G24AI2001
Email: g24ai2001@iitj.ac.in


Note: Dataset is loaded from http://lib.stat.cmu.edu/datasets/boston as required (sklearn.load_boston is deprecated).

© 2025 Arpit Tomar | G24AI2001
All rights reserved. For academic and educational use only!


