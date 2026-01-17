# Heart Disease Prediction Project

A refactored and optimized machine learning project for predicting heart disease.

## Project Structure

```
heart_disease_project/
├── data/
│   └── heart.csv                # Dataset
├── notebooks/
│   └── heart_disease_prediction.ipynb  # Original notebook
├── src/
│   ├── data_loader.py           # Data loading logic
│   ├── preprocessing.py         # Data cleaning and transformation
│   ├── models.py                # Model definitions and tuning
│   ├── evaluation.py            # Metrics and results
│   └── visualization.py         # Plots and graphs
├── main.py                      # Main entry point
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Setup & Installation

1.  Clone the repository (or download the files).
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure `heart.csv` is located in the `data/` directory.
2.  Run the main script:
    ```bash
    python main.py
    ```

## Features

- **Modular Design**: Separated concerns for better maintainability.
- **Automated Pipeline**: End-to-end data loading, processing, training, and evaluation.
- **Model Comparison**: Automatically evaluates multiple models (Logistic Regression, Random Forest, SVM, etc.).
- **Hyperparameter Tuning**: Optimizes the best performing model.
- **Visualization**: Generates EDA and feature importance plots.
