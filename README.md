# Heart Disease Prediction Project

![Heart Health](https://img.shields.io/badge/Health-AI_Diagnostics-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688) ![Frontend](https://img.shields.io/badge/Frontend-Vanilla_JS-f39f37)

## Project Overview

The **Heart Disease Prediction Project** is a full-stack Machine Learning application designed to predict the likelihood of heart disease in patients based on their clinical test results and physical parameters. Leveraging a trained Scikit-Learn `DecisionTreeClassifier` (or an optimized model pipeline), this project automatically processes patient vitals—such as age, sex, chest pain type, resting blood pressure, and cholesterol levels—to output a clear diagnostic risk assessment.

Unlike standalone Jupyter notebook analyses, this project provides a **deployable, end-to-end ecosystem**:
- A robust **Data Pipeline** to process historical physiological data.
- A **Model Training Suite** to compare, tune, and evaluate multiple algorithms.
- A **FastAPI Backend Services** to expose the best performing model via a typesafe REST API.
- A **Premium Web Interface** built with modern, minimalistic vanilla web technologies (HTML, CSS, JS) allowing seamless interactions out-of-the-box.

## Project Synopsis

According to the World Health Organization, cardiovascular diseases are the leading cause of death globally. Early diagnosis is critical. This project serves as a proof-of-concept for how AI can assist healthcare professionals in making rapid, data-driven diagnostic assessments. The model is trained on the classic UCI Heart Disease dataset, which contains 14 clinical attributes.

The core pipeline handles everything from data ingestion to model serialization. Once the model (`heart_disease_model.pkl`) is generated, the FastAPI backend loads it into memory and begins listening for POST requests. The vanilla web frontend captures user input through a dynamic, glassmorphism-styled form, sends the payload to the backend, and displays the risk percentage via an interactive gauge animation.

## Key Functionalities

### 1. Automated Machine Learning Pipeline (`main.py`)
- **Data Preprocessing**: Automatically handles numeric imputations, scaling, and categorical one-hot encoding without data leakage.
- **Model Selection**: Evaluates baseline performance across multiple models (Logistic Regression, Random Forest, SVM, Decision Tree, etc.).
- **Hyperparameter Tuning**: Utilizes `GridSearchCV` optimized via cross-validation to find the best configuration for the leading model.
- **Evaluation & Persistence**: Logs performance metrics (Accuracy, F1 Score) and serializes the winning model pipeline using `joblib`.

### 2. Typesafe RESTful API (`backend_api.py`)
- **FastAPI Framework**: Ensures high performance and automatic interactive API documentation (via Swagger UI at `/docs`).
- **Pydantic Validation**: Strictly enforces input data shapes and types before passing data to the model avoiding silent inference errors.
- **CORS Support**: Configured to instantly accept cross-origin requests from the local frontend application.

### 3. Native Web Interface (`frontend/`)
- **Zero-Build Architecture**: Runs straight in the browser without Node.js, Webpack, or Vite, ensuring maximum compatibility.
- **Premium UI/UX System**: Uses vanilla CSS variables to implement a state-of-the-art dark mode design featuring glass container effects and CSS transitions.
- **Asynchronous Operations**: Features seamless loading states and error handling during inference requests via modern JavaScript `async/await`.

---

## Project Structure

```
heart_disease_project/
├── backend_api.py               # FastAPI server and inference logic
├── data/
│   └── heart.csv                # Core diagnostic dataset
├── frontend/                    # Web Application Front-end
│   ├── index.html               # Semantic UI view
│   ├── style.css                # Premium dynamic stylesheets
│   └── app.js                   # API communication logic
├── notebooks/
│   └── heart_disease_prediction.ipynb  # Exploratory Data Analysis & Prototyping
├── src/                         # Core Machine Learning Modules
│   ├── data_loader.py           # Data ingestion
│   ├── preprocessing.py         # Data cleaning pipes
│   ├── models.py                # Estimator configurations
│   ├── evaluation.py            # Model scoring
│   └── visualization.py         # EDA plotting
├── main.py                      # Training CLI entry point
├── requirements.txt             # Project dependencies
└── README.md                    # Project Documentation
```

## Setup & Installation

1.  Clone the repository to your local machine:
    ```bash
    git clone https://github.com/orionhere/heart-disease-prediction.git
    cd heart-disease-prediction
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage Guide

### 1. Training the Model
Make sure `heart.csv` is present in the `data/` directory. Run the main script to re-train the models and generate a fresh `heart_disease_model.pkl` file:
```bash
python main.py
```

### 2. Running the Backend API
Start the FastAPI server to serve predictions:
```bash
uvicorn backend_api:app --reload
```
The API will become accessible at `http://127.0.0.1:8000`. You can visit `http://127.0.0.1:8000/docs` to test endpoints manually in your browser.

### 3. Running the Frontend UI
Simply open `frontend/index.html` in your favorite web browser (`Start-Process frontend/index.html` on Windows, or just double click the file).
It will automatically connect to the local backend API and present the premium assessment interface to input new patient data.
