# Heart Disease Prediction Project

A refactored and optimized machine learning project for predicting heart disease with a complete web frontend.

## Project Structure

```
heart_disease_project/
├── backend_api.py               # FastAPI backend
├── data/
│   └── heart.csv                # Dataset
├── frontend/                    # Web Application
│   ├── index.html               # Main UI view
│   ├── style.css                # Premium styling UI
│   └── app.js                   # Application logic
├── notebooks/
│   └── heart_disease_prediction.ipynb  # Original notebook
├── src/
│   ├── data_loader.py           # Data loading logic
│   ├── preprocessing.py         # Data cleaning and transformation
│   ├── models.py                # Model definitions and tuning
│   ├── evaluation.py            # Metrics and results
│   └── visualization.py         # Plots and graphs
├── main.py                      # Main entry point for training
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Setup & Installation

1.  Clone the repository (or download the files).
2.  Install all Python dependencies inside an active virtual environment:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Training the Model
Make sure `heart.csv` is located in the `data/` directory. Run the main script to train the model and generate `heart_disease_model.pkl`:
```bash
python main.py
```

### 2. Running the Backend API
Start the FastAPI server to serve predictions:
```bash
uvicorn backend_api:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 3. Running the Frontend UI
Simply open `frontend/index.html` in your favorite web browser.
It will automatically connect to the backend API running on port 8000 and present a premium user interface to submit patient diagnostic assessment data.

## Features

- **Modern Web UI**: A clean, minimalistic, glassmorphism UI built with Vanilla HTML, CSS, and JS.
- **RESTful API**: Fast and typesafe predictions via FastAPI.
- **Modular Design**: Separated concerns for better maintainability.
- **Automated ML Pipeline**: End-to-end data loading, processing, training, and evaluation.
