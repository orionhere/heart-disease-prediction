from src.data_loader import load_data
from src.preprocessing import identify_columns, create_preprocessor, split_data
from src.visualization import plot_eda, plot_feature_importance
from src.models import get_models, compare_models, get_param_grids, tune_model, save_model
from src.evaluation import evaluate_model
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import os

def main():
    print("Starting Heart Disease Prediction Pipeline...")
    
    # 1. Load Data
    data_path = os.path.join("data", "heart.csv")
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. EDA
    print("\nGenering EDA plots...")
    plot_eda(df)

    # 3. Preprocessing
    # Separate features for identification to avoid including target
    if 'target' in df.columns:
        features_df = df.drop('target', axis=1)
    else:
        features_df = df
        
    num_cols, cat_cols = identify_columns(features_df)
    preprocessor = create_preprocessor(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = split_data(df)

    # 4. Model Comparison
    models = get_models()
    results = compare_models(models, preprocessor, X_train, y_train)
    
    best_model_name = max(results, key=results.get)
    print(f"\nBest Baseline Model: {best_model_name} ({results[best_model_name]:.4f})")

    # 5. Model Tuning
    print(f"\nTuning {best_model_name}...")
    param_grids = get_param_grids()
    grid = param_grids.get(best_model_name, {})
    
    # Create the full pipeline for tuning
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', SelectKBest(score_func=f_classif, k='all')),
        ('classifier', models[best_model_name])
    ])
    
    best_estimator = tune_model(pipeline, grid, X_train, y_train)

    # 6. Final Evaluation
    evaluate_model(best_estimator, X_test, y_test, model_name=best_model_name)

    # 7. Feature Importance
    plot_feature_importance(best_estimator, num_cols, cat_cols)

    # 8. Save Model
    save_model(best_estimator, "heart_disease_model.pkl")

if __name__ == "__main__":
    main()
