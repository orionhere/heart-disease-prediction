import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_eda(df: pd.DataFrame):
    """
    Plots basic Exploratory Data Analysis.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Target Distribution
    sns.countplot(x='target', data=df, ax=ax[0], palette='coolwarm')
    ax[0].set_title("Distribution of Target Variable")

    # Correlation Heatmap
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=0.5, ax=ax[1])
    ax[1].set_title("Feature Correlation Matrix")

    plt.tight_layout()
    # Save instead of show
    output_path = "reports/figures/eda.png"
    plt.savefig(output_path)
    print(f"EDA plot saved to {output_path}")
    plt.close()

def plot_feature_importance(model, numerical_cols, categorical_cols):
    """
    Plots feature importance if available.
    """
    # Try to extract feature importance
    try:
        # Access the classifier step
        classifier = model.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            # Access the preprocessor to get feature names
            preprocessor = model.named_steps['preprocessor']
            
            # Note: This follows the order in ColumnTransformer [num, cat]
            num_names = numerical_cols
            cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
            feature_names = np.r_[num_names, cat_names]
            
            # Create DataFrame
            feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
            plt.title("Top 10 Feature Importances")
            output_path = "reports/figures/feature_importance.png"
            plt.savefig(output_path)
            print(f"Feature importance plot saved to {output_path}")
            plt.close()
        else:
            print("Model does not expose feature_importances_")
            
    except Exception as e:
        print(f"Could not extract/plot feature importance: {e}")
