import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def identify_columns(X: pd.DataFrame):
    """
    Identifies numerical and categorical columns.
    """
    categorical_cols = [col for col in X.columns if X[col].nunique() < 10]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    print("Categorical Features:", categorical_cols)
    print("Numerical Features:", numerical_cols)
    
    return numerical_cols, categorical_cols

def create_preprocessor(numerical_cols, categorical_cols):
    """
    Creates a ColumnTransformer for preprocessing.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        
    return preprocessor

def split_data(df: pd.DataFrame, target_col: str = 'target', test_size: float = 0.2, random_state: int = 42):
    """
    Splits data into train and test sets.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
