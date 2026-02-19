from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib

def get_models(random_state=42):
    """
    Returns a dictionary of models to evaluate.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(),
        "MLP Neural Net": MLPClassifier(max_iter=1000, random_state=random_state)
    }

def get_param_grids():
    """
    Returns parameter grids for hyperparameter tuning.
    """
    return {
        "Decision Tree": {
            'classifier__criterion': ['gini', 'entropy', 'log_loss'],
            'classifier__max_depth': [None, 3, 5, 7, 10, 15],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4, 6]
        },
        "Random Forest": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        },
        "Gradient Boosting": {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5]
        },
        "Logistic Regression": {
            'classifier__C': [0.1, 1, 10],
            'classifier__solver': ['liblinear', 'lbfgs']
        },
        "KNN": {
            'classifier__n_neighbors': [3, 5, 7, 9],
            'classifier__weights': ['uniform', 'distance']
        },
        "SVM": {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf']
        }
    }

def tune_model(pipeline, param_grid, X_train, y_train, cv=5):
    """
    Performs Grid Search CV to tune the model.
    """
    if not param_grid:
        print("No parameter grid provided suitable for this model. Fitting with defaults.")
        pipeline.fit(X_train, y_train)
        return pipeline

    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    search.fit(X_train, y_train)
    print("Best Params:", search.best_params_)
    print("Best CV Score:", search.best_score_)
    return search.best_estimator_

def compare_models(models, preprocessor, X_train, y_train, cv=5):
    """
    Compares models using cross-validation.
    """
    results = {}
    print("\nCross-Validation Scores (Accuracy):\n")

    for name, model in models.items():
        # Create pipeline: Preprocess -> Feature Selection -> Classifier
        clf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('selector', SelectKBest(score_func=f_classif, k='all')),
            ('classifier', model)
        ])
        
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        results[name] = cv_scores.mean()
        print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
    return results

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
