# ============================================================
# TRAINER MODULE
# Credit Score Classification Project
# ============================================================

import os
import time
import warnings

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


class Trainer:
    """
    Train, compare, tune, and save
    machine learning models.
    """

    def __init__(
        self,
        preprocessor,
        label_encoder,
        X_train,
        X_test,
        y_train,
        y_test,
        random_state=42,
    ):

        self.preprocessor = preprocessor
        self.label_encoder = label_encoder

        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

        self.random_state = random_state

        # ---------------------------------------------
        # Model Containers
        # ---------------------------------------------
        self.models = {}
        self.results = {}
        self.best_model_name = None
        self.best_pipeline = None
        self.best_predictions = None
        self.best_model = None
        self.best_score = 0.0

    # ========================================================
    # INITIALIZE MODELS
    # ========================================================

    def initialize_models(self):
        """
        Initialize all baseline models.
        """

        self.models = {
            "Logistic Regression": LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(
                random_state=self.random_state, n_estimators=200
            ),
            "XGBoost": XGBClassifier(
                objective="multi:softprob",
                random_state=self.random_state,
                eval_metric="mlogloss",
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                tree_method="hist",
            ),
        }

        print("=" * 60)
        print("Models Initialized")

        for model_name in self.models.keys():
            print(model_name)
        print()

        return self.models

    # ========================================================
    # HYPERPARAMETER SPACE
    # ========================================================

    def get_param_distributions(self):
        """
        Return parameter distributions
        for RandomizedSearchCV.
        """

        param_grid = {
            "Logistic Regression": {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__solver": ["lbfgs", "liblinear"],
            },
            "Decision Tree": {
                "classifier__criterion": ["gini", "entropy"],
                "classifier__max_depth": [5, 10, 15, 20, None],
                "classifier__min_samples_split": [2, 5, 10],
            },
            "Random Forest": {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [10, 20, None],
                "classifier__min_samples_split": [5, 10],
                "classifier__min_samples_leaf": [2, 4],
            },
            "XGBoost": {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.01, 0.05, 0.1],
                "classifier__max_depth": [3, 5, 7],
                "classifier__subsample": [0.8, 1.0],
            },
        }

        return param_grid

    # ========================================================
    # BUILD PIPELINE
    # ========================================================

    def build_pipeline(self, model):
        """
        Combine preprocessing
        and classifier into one pipeline.
        """

        pipeline = Pipeline(
            [("preprocessor", self.preprocessor), ("classifier", clone(model))]
        )

        return pipeline

    # ========================================================
    # EVALUATE SINGLE MODEL
    # ========================================================

    def evaluate_model(self, model_name, model):
        """
        Train and evaluate a single model.
        """
        print(f"Training {model_name}...")
        pipeline = self.build_pipeline(model)
        start_time = time.time()
        pipeline.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        y_pred = pipeline.predict(self.X_test)
        result = {
            "Model": model_name,
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            ),
            "F1 Score": f1_score(
                self.y_test, y_pred, average="weighted", zero_division=0
            ),
            "Training Time (s)": training_time,
            "Pipeline": pipeline,
            "Predictions": y_pred,
        }

        return result

    # ========================================================
    # TRAIN ALL BASELINE MODELS
    # ========================================================

    def train_models(self):
        """
        Train all baseline models and
        store their evaluation results.
        """
        self.initialize_models()

        print("=" * 60)
        print("Training Baseline Models")

        self.results = {}
        for model_name, model in self.models.items():
            result = self.evaluate_model(model_name, model)
            self.results[model_name] = result
        print("\nTraining Completed.\n")

        return self.results

    # ========================================================
    # COMPARE BASELINE MODELS
    # ========================================================

    def compare_models(self):
        """
        Create comparison table
        for all trained models.
        """

        comparison = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in result.items()
                    if key not in ["Pipeline", "Predictions"]
                }
                for result in self.results.values()
            ]
        )

        comparison = comparison.sort_values(by="F1 Score", ascending=False).reset_index(
            drop=True
        )

        print("=" * 60)
        print("Baseline Model Comparison")

        print(comparison)

        return comparison

    # ========================================================
    # SELECT BEST BASELINE MODEL
    # ========================================================

    def select_best_model(self):
        """
        Select the best baseline model
        based on weighted F1-score.
        """

        self.best_model_name = max(
            self.results, key=lambda name: self.results[name]["F1 Score"]
        )
        best_result = self.results[self.best_model_name]
        self.best_pipeline = best_result["Pipeline"]
        self.best_predictions = best_result["Predictions"]
        self.best_score = best_result["F1 Score"]
        self.best_model = clone(self.models[self.best_model_name])

        print()
        print("=" * 60)
        print(f"Best Baseline Model : {self.best_model_name}")
        print(f"Weighted F1 Score   : {self.best_score:.4f}")

        return self.best_pipeline

    # ========================================================
    # BASELINE SUMMARY
    # ========================================================

    def baseline_summary(self):
        """
        Display baseline performance summary.
        """

        comparison = self.compare_models()

        print()
        print("Best Baseline Model")
        print("-------------------")
        print(f"Model : {self.best_model_name}")
        print(f"Weighted F1 : {self.best_score:.4f}")

        return comparison

    # ========================================================
    # HYPERPARAMETER TUNING
    # ========================================================

    def tune_best_model(self, n_iter=20, cv=5, scoring="f1_weighted"):
        """
        Perform RandomizedSearchCV on the
        best baseline model.
        """

        print("=" * 60)
        print("Hyperparameter Tuning")

        pipeline = self.build_pipeline(self.best_model)
        param_grid = self.get_param_distributions()[self.best_model_name]

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            random_state=self.random_state,
            n_jobs=1,
            verbose=1,
        )

        start = time.time()

        random_search.fit(self.X_train, self.y_train)

        tuning_time = time.time() - start

        # -------------------------------------------------------
        # Evaluate tuned model on test set
        # -------------------------------------------------------

        tuned_pipeline = random_search.best_estimator_

        tuned_predictions = tuned_pipeline.predict(self.X_test)

        tuned_f1 = f1_score(
            self.y_test,
            tuned_predictions,
            average="weighted",
            zero_division=0,
        )

        baseline_f1 = self.results[self.best_model_name]["F1 Score"]

        print()
        print("Best Parameters")
        print(random_search.best_params_)
        print()

        print(f"Baseline Test F1 : {baseline_f1:.4f}")
        print(f"Tuned Test F1    : {tuned_f1:.4f}")
        print(f"Best CV F1       : {random_search.best_score_:.4f}")
        print(f"Tuning Time      : {tuning_time:.2f} seconds")

        # -------------------------------------------------------
        # Keep the better model
        # -------------------------------------------------------

        if tuned_f1 >= baseline_f1:

            print("\nTuned model performs better. Using tuned model.")

            self.best_pipeline = tuned_pipeline
            self.best_predictions = tuned_predictions
            self.best_score = tuned_f1

        else:

            print("\nBaseline model performs better. Keeping baseline model.")

            self.best_pipeline = self.results[self.best_model_name]["Pipeline"]
            self.best_predictions = self.results[self.best_model_name]["Predictions"]
            self.best_score = baseline_f1

        return random_search

    # ========================================================
    # LOG EXPERIMENT TO MLFLOW
    # ========================================================

    def log_mlflow(self, random_search):
        """
        Log experiment using MLflow.
        """

        with mlflow.start_run():

            mlflow.log_param("Model", self.best_model_name)

            mlflow.log_params(random_search.best_params_)

            accuracy = accuracy_score(self.y_test, self.best_predictions)

            precision = precision_score(
                self.y_test, self.best_predictions, average="weighted", zero_division=0
            )

            recall = recall_score(
                self.y_test, self.best_predictions, average="weighted", zero_division=0
            )

            f1 = f1_score(
                self.y_test, self.best_predictions, average="weighted", zero_division=0
            )

            mlflow.log_metric("Accuracy", accuracy)

            mlflow.log_metric("Precision", precision)

            mlflow.log_metric("Recall", recall)

            mlflow.log_metric("F1 Score", f1)

            mlflow.sklearn.log_model(sk_model=self.best_pipeline, name="model")

            artifacts = [
                "classification_report.txt",
                "confusion_matrix.png",
                "baseline_results.csv",
            ]

            for artifact in artifacts:
                if os.path.exists(artifact):
                    mlflow.log_artifact(artifact)

            print()
            print("MLflow logging completed.")

    # ========================================================
    # SAVE TRAINED MODEL
    # ========================================================

    def save_model(
        self,
        model_path="models/best_pipeline.pkl",
        encoder_path="models/label_encoder.pkl",
    ):
        """
        Save trained pipeline and
        label encoder.
        """

        os.makedirs("models", exist_ok=True)
        joblib.dump(
            self.best_pipeline,
            model_path,
            compress=("xz", 3),
        )
        joblib.dump(self.label_encoder, encoder_path, compress=3)

        print()
        print(f"Model saved to : {model_path}")
        print(f"LabelEncoder saved to : {encoder_path}")
