# ============================================================
# EVALUATOR MODULE
# Credit Score Classification Project
# ============================================================
import os
import warnings
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class Evaluator:
    """
    Evaluate trained machine learning models and
    generate evaluation reports.
    """

    def __init__(self, pipeline, X_test, y_test, label_encoder, comparison_table=None):
        self.pipeline = pipeline
        self.X_test = X_test
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.comparison_table = comparison_table
        self.y_pred = None
        self.metrics = {}
        os.makedirs("reports", exist_ok=True)

    # ========================================================
    # MAKE PREDICTIONS
    # ========================================================

    def evaluate(self):
        """
        Generate predictions and
        calculate evaluation metrics.
        """

        print("=" * 60)
        print("Evaluating Best Model")
        self.y_pred = self.pipeline.predict(self.X_test)
        self.metrics = {
            "Accuracy": accuracy_score(self.y_test, self.y_pred),
            "Precision": precision_score(
                self.y_test, self.y_pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                self.y_test, self.y_pred, average="weighted", zero_division=0
            ),
            "F1 Score": f1_score(
                self.y_test, self.y_pred, average="weighted", zero_division=0
            ),
        }
        print()
        for metric, value in self.metrics.items():
            print(f"{metric:<12}: {value:.4f}")

        return self.metrics

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    def generate_classification_report(self):
        """
        Generate sklearn classification report.
        """

        if self.y_pred is None:
            self.evaluate()

        report = classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0,
        )
        print()
        print("=" * 60)
        print("Classification Report")
        print(report)

        return report

    # ========================================================
    # SAVE CLASSIFICATION REPORT
    # ========================================================

    def save_classification_report(self, filename="reports/classification_report.txt"):
        """
        Save classification report as text file.
        """

        if self.y_pred is None:
            self.evaluate()

        report = classification_report(
            self.y_test,
            self.y_pred,
            target_names=self.label_encoder.classes_,
            zero_division=0,
        )

        with open(filename, "w") as file:
            file.write(report)

        print()
        print(f"Classification report saved to {filename}")

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    def plot_confusion_matrix(self):
        """
        Generate confusion matrix plot.
        """

        if self.y_pred is None:
            self.evaluate()

        cm = confusion_matrix(self.y_test, self.y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.label_encoder.classes_
        )

        display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        return fig

    # ========================================================
    # SAVE CONFUSION MATRIX
    # ========================================================

    def save_confusion_matrix(self, filename="reports/confusion_matrix.png"):
        """
        Save confusion matrix figure.
        """
        fig = self.plot_confusion_matrix()
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print()
        print(f"Confusion matrix saved to {filename}")

    # ========================================================
    # SAVE BASELINE RESULTS
    # ========================================================

    def save_baseline_results(self, filename="reports/baseline_results.csv"):
        """
        Save baseline comparison table.
        """

        if self.comparison_table is None:
            print()
            print("No comparison table available.")
            return

        table = self.comparison_table.copy()
        columns_to_drop = ["Pipeline", "Predictions"]
        existing_columns = [
            column for column in columns_to_drop if column in table.columns
        ]

        if existing_columns:
            table = table.drop(columns=existing_columns)

        table.to_csv(filename, index=False)
        print()
        print(f"Baseline results saved to {filename}")

    # ========================================================
    # DISPLAY BASELINE RESULTS
    # ========================================================

    def display_baseline_results(self):
        """
        Display baseline comparison table.
        """

        if self.comparison_table is None:
            print()
            print("No comparison table available.")
            return

        print()
        print("=" * 60)
        print("Baseline Model Comparison")
        print(self.comparison_table)

    # ========================================================
    # COMPLETE EVALUATION PIPELINE
    # ========================================================

    def evaluate_all(self):
        """
        Execute the complete evaluation workflow.
        """

        print("=" * 60)
        print("MODEL EVALUATION")
        self.evaluate()
        self.generate_classification_report()
        self.save_classification_report()
        self.save_confusion_matrix()
        self.display_baseline_results()
        self.save_baseline_results()

        print()
        print("=" * 60)
        print("Evaluation Completed")

        return self.metrics

    # ========================================================
    # EVALUATION SUMMARY
    # ========================================================

    def summary(self):
        """
        Display evaluation summary.
        """

        if not self.metrics:
            self.evaluate()

        print()
        print("=" * 60)
        print("Evaluation Summary")
        for metric, value in self.metrics.items():
            print(f"{metric:<12}: {value:.4f}")

        return self.metrics
