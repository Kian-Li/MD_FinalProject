# ============================================================
# MAIN PIPELINE
# Credit Score Classification Project
# ============================================================

from preprocessing import Preprocessor
from trainer import Trainer
from evaluator import Evaluator


def main():

    print("=" * 60)
    print("Credit Score Classification Pipeline")
    print("=" * 60)

    # -------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------

    preprocessor = Preprocessor(data_path="data_D.csv")

    X_train, X_test, y_train, y_test, transformer, label_encoder = (
        preprocessor.preprocess()
    )

    # -------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------

    trainer = Trainer(
        preprocessor=transformer,
        label_encoder=label_encoder,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    trainer.train_models()
    trainer.select_best_model()
    comparison_table = trainer.baseline_summary()
    random_search = trainer.tune_best_model()

    # -------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------

    evaluator = Evaluator(
        pipeline=trainer.best_pipeline,
        X_test=X_test,
        y_test=y_test,
        label_encoder=label_encoder,
        comparison_table=comparison_table,
    )

    evaluator.evaluate_all()

    # -------------------------------------------------------
    # MLFLOW
    # -------------------------------------------------------

    trainer.log_mlflow(random_search)

    # -------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------

    trainer.save_model()

    print()

    print("=" * 60)
    print("PROJECT COMPLETED")
    print()
    print("Best Model :", trainer.best_model_name)
    print("Final F1 :", evaluator.metrics["F1 Score"])
    print("=" * 60)


if __name__ == "__main__":
    main()
