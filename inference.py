# ============================================================
# INFERENCE MODULE
# Credit Score Classification Project
# ============================================================
import os
import joblib
import pandas as pd

class Inference:
    """
    Load trained model and perform predictions
    on new customer data.
    """

    def __init__(
        self, model_path="models/best_pipeline.pkl", encoder_path="models/label_encoder.pkl"
    ):

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder file '{encoder_path}' not found.")

        self.pipeline = joblib.load(model_path)
        self.label_encoder = joblib.load(encoder_path)

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self, input_data):
        """
        Predict credit score for new data.

        Parameters
        ----------
        input_data : pandas.DataFrame

        Returns
        -------
        prediction : list
        """
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(
                "input_data must be a pandas DataFrame."
            )
        prediction = self.pipeline.predict(input_data)
        prediction = self.label_encoder.inverse_transform(prediction)

        return prediction[0]

    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    def predict_probability(self, input_data):
        """
        Return class probabilities.
        """

        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(input_data)
            probability_df = pd.DataFrame(
                probabilities, columns=self.label_encoder.classes_
            )

            return probability_df

        return None

    # ========================================================
    # PREDICT FROM DICTIONARY
    # ========================================================

    def predict_from_dict(self, data):
        """
        Predict from dictionary input.
        """

        dataframe = pd.DataFrame([data])
        prediction = self.predict(dataframe)
        probability = self.predict_probability(dataframe)

        return {
            "prediction": prediction[0],
            "probability": probability
        }
    
    def __repr__(self):
        return "Inference(model_loaded=True)"
