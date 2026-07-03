# ============================================================
# PREPROCESSING MODULE
# Credit Score Classification Project
# ============================================================

import re
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


class Preprocessor:
    """
    Handle all preprocessing steps including:
    - Data loading
    - Data cleaning
    - Data type conversion
    - Feature engineering
    - Train-test split
    - ColumnTransformer creation
    """

    def __init__(
        self, data_path, target_column="Credit_Score", random_state=42, test_size=0.2
    ):

        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state
        self.test_size = test_size

        self.df = None

        self.label_encoder = LabelEncoder()

        self.preprocessor = None

        self.numeric_features = []
        self.categorical_features = []

        # ----------------------------------------------------
        # Columns to remove
        # ----------------------------------------------------

        self.drop_columns = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"]

        # ----------------------------------------------------
        # Numeric columns stored as object
        # ----------------------------------------------------

        self.numeric_columns = [
            "Age",
            "Annual_Income",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Outstanding_Debt",
            "Amount_invested_monthly",
            "Monthly_Balance",
        ]

        # ----------------------------------------------------
        # Placeholder replacements
        # ----------------------------------------------------

        self.placeholder_mapping = {
            "Occupation": "_______",
            "Credit_Mix": "_",
            "Payment_Behaviour": "!@9#%8",
            "Payment_of_Min_Amount": "NM",
        }

    # ========================================================
    # LOAD DATASET
    # ========================================================

    def load_data(self):
        print("=" * 60)
        print("Loading Dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset Shape : {self.df.shape}")

        return self.df

    # ========================================================
    # PRIVATE HELPER
    # Convert Credit History Age into months
    # ========================================================

    def _convert_credit_history(self, value):
        if pd.isna(value):
            return np.nan
        
        if not isinstance(value, str):
            return np.nan
        
        match = re.search(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?", str(value))

        if match:
            years = int(match.group(1))
            months = int(match.group(2))
            return years * 12 + months

        return np.nan

    # ========================================================
    # PRIVATE HELPER
    # Remove underscore and convert to numeric
    # ========================================================

    def _convert_numeric_columns(self, df):
        for column in self.numeric_columns:
            df[column] = df[column].astype(str).str.replace("_", "", regex=False)
            df[column] = pd.to_numeric(df[column], errors="coerce")

        return df

    # ========================================================
    # CLEAN DATA
    # ========================================================

    def clean_data(self):
        """
        Clean dataset by:
        - Removing unnecessary columns
        - Replacing placeholder values
        - Converting data types
        - Handling impossible values
        """

        print("=" * 60)
        print("Cleaning Dataset...")

        df = self.df.copy()

        # ----------------------------------------------------
        # Drop unnecessary columns
        # ----------------------------------------------------

        df.drop(columns=self.drop_columns, inplace=True, errors="ignore")

        # ----------------------------------------------------
        # Replace placeholder values with NaN
        # ----------------------------------------------------

        for column, placeholder in self.placeholder_mapping.items():

            df[column] = df[column].replace(placeholder, np.nan)

        # ----------------------------------------------------
        # Convert numeric columns
        # ----------------------------------------------------

        df = self._convert_numeric_columns(df)

        # ----------------------------------------------------
        # Convert Credit History Age
        # Example:
        # 12 Years and 5 Months -> 149
        # ----------------------------------------------------

        df["Credit_History_Age"] = df["Credit_History_Age"].apply(
            self._convert_credit_history
        )

        # ----------------------------------------------------
        # Delay From Due Date
        # Negative delay is impossible
        # ----------------------------------------------------

        df["Delay_from_due_date"] = df["Delay_from_due_date"].clip(lower=0)

        # ----------------------------------------------------
        # Age
        # Keep only reasonable ages
        # ----------------------------------------------------

        df.loc[(df["Age"] < 0) | (df["Age"] > 100), "Age"] = np.nan

        # ----------------------------------------------------
        # Changed Credit Limit
        # Negative credit limit change is invalid
        # ----------------------------------------------------

        df.loc[df["Changed_Credit_Limit"] < 0, "Changed_Credit_Limit"] = np.nan

        # ----------------------------------------------------
        # Monthly Balance
        # Balance cannot be negative
        # ----------------------------------------------------

        df.loc[df["Monthly_Balance"] < 0, "Monthly_Balance"] = np.nan
        # ----------------------------------------------------
        # Amount Invested Monthly
        # Investment cannot be negative
        # ----------------------------------------------------

        df.loc[df["Amount_invested_monthly"] < 0, "Amount_invested_monthly"] = np.nan

        # ----------------------------------------------------
        # Number of Loans
        # ----------------------------------------------------

        df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan

        # ----------------------------------------------------
        # Number of Delayed Payments
        # ----------------------------------------------------

        df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan

        # ----------------------------------------------------
        # Annual Income
        # ----------------------------------------------------

        df.loc[df["Annual_Income"] <= 0, "Annual_Income"] = np.nan

        # ----------------------------------------------------
        # Outstanding Debt
        # ----------------------------------------------------

        df.loc[df["Outstanding_Debt"] < 0, "Outstanding_Debt"] = np.nan

        # ----------------------------------------------------
        # Monthly Inhand Salary
        # ----------------------------------------------------

        df.loc[df["Monthly_Inhand_Salary"] <= 0, "Monthly_Inhand_Salary"] = np.nan

        # ----------------------------------------------------
        # Credit Utilization Ratio
        # Must be between 0 and 100
        # ----------------------------------------------------

        df.loc[
            (df["Credit_Utilization_Ratio"] < 0)
            | (df["Credit_Utilization_Ratio"] > 100),
            "Credit_Utilization_Ratio",
        ] = np.nan

        # ----------------------------------------------------
        # Total EMI
        # ----------------------------------------------------

        df.loc[df["Total_EMI_per_month"] < 0, "Total_EMI_per_month"] = np.nan

        # ----------------------------------------------------
        # Store cleaned dataframe
        # ----------------------------------------------------

        self.df = df
        print("Cleaning completed.")
        print(f"Dataset Shape : {self.df.shape}")
        missing = self.df.isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            print("\nRemaining Missing Values")
            print(missing.sort_values(ascending=False))
        
        return self.df

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    def prepare_features(self):
        """
        Separate features and target,
        encode target labels,
        and perform train-test split.
        """

        print("=" * 60)
        print("Preparing Features...")

        # ----------------------------------------------------
        # Separate Features and Target
        # ----------------------------------------------------

        X = self.df.drop(columns=self.target_column)
        y = self.label_encoder.fit_transform(self.df[self.target_column])

        # ----------------------------------------------------
        # Train Test Split
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        print(f"Training Samples : {X_train.shape[0]}")
        print(f"Testing Samples  : {X_test.shape[0]}")

        return (X_train, X_test, y_train, y_test)

    # ========================================================
    # BUILD COLUMN TRANSFORMER
    # ========================================================

    def build_preprocessor(self, X_train):
        """
        Create preprocessing pipeline
        consisting of:
        - Median imputation + scaling
        - Most frequent imputation + one-hot encoding
        """

        print("=" * 60)
        print("Building ColumnTransformer...")

        # ----------------------------------------------------
        # Detect Feature Types
        # ----------------------------------------------------

        self.numeric_features = X_train.select_dtypes(
            include=np.number
        ).columns.tolist()

        self.categorical_features = X_train.select_dtypes(
            include=["object"]
        ).columns.tolist()

        # ----------------------------------------------------
        # Numerical Pipeline
        # ----------------------------------------------------

        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # ----------------------------------------------------
        # Categorical Pipeline
        # ----------------------------------------------------

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        
        # ----------------------------------------------------
        # Column Transformer
        # ----------------------------------------------------

        self.preprocessor = ColumnTransformer(
            [
                ("numerical", numeric_pipeline, self.numeric_features),
                ("categorical", categorical_pipeline, self.categorical_features),
            ]
        )
        X_train_transformed = pd.DataFrame(
            self.preprocessor.fit_transform(X_train)
        )

        print(X_train_transformed.isna().sum().sum())

        print(f"Numerical Features   : {len(self.numeric_features)}")
        print(f"Categorical Features : {len(self.categorical_features)}")

        return self.preprocessor

    # ========================================================
    # COMPLETE PREPROCESSING PIPELINE
    # ========================================================

    def preprocess(self):
        """
        Execute the complete preprocessing workflow.
        """

        self.load_data()
        self.clean_data()
        X_train, X_test, y_train, y_test = self.prepare_features()
        preprocessor = self.build_preprocessor(X_train)

        return (X_train, X_test, y_train, y_test, preprocessor, self.label_encoder)
