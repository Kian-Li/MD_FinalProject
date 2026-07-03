# ============================================================
# STREAMLIT APPLICATION
# Credit Score Classification System
# ============================================================

import streamlit as st
import pandas as pd

from inference import Inference


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Score Classification System",
    page_icon="credit_card",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load trained pipeline only once.
    """
    return Inference()


predictor = load_model()


# ============================================================
# SESSION STATE
# ============================================================

default_values = {
    "month": "January",
    "age": 30,
    "occupation": "Engineer",
    "annual_income": 50000.0,
    "monthly_salary": 4000.0,
    "num_bank_accounts": 2,
    "num_credit_cards": 3,
    "interest_rate": 8,
    "num_loans": 1,
    "type_of_loan": "Auto Loan",
    "delay_due": 5,
    "delayed_payment": 2,
    "changed_credit_limit": 5.0,
    "credit_inquiries": 3,
    "credit_mix": "Standard",
    "outstanding_debt": 1000.0,
    "credit_utilization": 30.0,
    "credit_history": 120,
    "payment_min": "Yes",
    "total_emi": 100.0,
    "invested_monthly": 200.0,
    "payment_behaviour": "High_spent_Medium_value_payments",
    "monthly_balance": 500.0
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Project Information")
st.sidebar.markdown("---")
with st.sidebar.expander("Application", expanded=True):
    st.write(
        """
        Credit Score Classification System
        developed using supervised machine
        learning techniques.
        """
    )

with st.sidebar.expander("Models Compared"):
    st.write("""
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
""")

with st.sidebar.expander("Preprocessing"):
    st.write("""
- Missing Value Imputation
- One-Hot Encoding
- Standard Scaling
- ColumnTransformer
- Scikit-Learn Pipeline
""")

with st.sidebar.expander("Deployment"):
    st.write("""
- Streamlit
- GitHub
- AWS Cloud
""")

st.sidebar.markdown("---")

if st.sidebar.button(
    "Load Sample Customer",
    use_container_width=True
):
    st.session_state.month = "March"
    st.session_state.age = 35
    st.session_state.occupation = "Engineer"
    st.session_state.annual_income = 72000.0
    st.session_state.monthly_salary = 6000.0
    st.session_state.num_bank_accounts = 3
    st.session_state.num_credit_cards = 4
    st.session_state.interest_rate = 6
    st.session_state.num_loans = 2
    st.session_state.type_of_loan = "Auto Loan"
    st.session_state.delay_due = 2
    st.session_state.delayed_payment = 1
    st.session_state.changed_credit_limit = 8.0
    st.session_state.credit_inquiries = 2
    st.session_state.credit_mix = "Good"
    st.session_state.outstanding_debt = 800.0
    st.session_state.credit_utilization = 24.0
    st.session_state.credit_history = 180
    st.session_state.payment_min = "Yes"
    st.session_state.total_emi = 250.0
    st.session_state.invested_monthly = 600.0
    st.session_state.payment_behaviour = (
        "High_spent_Medium_value_payments"
    )
    st.session_state.monthly_balance = 1200.0
    st.sidebar.success("Sample customer loaded.")


# ============================================================
# PAGE HEADER
# ============================================================

st.title("Credit Score Classification System")
st.caption(
    "Machine Learning-Based Customer Credit Risk Assessment"
)
st.write(
    """
This application predicts a customer's credit score
using a trained machine learning classification model.

Provide the customer's financial information,
then press **Predict Credit Score**
to generate a prediction.
"""
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric(
        label="Best Model",
        value="Random Forest"
    )

with metric_col2:
    st.metric(
        label="Target Classes",
        value="3"
    )

with metric_col3:
    st.metric(
        label="Input Features",
        value="23"
    )

st.divider()

# ============================================================
# CUSTOMER INPUT
# ============================================================

st.header("Customer Information")
tab1, tab2, tab3 = st.tabs(
    [
        "Personal Information",
        "Financial Information",
        "Credit Information"
    ]
)

# ============================================================
# TAB 1
# ============================================================

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        month = st.selectbox(
            "Month",
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August"
            ],
            key="month"
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            key="age"
        )

        occupation = st.selectbox(
            "Occupation",
            [
                "Accountant",
                "Architect",
                "Developer",
                "Doctor",
                "Engineer",
                "Entrepreneur",
                "Journalist",
                "Lawyer",
                "Manager",
                "Mechanic",
                "Media_Manager",
                "Musician",
                "Scientist",
                "Teacher",
                "Writer"
            ],
            key="occupation"
        )

        annual_income = st.number_input(
            "Annual Income",
            min_value=0.0,
            key="annual_income"
        )

        monthly_salary = st.number_input(
            "Monthly Inhand Salary",
            min_value=0.0,
            key="monthly_salary"
        )

    with col2:
        num_bank_accounts = st.number_input(
            "Number of Bank Accounts",
            min_value=0,
            key="num_bank_accounts"
        )

        num_credit_cards = st.number_input(
            "Number of Credit Cards",
            min_value=0,
            key="num_credit_cards"
        )

        interest_rate = st.number_input(
            "Interest Rate (%)",
            min_value=0,
            key="interest_rate"
        )

        num_loans = st.number_input(
            "Number of Loans",
            min_value=0,
            key="num_loans"
        )

        loan_options = [
            "Auto Loan",
            "Credit-Builder Loan",
            "Debt Consolidation Loan",
            "Home Equity Loan",
            "Home Loan",
            "Mortgage Loan",
            "Not Specified",
            "Payday Loan",
            "Personal Loan",
            "Student Loan"
        ]

        selected_loans = st.multiselect(
            "Type of Loan",
            options=loan_options,
            default=["Auto Loan"]

        )

        type_of_loan = ", ".join(selected_loans)


# ============================================================
# TAB 2
# ============================================================

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        delay_due = st.number_input(
            "Delay From Due Date",
            min_value=0,
            key="delay_due"
        )

        delayed_payment = st.number_input(
            "Number of Delayed Payments",
            min_value=0,
            key="delayed_payment"
        )

        changed_credit_limit = st.number_input(
            "Changed Credit Limit",
            key="changed_credit_limit"
        )

        credit_inquiries = st.number_input(
            "Number of Credit Inquiries",
            min_value=0,
            key="credit_inquiries"
        )

        credit_mix = st.selectbox(
            "Credit Mix",
            [
                "Good",
                "Standard",
                "Bad"
            ],
            key="credit_mix"
        )

    with col2:
        outstanding_debt = st.number_input(
            "Outstanding Debt",
            min_value=0.0,
            key="outstanding_debt"
        )

        credit_utilization = st.slider(
            "Credit Utilization Ratio",
            min_value=0.0,
            max_value=100.0,
            key="credit_utilization"
        )

        credit_history = st.number_input(
            "Credit History (Months)",
            min_value=0,
            key="credit_history"
        )

        payment_min = st.selectbox(
            "Payment of Minimum Amount",
            [
                "Yes",
                "No"
            ],
            key="payment_min"
        )

        total_emi = st.number_input(
            "Total EMI per Month",
            min_value=0.0,
            key="total_emi"
        )


# ============================================================
# TAB 3
# ============================================================

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        invested_monthly = st.number_input(
            "Amount Invested Monthly",
            min_value=0.0,
            key="invested_monthly"
        )

        payment_behaviour = st.selectbox(
            "Payment Behaviour",
            [
                "High_spent_Large_value_payments",
                "High_spent_Medium_value_payments",
                "High_spent_Small_value_payments",
                "Low_spent_Large_value_payments",
                "Low_spent_Medium_value_payments",
                "Low_spent_Small_value_payments"
            ],
            key="payment_behaviour"
        )

    with col2:
        monthly_balance = st.number_input(
            "Monthly Balance",
            key="monthly_balance"
        )

st.divider()

# ============================================================
# INPUT DATAFRAME
# ============================================================

input_data = {
    "Month": month,
    "Age": age,
    "Occupation": occupation,
    "Annual_Income": annual_income,
    "Monthly_Inhand_Salary": monthly_salary,
    "Num_Bank_Accounts": num_bank_accounts,
    "Num_Credit_Card": num_credit_cards,
    "Interest_Rate": interest_rate,
    "Num_of_Loan": num_loans,
    "Type_of_Loan": type_of_loan,
    "Delay_from_due_date": delay_due,
    "Num_of_Delayed_Payment": delayed_payment,
    "Changed_Credit_Limit": changed_credit_limit,
    "Num_Credit_Inquiries": credit_inquiries,
    "Credit_Mix": credit_mix,
    "Outstanding_Debt": outstanding_debt,
    "Credit_Utilization_Ratio": credit_utilization,
    "Credit_History_Age": credit_history,
    "Payment_of_Min_Amount": payment_min,
    "Total_EMI_per_month": total_emi,
    "Amount_invested_monthly": invested_monthly,
    "Payment_Behaviour": payment_behaviour,
    "Monthly_Balance": monthly_balance
}


# ============================================================
# PREDICTION
# ============================================================

st.header("Prediction")
predict_button = st.button(
    "Predict Credit Score",
    use_container_width=True,
    type="primary"
)

if predict_button:
    try:
        with st.spinner("Running prediction..."):
            result = predictor.predict_from_dict(
                input_data
            )

        prediction = result["prediction"]
        probability = result["probability"]

        # -----------------------------------------------
        # Prediction Card
        # -----------------------------------------------

        st.subheader("Prediction Result")

        if prediction == "Good":
            st.success(
                f"Predicted Credit Score: {prediction}"
            )

        elif prediction == "Standard":
            st.info(
                f"Predicted Credit Score: {prediction}"
            )

        else:
            st.error(
                f"Predicted Credit Score: {prediction}"
            )

        # -----------------------------------------------
        # Confidence Score
        # -----------------------------------------------

        if probability is not None:
            confidence = probability.max(axis=1).iloc[0]
            st.subheader("Prediction Confidence")
            st.progress(float(confidence))
            st.write(
                f"Confidence Score: {confidence:.2%}"
            )

        # -----------------------------------------------
        # Probability Table
        # -----------------------------------------------

        if probability is not None:
            st.subheader("Prediction Probability")
            st.dataframe(
                probability.style.format("{:.2%}"),
                use_container_width=True
            )

        # -----------------------------------------------
        # Probability Chart
        # -----------------------------------------------

        if probability is not None:
            st.subheader("Probability Distribution")
            st.bar_chart(
                probability.T
            )
    except Exception as error:
        st.error(
            f"Prediction failed.\n\n{error}"
        )

st.divider()

# ============================================================
# CUSTOMER SUMMARY
# ============================================================

with st.expander(
    "Customer Summary",
    expanded=False
):
    summary_df = pd.DataFrame(
        [input_data]
    )

    st.dataframe(
        summary_df,
        use_container_width=True
    )

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "Model Information",
    expanded=False
):
    st.markdown("""
    ### Model Pipeline

    This application uses a complete Scikit-Learn Pipeline:

    1. Missing Value Imputation
    2. Feature Engineering
    3. ColumnTransformer
    4. One-Hot Encoding
    5. Standard Scaling
    6. Machine Learning Classifier

    The final deployed model was selected after
    baseline comparison and hyperparameter tuning.
    """)

st.divider()


# ============================================================
# SYSTEM INFORMATION
# ============================================================

with st.expander(
    "System Information",
    expanded=False
):
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Machine Learning Task",
            "Classification"
        )

        st.metric(
            "Target Classes",
            "3"
        )

    with col2:
        st.metric(
            "Deployment",
            "Streamlit"
        )

        st.metric(
            "Framework",
            "Scikit-Learn"
        )

st.divider()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
---
<center>

### Credit Score Classification System

Machine Learning Classification Project

Built using

**Python • Scikit-Learn • XGBoost • Streamlit • MLflow**

</center>
""",
    unsafe_allow_html=True
)