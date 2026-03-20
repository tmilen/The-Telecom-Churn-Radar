import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------------------
#                                Page Config
# ---------------------------------------------------------------------------------------
st.set_page_config(
    page_title="The Telecom Churn Radar",
    page_icon="📡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_model.pkl"


# ---------------------------------------------------------------------------------------
#                                Load Model
# ---------------------------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: Path):
    return joblib.load(model_path)


model_pipeline = load_model(MODEL_PATH)


st.title("📡 The Telecom Churn Radar")
st.write(
    "This web app predicts whether a telecom customer is likely to churn "
    "based on service, invoice, payment, and behavioral information."
)


# ---------------------------------------------------------------------------------------
#                               Build Input DataFrame
# ---------------------------------------------------------------------------------------
def build_input_dataframe(
    service_type,
    customer_category,
    township,
    plan_name,
    active_month,
    active_days,
    suspend_duration_month,
    suspend_duration_days,
    no_of_mrc_invoice,
    no_of_oti_invoice,
    total_mrc_invoice_amount,
    total_oti_invoice_amount,
    total_payment,
    balance,
    advance_benefit_offered,
    advance_benefit_consumed,
    advance_benefit_balance
):
    input_data = pd.DataFrame(
        [{
            "Service Type": service_type,
            "Customer Category": customer_category,
            "Township": township,
            "Plan Name": plan_name,
            "Active Month": active_month,
            "Active Days": active_days,
            "Suspend Duration in Month": suspend_duration_month,
            "Suspend Duration in Days": suspend_duration_days,
            "No Of MRC Invoice": no_of_mrc_invoice,
            "No Of OTI Invoice": no_of_oti_invoice,
            "Total MRC Invoice Amount": total_mrc_invoice_amount,
            "Total OTI Invoice Amount": total_oti_invoice_amount,
            "Total Payment": total_payment,
            "Balance": balance,
            "Advance Benefit Offered": advance_benefit_offered,
            "Advance Benefit Consumed": advance_benefit_consumed,
            "Advance Benefit Balance": advance_benefit_balance
        }]
    )
    return input_data


# ---------------------------------------------------------------------------------------
#                                Sidebar User Inputs
# ---------------------------------------------------------------------------------------
st.sidebar.header("Customer Input")

service_type = st.sidebar.selectbox(
    "Service Type",
    ["Postpaid", "Prepaid"]
)

customer_category = st.sidebar.text_input(
    "Customer Category",
    value="Residential"
)

township = st.sidebar.text_input(
    "Township",
    value="Hlaing"
)

plan_name = st.sidebar.text_input(
    "Plan Name",
    value="Basic Plan"
)

active_month = st.sidebar.number_input(
    "Active Month",
    min_value=0,
    value=6,
    step=1
)

active_days = st.sidebar.number_input(
    "Active Days",
    min_value=0.0,
    value=180.0,
    step=1.0
)

suspend_duration_month = st.sidebar.number_input(
    "Suspend Duration in Month",
    min_value=0,
    value=0,
    step=1
)

suspend_duration_days = st.sidebar.number_input(
    "Suspend Duration in Days",
    min_value=0.0,
    value=0.0,
    step=1.0
)

no_of_mrc_invoice = st.sidebar.number_input(
    "No Of MRC Invoice",
    min_value=0,
    value=0,
    step=1
)

no_of_oti_invoice = st.sidebar.number_input(
    "No Of OTI Invoice",
    min_value=0,
    value=0,
    step=1
)

total_mrc_invoice_amount = st.sidebar.number_input(
    "Total MRC Invoice Amount",
    value=0.0,
    step=1000.0
)

total_oti_invoice_amount = st.sidebar.number_input(
    "Total OTI Invoice Amount",
    value=0.0,
    step=1000.0
)

total_payment = st.sidebar.number_input(
    "Total Payment",
    value=0.0,
    step=1000.0
)

balance = st.sidebar.number_input(
    "Balance",
    value=0.0,
    step=1000.0
)

advance_benefit_offered = st.sidebar.number_input(
    "Advance Benefit Offered",
    value=0.0,
    step=1000.0
)

advance_benefit_consumed = st.sidebar.number_input(
    "Advance Benefit Consumed",
    value=0.0,
    step=1000.0
)

advance_benefit_balance = st.sidebar.number_input(
    "Advance Benefit Balance",
    value=0.0,
    step=1000.0
)

left_col, right_col = st.columns([1.1, 1]) #page layout

# ---------------------------------------------------------------------------------------
#                                Prediction Section
# ---------------------------------------------------------------------------------------
with left_col:
    st.subheader("Prediction")

    input_df = build_input_dataframe(
        service_type=service_type,
        customer_category=customer_category,
        township=township,
        plan_name=plan_name,
        active_month=active_month,
        active_days=active_days,
        suspend_duration_month=suspend_duration_month,
        suspend_duration_days=suspend_duration_days,
        no_of_mrc_invoice=no_of_mrc_invoice,
        no_of_oti_invoice=no_of_oti_invoice,
        total_mrc_invoice_amount=total_mrc_invoice_amount,
        total_oti_invoice_amount=total_oti_invoice_amount,
        total_payment=total_payment,
        balance=balance,
        advance_benefit_offered=advance_benefit_offered,
        advance_benefit_consumed=advance_benefit_consumed,
        advance_benefit_balance=advance_benefit_balance
    )

    st.write("### Input Customer Details")
    st.dataframe(input_df, use_container_width=True)

    if st.button("Predict Churn", use_container_width=True):
        prediction = model_pipeline.predict(input_df)[0]
        probability = model_pipeline.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"Prediction: Customer is likely to CHURN")
        else:
            st.success(f"Prediction: Customer is likely to STAY")

        st.metric("Churn Probability", f"{probability:.2%}")


# ---------------------------------------------------------------------------------------
#                                Feature Importance Section
# ---------------------------------------------------------------------------------------
with right_col:
    st.subheader("Target Graph: Feature Importance")

    try:
        classifier = model_pipeline.named_steps["classifier"]
        preprocessor = model_pipeline.named_steps["preprocessor"]

        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]

        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients
        })

        feature_importance_df["AbsCoefficient"] = feature_importance_df["Coefficient"].abs()
        top_features = feature_importance_df.sort_values(
            by="AbsCoefficient",
            ascending=False
        ).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#F56CC0" if value > 0 else "#4E36D2" for value in top_features["Coefficient"]]

        ax.barh(top_features["Feature"], top_features["Coefficient"], color=colors)
        ax.set_title("Top 10 Features Influencing Churn")
        ax.set_xlabel("Coefficient Value")
        ax.set_ylabel("Feature")
        ax.invert_yaxis()

        st.pyplot(fig)

    except Exception as error:
        st.warning(
            "Feature importance chart could not be generated. "
            "This chart works best when the final model is Logistic Regression."
        )
        st.caption(f"Details: {error}")


# ---------------------------------------------------------------------------------------
#                                Footer
# ---------------------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit | Model: Logistic Regression Pipeline | "
    "Project: The Telecom Churn Radar"
)