import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Telco Customer Analysis",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    return df


df = load_data()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("What factors are associated with higher monthly charges?")

st.markdown(
    """
    This app explores IBM's Telco Customer Churn dataset to understand
    how customer characteristics, services, and contracts are related to
    monthly charges. It also uses a linear regression model to predict
    monthly charges.
    """
)


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Filters")

contract_options = sorted(df["Contract"].unique())

selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    contract_options,
    default=contract_options
)

internet_options = sorted(df["InternetService"].unique())

selected_internet = st.sidebar.multiselect(
    "Internet Service",
    internet_options,
    default=internet_options
)


filtered = df[
    (df["Contract"].isin(selected_contracts))
    &
    (df["InternetService"].isin(selected_internet))
]


if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()


# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------

st.header("Customer Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Customers",
    len(filtered)
)

c2.metric(
    "Avg Monthly Charges",
    f"${filtered['MonthlyCharges'].mean():.2f}"
)

c3.metric(
    "Avg Tenure",
    f"{filtered['tenure'].mean():.1f} months"
)

churn_rate = (
    filtered["Churn"].eq("Yes").mean() * 100
)

c4.metric(
    "Churn Rate",
    f"{churn_rate:.1f}%"
)


st.divider()


# --------------------------------------------------
# MONTHLY CHARGES DISTRIBUTION
# --------------------------------------------------

st.header("Exploration")

st.subheader("Monthly Charges Distribution")

fig, ax = plt.subplots()

sns.histplot(
    data=filtered,
    x="MonthlyCharges",
    bins=25,
    ax=ax
)

ax.set_title("Distribution of Monthly Charges")
ax.set_xlabel("Monthly Charges ($)")
ax.set_ylabel("Number of Customers")

st.pyplot(fig)

st.caption(
    "Monthly charges vary widely across customers, reflecting differences in services and plans."
)


# --------------------------------------------------
# TENURE VS MONTHLY CHARGES
# --------------------------------------------------

st.subheader("Tenure vs. Monthly Charges")

fig, ax = plt.subplots()

sns.scatterplot(
    data=filtered,
    x="tenure",
    y="MonthlyCharges",
    hue="Churn",
    alpha=0.6,
    ax=ax
)

ax.set_title("Tenure and Monthly Charges")
ax.set_xlabel("Tenure (Months)")
ax.set_ylabel("Monthly Charges ($)")

st.pyplot(fig)

st.caption(
    "This chart shows the relationship between customer tenure and monthly charges, with churn status shown for context."
)


# --------------------------------------------------
# AVG MONTHLY CHARGES BY INTERNET SERVICE
# --------------------------------------------------

st.subheader("Average Monthly Charges by Internet Service")

internet_charges = (
    filtered
    .groupby("InternetService")["MonthlyCharges"]
    .mean()
    .sort_values()
)

fig, ax = plt.subplots()

internet_charges.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Average Monthly Charges by Internet Service")
ax.set_xlabel("Internet Service")
ax.set_ylabel("Average Monthly Charges ($)")

plt.xticks(rotation=0)

st.pyplot(fig)

st.caption(
    "Average monthly charges differ substantially depending on the type of internet service."
)


# --------------------------------------------------
# CHURN EXPLORATION
# --------------------------------------------------

st.divider()

st.header("Churn Exploration")

st.subheader("Churn Rate by Contract Type")

churn_by_contract = (
    filtered
    .groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .sort_values()
)

fig, ax = plt.subplots()

churn_by_contract.plot(
    kind="bar",
    ax=ax
)

ax.set_title("Churn Rate by Contract Type")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Churn Rate (%)")

plt.xticks(rotation=0)

st.pyplot(fig)

st.caption(
    "Month-to-month customers tend to have a much higher churn rate than customers with longer contracts."
)


# --------------------------------------------------
# LINEAR REGRESSION MODEL
# --------------------------------------------------

st.divider()

st.header("Predicting Monthly Charges")

st.markdown(
    """
    The target variable is **MonthlyCharges**.

    The baseline model uses only **tenure**.

    The full model adds customer contract and service characteristics.
    """
)


model_columns = [
    "tenure",
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "MonthlyCharges"
]

model_df = df[model_columns].copy()


# --------------------------------------------------
# ONE-HOT ENCODING
# --------------------------------------------------

categorical_columns = [
    "Contract",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

encoded_df = pd.get_dummies(
    model_df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)
# Remove redundant "No internet service" dummy variables.
# InternetService_No already captures this information.
redundant_columns = [
    c for c in encoded_df.columns
    if c.endswith("_No internet service")
]

encoded_df = encoded_df.drop(columns=redundant_columns)

X = encoded_df.drop(
    columns="MonthlyCharges"
)

y = encoded_df["MonthlyCharges"]


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# FIT MODELS
# --------------------------------------------------

@st.cache_resource
def fit_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Baseline: tenure only
baseline_model = fit_model(
    X_train[["tenure"]],
    y_train
)

baseline_predictions = baseline_model.predict(
    X_test[["tenure"]]
)


# Full model
full_model = fit_model(
    X_train,
    y_train
)

full_predictions = full_model.predict(
    X_test
)


# --------------------------------------------------
# RMSE
# --------------------------------------------------

baseline_rmse = mean_squared_error(
    y_test,
    baseline_predictions
) ** 0.5


full_rmse = mean_squared_error(
    y_test,
    full_predictions
) ** 0.5


m1, m2 = st.columns(2)

m1.metric(
    "Baseline RMSE",
    f"${baseline_rmse:.2f}"
)

m2.metric(
    "Full Model RMSE",
    f"${full_rmse:.2f}"
)

st.caption(
    "Lower RMSE means better predictions. The full model uses service and contract information in addition to tenure."
)


# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

st.subheader("Actual vs. Predicted Monthly Charges")

fig, ax = plt.subplots()

ax.scatter(
    y_test,
    full_predictions,
    alpha=0.5
)

minimum = min(
    y_test.min(),
    full_predictions.min()
)

maximum = max(
    y_test.max(),
    full_predictions.max()
)

ax.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red"
)

ax.set_xlabel("Actual Monthly Charges ($)")
ax.set_ylabel("Predicted Monthly Charges ($)")
ax.set_title("Actual vs. Predicted Monthly Charges")

st.pyplot(fig)

st.caption(
    "Points closer to the red diagonal line represent more accurate predictions."
)
st.info(
    """
    Key finding: Internet service type and optional services such as
    streaming appear to explain much more variation in monthly charges
    than tenure alone.
    """
)

# --------------------------------------------------
# MODEL COEFFICIENTS
# --------------------------------------------------

st.subheader("Model Coefficients")

coefficients = pd.Series(
    full_model.coef_,
    index=X_train.columns
)

coefficient_table = (
    coefficients
    .sort_values(
        key=abs,
        ascending=False
    )
    .rename("Coefficient")
    .to_frame()
)

st.dataframe(
    coefficient_table
)

st.caption(
    """
    Coefficients show how each feature is associated with predicted
    monthly charges while holding the other variables constant.
    Positive values are associated with higher predicted charges,
    while negative values are associated with lower predicted charges.
    """
)


# --------------------------------------------------
# DATA NOTES
# --------------------------------------------------

with st.expander("Data and Modeling Notes"):
    st.markdown(
        """
        **Target:** MonthlyCharges

        **Train/test split:** 80% training, 20% testing

        **Random state:** 42

        **Categorical variables:** One-hot encoded using `drop_first=True`

        **Important:** These results show associations and predictive
        relationships. They do not establish causal effects.
        """
    )


# --------------------------------------------------
# OPTIONAL RAW DATA
# --------------------------------------------------

if st.checkbox("Show filtered raw data"):
    st.dataframe(filtered)