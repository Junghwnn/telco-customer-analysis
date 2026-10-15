# Telco Customer Analysis

## Central Question

What factors are associated with higher monthly charges?

## Dataset

This project uses the IBM Telco Customer Churn dataset from Kaggle.
The dataset contains information about 7,043 telecom customers,
including their services, contracts, tenure, monthly charges, and churn status.

## App

The Streamlit app allows users to:

- Filter customers by contract type and internet service
- Explore monthly charge distributions
- Compare monthly charges across internet service types
- Explore churn rates across contract types
- Build a linear regression model to predict monthly charges
- Compare a baseline model with a model using customer service and contract information

## Model

The target variable is `MonthlyCharges`.

The data is split into 80% training and 20% testing data.

Categorical variables are converted using one-hot encoding with
`drop_first=True`.

Model performance is evaluated using RMSE.

## Main Findings

Customers with fiber optic internet tend to have substantially higher
monthly charges than customers with DSL or no internet service.

Streaming TV and streaming movies are also associated with higher
predicted monthly charges.

Month-to-month customers have a substantially higher churn rate than
customers with one-year or two-year contracts.

These findings represent associations and predictive relationships,
not causal effects.

## Faithfulness and Granularity

The visualizations summarize patterns directly from the dataset.
Aggregated charts show averages or churn rates for customer groups,
while the scatter plots preserve customer-level observations.

The analysis does not claim that the observed relationships are causal.

## Live App

[Add Streamlit link here]