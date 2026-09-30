# Customer-Churn-Analysis
# ================================================================
# CUSTOMER CHURN ANALYSIS
# ================================================================
# Objective:
# 1. Calculate churn rate
# 2. Calculate Customer Lifetime Value (CLV)
# 3. Calculate Monthly Recurring Revenue (MRR) loss
# 4. Analyze churn by contract type, tenure and payment method
# 5. Generate bar plots, boxplots and heatmaps
# 6. Identify important churn drivers
# 7. Generate business recommendations
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------

FILE_NAME = "churn.csv"

df = pd.read_csv(FILE_NAME)

print("=" * 70)
print("CUSTOMER CHURN ANALYSIS")
print("=" * 70)

print("\nDataset loaded successfully.")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nFirst 5 records:")
print(df.head())


# ------------------------------------------------
# 2. DATA INFORMATION
# ------------------------------------------------

print("\n" + "=" * 70)
print("DATA INFORMATION")
print("=" * 70)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())


# ------------------------------------------------
# 3. REMOVE DUPLICATES
# ------------------------------------------------

df = df.drop_duplicates()

print("\nAfter removing duplicates:")
print("Rows:", len(df))


# ------------------------------------------------
# 4. REMOVE CUSTOMER ID COLUMN
# ------------------------------------------------

id_columns = [
    "customerID",
    "CustomerID",
    "customer_id",
    "ID"
]

for column in id_columns:
    if column in df.columns:
        df.drop(column, axis=1, inplace=True)
        print("Removed ID column:", column)


# ------------------------------------------------
# 5. FIND CHURN COLUMN
# ------------------------------------------------

churn_candidates = [
    "Churn",
    "churn",
    "Churned",
    "churned",
    "Exited",
    "exited"
]

churn_col = None

for column in churn_candidates:
    if column in df.columns:
        churn_col = column
        break

if churn_col is None:
    raise ValueError(
        "Churn column not found. Rename your churn column to 'Churn'."
    )

print("\nChurn column:", churn_col)


# ------------------------------------------------
# 6. CONVERT TOTAL CHARGES
# ------------------------------------------------

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )


# ------------------------------------------------
# 7. HANDLE MISSING VALUES
# ------------------------------------------------

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    if df[column].isnull().sum() > 0:

        df[column] = df[column].fillna(
            df[column].mode()[0]
        )


# ------------------------------------------------
# 8. CREATE CHURN FLAG
# ------------------------------------------------

def convert_churn(value):

    value = str(value).strip().lower()

    if value in [
        "yes",
        "y",
        "true",
        "1",
        "churn",
        "churned",
        "exited"
    ]:
        return 1

    return 0


df["Churn_Flag"] = df[churn_col].apply(
    convert_churn
)


# ------------------------------------------------
# 9. BASIC CHURN CALCULATION
# ------------------------------------------------

total_customers = len(df)

churned_customers = df[
    "Churn_Flag"
].sum()

retained_customers = (
    total_customers -
    churned_customers
)

churn_rate = (
    churned_customers /
    total_customers
) * 100

retention_rate = (
    retained_customers /
    total_customers
) * 100


print("\n" + "=" * 70)
print("BASELINE CHURN ANALYSIS")
print("=" * 70)

print("Total customers      :", total_customers)
print("Churned customers    :", churned_customers)
print("Retained customers   :", retained_customers)
print("Churn rate            : {:.2f}%".format(
    churn_rate
))
print("Retention rate        : {:.2f}%".format(
    retention_rate
))


# ------------------------------------------------
# 10. FIND MONTHLY CHARGES COLUMN
# ------------------------------------------------

monthly_candidates = [
    "MonthlyCharges",
    "MonthlyCharge",
    "Monthly_Revenue",
    "monthly_charges",
    "monthly_revenue"
]

monthly_col = None

for column in monthly_candidates:

    if column in df.columns:
        monthly_col = column
        break

if monthly_col is None:

    raise ValueError(
        "MonthlyCharges column not found."
    )


# Convert to numeric

df[monthly_col] = pd.to_numeric(
    df[monthly_col],
    errors="coerce"
)

df[monthly_col] = df[monthly_col].fillna(
    df[monthly_col].median()
)


# ------------------------------------------------
# 11. MRR CALCULATION
# ------------------------------------------------

total_mrr = df[
    monthly_col
].sum()

mrr_loss = df[
    df["Churn_Flag"] == 1
][monthly_col].sum()

remaining_mrr = df[
    df["Churn_Flag"] == 0
][monthly_col].sum()


print("\n" + "=" * 70)
print("MONTHLY RECURRING REVENUE")
print("=" * 70)

print(
    "Total MRR       : ${:,.2f}".format(
        total_mrr
    )
)

print(
    "MRR lost        : ${:,.2f}".format(
        mrr_loss
    )
)

print(
    "Remaining MRR   : ${:,.2f}".format(
        remaining_mrr
    )
)


# ------------------------------------------------
# 12. CUSTOMER LIFETIME VALUE
# ------------------------------------------------

average_monthly_revenue = df[
    monthly_col
].mean()

monthly_churn_rate = (
    churn_rate / 100
)

if monthly_churn_rate > 0:

    expected_lifetime = (
        1 / monthly_churn_rate
    )

else:

    expected_lifetime = np.inf


clv = (
    average_monthly_revenue *
    expected_lifetime
)


print("\n" + "=" * 70)
print("CUSTOMER LIFETIME VALUE")
print("=" * 70)

print(
    "Average monthly revenue : ${:.2f}".format(
        average_monthly_revenue
    )
)

print(
    "Expected lifetime       : {:.2f} months".format(
        expected_lifetime
    )
)

print(
    "Estimated CLV           : ${:,.2f}".format(
        clv
    )
)


# ------------------------------------------------
# 13. FIND CONTRACT COLUMN
# ------------------------------------------------

contract_candidates = [
    "Contract",
    "contract",
    "ContractType",
    "contract_type"
]

contract_col = None

for column in contract_candidates:

    if column in df.columns:
        contract_col = column
        break


# ------------------------------------------------
# 14. CHURN BY CONTRACT
# ------------------------------------------------

if contract_col is not None:

    contract_churn = (
        df.groupby(contract_col)
        ["Churn_Flag"]
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("CHURN BY CONTRACT TYPE")
    print("=" * 70)

    print(
        contract_churn
    )

else:

    contract_churn = None


# ------------------------------------------------
# 15. FIND TENURE COLUMN
# ------------------------------------------------

tenure_candidates = [
    "tenure",
    "Tenure",
    "TenureMonths",
    "tenure_months"
]

tenure_col = None

for column in tenure_candidates:

    if column in df.columns:
        tenure_col = column
        break


# ------------------------------------------------
# 16. CHURN BY TENURE
# ------------------------------------------------

if tenure_col is not None:

    df[tenure_col] = pd.to_numeric(
        df[tenure_col],
        errors="coerce"
    )

    df[tenure_col] = df[
        tenure_col
    ].fillna(
        df[tenure_col].median()
    )

    df["Tenure_Group"] = pd.cut(
        df[tenure_col],
        bins=[
            -1,
            12,
            24,
            48,
            72,
            np.inf
        ],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months",
            "72+ months"
        ]
    )

    tenure_churn = (
        df.groupby(
            "Tenure_Group",
            observed=True
        )["Churn_Flag"]
        .mean()
        .mul(100)
    )

    print("\n" + "=" * 70)
    print("CHURN BY TENURE")
    print("=" * 70)

    print(
        tenure_churn
    )

else:

    tenure_churn = None


# ------------------------------------------------
# 17. FIND PAYMENT METHOD COLUMN
# ------------------------------------------------

payment_candidates = [
    "PaymentMethod",
    "payment_method",
    "Payment_Method",
    "payment"
]

payment_col = None

for column in payment_candidates:

    if column in df.columns:
        payment_col = column
        break


# ------------------------------------------------
# 18. CHURN BY PAYMENT METHOD
# ------------------------------------------------

if payment_col is not None:

    payment_churn = (
        df.groupby(payment_col)
        ["Churn_Flag"]
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    print("\n" + "=" * 70)
    print("CHURN BY PAYMENT METHOD")
    print("=" * 70)

    print(
        payment_churn
    )

else:

    payment_churn = None


# ================================================================
# VISUALIZATION
# ================================================================

sns.set_theme(
    style="whitegrid"
)


# ------------------------------------------------
# 19. CHURN DISTRIBUTION
# ------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

churn_counts = df[
    "Churn_Flag"
].value_counts()

plt.bar(
    ["Retained", "Churned"],
    [
        churn_counts.get(0, 0),
        churn_counts.get(1, 0)
    ]
)

plt.title(
    "Customer Churn Distribution"
)

plt.xlabel(
    "Customer Status"
)

plt.ylabel(
    "Number of Customers"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------
# 20. CONTRACT CHURN BAR CHART
# ------------------------------------------------

if contract_churn is not None:

    plt.figure(
        figsize=(9, 5)
    )

    contract_churn.sort_values().plot(
        kind="bar"
    )

    plt.title(
        "Churn Rate by Contract Type"
    )

    plt.xlabel(
        "Contract Type"
    )

    plt.ylabel(
        "Churn Rate (%)"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------
# 21. TENURE CHURN BAR CHART
# ------------------------------------------------

if tenure_churn is not None:

    plt.figure(
        figsize=(9, 5)
    )

    tenure_churn.plot(
        kind="bar"
    )

    plt.title(
        "Churn Rate by Customer Tenure"
    )

    plt.xlabel(
        "Tenure Group"
    )

    plt.ylabel(
        "Churn Rate (%)"
    )

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------
# 22. PAYMENT METHOD CHURN
# ------------------------------------------------

if payment_churn is not None:

    plt.figure(
        figsize=(10, 5)
    )

    payment_churn.sort_values().plot(
        kind="bar"
    )

    plt.title(
        "Churn Rate by Payment Method"
    )

    plt.xlabel(
        "Payment Method"
    )

    plt.ylabel(
        "Churn Rate (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.show()


# ------------------------------------------------
# 23. MONTHLY CHARGES BOXPLOT
# ------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(
    data=df,
    x="Churn_Flag",
    y=monthly_col
)

plt.title(
    "Monthly Charges vs Churn"
)

plt.xlabel(
    "Customer Status"
)

plt.ylabel(
    "Monthly Charges"
)

plt.xticks(
    [0, 1],
    ["Retained", "Churned"]
)

plt.tight_layout()

plt.show()


# ------------------------------------------------
# 24. AVERAGE MONTHLY CHARGES
# ------------------------------------------------

average_charges = (
    df.groupby(
        "Churn_Flag"
    )[monthly_col]
    .mean()
)

print("\n" + "=" * 70)
print("AVERAGE MONTHLY CHARGES")
print("=" * 70)

print(
    average_charges
)


plt.figure(
    figsize=(8, 5)
)

average_charges.plot(
    kind="bar"
)

plt.title(
    "Average Monthly Charges by Churn Status"
)

plt.xlabel(
    "Churn Status"
)

plt.ylabel(
    "Average Monthly Charges"
)

plt.xticks(
    [0, 1],
    ["Retained", "Churned"],
    rotation=0
)

plt.tight_layout()

plt.show()


# ------------------------------------------------
# 25. CORRELATION HEATMAP
# ------------------------------------------------

numeric_data = df.select_dtypes(
    include=np.number
)

plt.figure(
    figsize=(12, 8)
)

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.show()


# ------------------------------------------------
# 26. AUTOMATIC KEY DRIVER ANALYSIS
# ------------------------------------------------

print("\n" + "=" * 70)
print("KEY BEHAVIORAL DRIVERS OF CHURN")
print("=" * 70)


if contract_churn is not None:

    highest_contract = (
        contract_churn.idxmax()
    )

    highest_contract_rate = (
        contract_churn.max()
    )

    print(
        "\nHighest contract-related churn:"
    )

    print(
        "{} : {:.2f}%".format(
            highest_contract,
            highest_contract_rate
        )
    )


if tenure_churn is not None:

    highest_tenure = (
        tenure_churn.idxmax()
    )

    highest_tenure_rate = (
        tenure_churn.max()
    )

    print(
        "\nHighest tenure-related churn:"
    )

    print(
        "{} : {:.2f}%".format(
            highest_tenure,
            highest_tenure_rate
        )
    )


if payment_churn is not None:

    highest_payment = (
        payment_churn.idxmax()
    )

    highest_payment_rate = (
        payment_churn.max()
    )

    print(
        "\nHighest payment-method churn:"
    )

    print(
        "{} : {:.2f}%".format(
            highest_payment,
            highest_payment_rate
        )
    )


print(
    "\nOverall churn rate: {:.2f}%".format(
        churn_rate
    )
)

print(
    "MRR lost because of churn: ${:,.2f}".format(
        mrr_loss
    )
)


# ================================================================
# 27. FINAL SUMMARY TABLE
# ================================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Churned Customers",
        "Retained Customers",
        "Churn Rate",
        "Retention Rate",
        "Average Monthly Revenue",
        "Total MRR",
        "MRR Lost",
        "Remaining MRR",
        "Expected Lifetime",
        "Estimated CLV"
    ],

    "Value": [
        total_customers,
        churned_customers,
        retained_customers,
        "{:.2f}%".format(churn_rate),
        "{:.2f}%".format(retention_rate),
        "${:.2f}".format(
            average_monthly_revenue
        ),
        "${:,.2f}".format(
            total_mrr
        ),
        "${:,.2f}".format(
            mrr_loss
        ),
        "${:,.2f}".format(
            remaining_mrr
        ),
        "{:.2f} months".format(
            expected_lifetime
        ),
        "${:,.2f}".format(
            clv
        )
    ]
})


print("\n" + "=" * 70)
print("FINAL ANALYSIS SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False
    )
)


# ------------------------------------------------
# 28. BUSINESS RECOMMENDATIONS
# ------------------------------------------------

print("\n" + "=" * 70)
print("BUSINESS RECOMMENDATIONS")
print("=" * 70)

print("""
1. Identify high-risk customers early using their churn behavior.

2. Improve onboarding and customer support, especially for new customers.

3. Encourage customers to choose longer-term contracts where appropriate.

4. Monitor payment-related problems and provide convenient payment options.

5. Provide targeted retention offers to customers showing signs of churn.

6. Pay special attention to high-value customers because their churn
   can result in significant MRR loss.

7. Continuously monitor churn rate, retention rate, CLV and MRR loss.

8. Build a monthly dashboard to track customer behavior and churn trends.
""")


# ------------------------------------------------
# 29. SAVE CLEANED DATA
# ------------------------------------------------

df.to_csv(
    "cleaned_customer_churn.csv",
    index=False
)

summary.to_csv(
    "churn_analysis_summary.csv",
    index=False
)


# ------------------------------------------------
# 30. FINAL MESSAGE
# ------------------------------------------------

print("\n" + "=" * 70)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Files created:
1. cleaned_customer_churn.csv
2. churn_analysis_summary.csv

The analysis includes:
- Churn rate
- Retention rate
- Customer Lifetime Value (CLV)
- Monthly Recurring Revenue (MRR)
- MRR loss
- Contract analysis
- Tenure analysis
- Payment method analysis
- Bar charts
- Boxplots
- Correlation heatmap
- Key churn drivers
- Business recommendations
""")