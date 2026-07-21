import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="DHL Shipment Dashboard",
    page_icon="📦",
    layout="wide"
)

# Title
st.title("📦 DHL Shipment Analytics using EDA with Streamlit Dashboard")

st.write("Welcome to the DHL Shipment Dashboard")

# Load Dataset
df = pd.read_csv("DHL_Shipment_Data.csv")

# Sidebar Filter
st.sidebar.title("Dashboard Filters")
region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + sorted(df["region"].unique().tolist())
)
if region != "All":
    df = df[df["region"] == region]

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset Information
st.subheader("Dataset Shape")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

# Dataset Information
st.subheader("Dataset Information")

st.write("Column Names:")
st.write(df.columns.tolist())

# Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Delivery Status")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(
    x="delivery_status",
    data=df,
    palette="Set2",
    ax=ax
)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Delivery Mode Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(
    x="delivery_mode",
    data=df,
    palette="Set1",
    ax=ax
)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Weather Condition Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.countplot(
    x="weather_condition",
    data=df,
    palette="Dark2",
    ax=ax
)
plt.xticks(rotation=45)
st.pyplot(fig)

# Delivery Cost Distribution
st.subheader("Delivery Cost Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(
    df["delivery_cost"],
    bins=20,
    kde=True,
    color="royalblue",
    ax=ax
)
st.pyplot(fig)


# Distance Distribution
st.subheader("Distance Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(
    df["distance_km"],
    bins=20,
    kde=True,
    color="green",
    ax=ax
)
st.pyplot(fig)

# Package Weight Distribution
st.subheader("Package Weight Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.histplot(
    df["package_weight_kg"],
    bins=20,
    kde=True,
    color="red",
    ax=ax
)
st.pyplot(fig)


# Delivery Cost vs Delivery Status
st.subheader("Delivery Cost vs Delivery Status")
fig, ax = plt.subplots(figsize=(7,4))
sns.boxplot(
    x="delivery_status",
    y="delivery_cost",
    data=df,
    color="gold",
    ax=ax
)
plt.xticks(rotation=45)
st.pyplot(fig)

# Distance vs Delivery Cost (Scatter Plot)
st.subheader("Distance vs Delivery Cost")
fig, ax = plt.subplots(figsize=(7,4))
sns.scatterplot(
    x="distance_km",
    y="delivery_cost",
    data=df,
    color="purple",
    s=70,
    ax=ax
)
st.pyplot(fig)


# Region Distribution (Line Chart)
st.subheader("Region Distribution")
fig, ax = plt.subplots(figsize=(7,4))
df["region"].value_counts().sort_index().plot(kind="line", marker="o",color="orange",linewidth=3,ax=ax)
st.pyplot(fig)


# Delivery Partner (Bar Chart)
st.subheader("Delivery Partner")
fig, ax = plt.subplots(figsize=(7,4))
df["delivery_partner"].value_counts().plot(kind="bar",color="teal",edgecolor="black",ax=ax)
st.pyplot(fig)

# Vehicle Type (Horizontal Bar Chart)
st.subheader("Vehicle Type")
fig, ax = plt.subplots(figsize=(7,4))
df["vehicle_type"].value_counts().plot(kind="barh", color="brown",ax=ax)
st.pyplot(fig)


# Delivery Status (Pie Chart)
st.subheader("Delivery Status Pie Chart")
fig, ax = plt.subplots(figsize=(6,6))
df["delivery_status"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax
)
ax.set_ylabel("")
st.pyplot(fig)


# Correlation Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)


# Delivery Cost Distribution (Violin Plot)
st.subheader("Delivery Cost Distribution")
fig, ax = plt.subplots(figsize=(6,4))
sns.violinplot(y=df["delivery_cost"], color="violet",linewidth=2,ax=ax)
st.pyplot(fig)

# Sidebar
st.sidebar.title("Dashboard Filters")
region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + list(df["region"].unique())
)
if region != "All":
    df = df[df["region"] == region]

st.subheader("Dashboard Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Shipments", len(df))
col2.metric("Average Cost", round(df["delivery_cost"].mean(), 2))
col3.metric("Average Distance", round(df["distance_km"].mean(), 2)) 

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Copy dataframe
ml_df = df.copy()

# Drop ID column
if "delivery_id" in ml_df.columns:
    ml_df.drop("delivery_id", axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in ml_df.select_dtypes(include="object").columns:
    ml_df[col] = le.fit_transform(ml_df[col])

st.write(ml_df.columns.tolist())

st.subheader("ML DataFrame Columns")
st.write(ml_df.columns.tolist())
# Target column
X = ml_df.drop("delayed", axis=1)
y = ml_df["delayed"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# ROC AUC CURVE
from sklearn.metrics import roc_auc_score , roc_curve
st.subheader("ROC AUC Curve")
y_prob = lr.predict_proba(X_test)
auc = roc_auc_score(y_test, y_prob[:,1])
st.success(f"ROC AUC Score : {auc:.3f}")
fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
ax.plot([0,1], [0,1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.setTitle = "ROC Curve"
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)


# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score

st.subheader("Model Performance")
lr_accuracy = accuracy_score(y_test, y_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
st.write("### Logistic Regression Accuracy")
st.success(f"{lr_accuracy*100:.2f}%")
st.write("### Random Forest Accuracy")
st.success(f"{rf_accuracy*100:.2f}%")

#comnfusion matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, rf_pred)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    linewidths=1,
    linecolor="black",
    cbar=True,
    ax=ax
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

st.pyplot(fig)


#important features
st.subheader("Important Features")
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})
importance = importance.sort_values(by="Importance", ascending=False)
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(
    data=importance.head(10),
    x="Importance",
    y="Feature",
    palette="viridis",
    edgecolor="black",
    ax=ax
)
ax.set_title("Important Features")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
st.pyplot(fig)


st.markdown("---")
st.markdown("### DHL Shipment Analytics using EDA with Streamlit Dashboard")
st.write("Developed using Python, Pandas, Seaborn, Matplotlib, Scikit-learn and Streamlit.")


st.subheader("Download Dataset")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="DHL_Shipment_Data.csv",
    mime="text/csv"
)



st.subheader("View Full Dataset")
if st.checkbox("Show Complete Dataset"):
    st.dataframe(df)


st.sidebar.markdown("---")
st.sidebar.header("Project Details")
st.sidebar.write("Project:")
st.sidebar.write("DHL Shipment Analytics using EDA with Streamlit Dashboard")
st.sidebar.write("Tools Used:")
st.sidebar.write("- Python")
st.sidebar.write("- Pandas")
st.sidebar.write("- Matplotlib")
st.sidebar.write("- Seaborn")
st.sidebar.write("- Streamlit")
st.sidebar.write("- Scikit-learn")



# PROJECT COMPLETED MESSAGE
st.success("DHL Shipment Analytics Dashboard Loaded Successfully!")

# THANK YOU MESSAGE
st.markdown("---")
st.write("Thank you for using the DHL Shipment Analytics Dashboard.")