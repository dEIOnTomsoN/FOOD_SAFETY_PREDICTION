
import pickle
import pandas as pd
import streamlit as st

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load Scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load Dataset for Category Encoding
df = pd.read_csv("food_safety.csv")

product_type_mapping = {
    value: index
    for index, value in enumerate(sorted(df["product_type"].unique()))
}

packaging_type_mapping = {
    value: index
    for index, value in enumerate(sorted(df["packaging_type"].unique()))
}


def preprocess_and_predict(features):
    input_data = pd.DataFrame([features])

    required_columns = [
        "product_type",
        "packaging_type",
        "shelf_life_days",
        "storage_temperature_C",
        "humidity_level_percent",
        "moisture_content",
        "preservative_level",
        "pH_level",
        "inspection_score_0_100",
        "customer_rating_1_5"
    ]

    input_data = input_data[required_columns]

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[:, 1]

    return prediction[0], probability[0]


# Streamlit App

st.set_page_config(
    page_title="Food Safety Prediction",
    page_icon="🍱",
    layout="wide"
)

st.title("Food Safety Approval Prediction")

st.write(
    "Enter the product details below to predict its approval status."
)

st.divider()

# Row 1 - Four Columns

col1, col2, col3, col4 ,col5= st.columns(5)

with col1:
    product_type = st.selectbox(
        "Product Type",
        list(product_type_mapping.keys())
    )

with col2:
    packaging_type = st.selectbox(
        "Packaging Type",
        list(packaging_type_mapping.keys())
    )

with col3:
    shelf_life_days = st.number_input(
        "Shelf Life (Days)",
        min_value=1,
        value=30
    )

with col4:
    storage_temperature_C = st.number_input(
        "Storage Temperature (°C)",
        value=5.0
    )

with col5:
    inspection_score_0_100 = st.number_input(
        "Inspection Score (0-100)",
        min_value=0,
        max_value=100,
        value=75
    )
# Row 2 - Four Columns

col1, col2, col3, col4,col5= st.columns(5)

with col1:
    humidity_level_percent = st.number_input(
        "Humidity Level (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

with col2:
    moisture_content = st.number_input(
        "Moisture Content",
        min_value=0.0,
        value=0.5
    )

with col3:
    preservative_level = st.number_input(
        "Preservative Level",
        min_value=0.0,
        value=1.0
    )

with col4:
    pH_level = st.number_input(
        "pH Level",
        min_value=0.0,
        max_value=14.0,
        value=7.0
    )

with col5:
    customer_rating_1_5 = st.number_input(
        "Customer Rating (1-5)",
        min_value=1,
        max_value=5,
        value=4
    )






st.divider()

# Encoding and Features

features = {
    "product_type": product_type_mapping[product_type],
    "packaging_type": packaging_type_mapping[packaging_type],
    "shelf_life_days": shelf_life_days,
    "storage_temperature_C": storage_temperature_C,
    "humidity_level_percent": humidity_level_percent,
    "moisture_content": moisture_content,
    "preservative_level": preservative_level,
    "pH_level": pH_level,
    "inspection_score_0_100": inspection_score_0_100,
    "customer_rating_1_5": customer_rating_1_5
}


# Prediction

if st.button("Predict Approval", type="primary"):

    prediction, probability = preprocess_and_predict(features)

    if prediction == 1:
        st.success("Predicted Status: APPROVED")
        st.write(f"Model probability of approval: {probability:.2%}")
    else:
        st.error("Predicted Status: REJECTED")
        st.write(f"Model probability of approval: {probability:.2%}")

   

# Custom Color Theme

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F5F0FF 0%, #E8DEFF 100%);
}

h1 {
    color: #24134D;
    font-weight: 800;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    border: 1px solid #D8C8FF;
    border-radius: 12px;
}

div[data-baseweb="input"] > div {
    background-color: #FFFFFF;
    border: 1px solid #D8C8FF;
    border-radius: 12px;
}

.stButton > button {
    background: linear-gradient(90deg, #8B5CF6, #6D28D9);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: #5B21B6;
    color: white;
    border: none;
}

div[data-testid="stVerticalBlock"] {
    gap: 1rem;
}
</style>
""", unsafe_allow_html=True)
