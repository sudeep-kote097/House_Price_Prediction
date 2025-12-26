import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        with open('house_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please retrain the model using: `python model.py`")
        st.stop()

model, scaler, feature_names = load_model()

# Title and description
st.title("🏠 House Price Prediction App")
st.write("Predict house prices based on property details")

# Sidebar with model info
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write(f"**scikit-learn version:** {sklearn.__version__}")
    st.write(f"**Features used:** {len(feature_names)}")
    st.write(f"**Model type:** Random Forest Regressor")

# Create input form
st.subheader("Enter Property Details")

# Example: Common house features (adjust based on your dataset)
col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    sqft_living = st.number_input("Living Area (sqft)", min_value=500, max_value=10000, value=2000)

with col2:
    sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, max_value=50000, value=5000)
    floors = st.number_input("Floors", min_value=1, max_value=4, value=1)
    waterfront = st.selectbox("Waterfront", ["No", "Yes"])

with col3:
    condition = st.slider("Condition (1-5)", min_value=1, max_value=5, value=3)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2024, value=2000)
    yr_renovated = st.number_input("Year Renovated", min_value=0, max_value=2024, value=0)

# Predict button
if st.button("💰 Predict Price", use_container_width=True):
    try:
        # Prepare input data (adjust based on your actual features)
        input_dict = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'floors': floors,
            'waterfront': 1 if waterfront == "Yes" else 0,
            'condition': condition,
            'yr_built': yr_built,
            'yr_renovated': yr_renovated
        }
        
        # Create DataFrame with all features (add missing features with default values)
        input_data = pd.DataFrame([input_dict])
        
        # Add any missing features with 0
        for feature in feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Reorder columns to match training data
        input_data = input_data[feature_names]
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display results
        st.success(f"### 🏡 Predicted House Price: ${prediction:,.2f}")
        
        # Additional information
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated Price", f"${prediction:,.0f}")
        with col2:
            st.metric("Price per sqft", f"${prediction/sqft_living:.2f}")
        with col3:
            confidence = "High" if abs(prediction - 500000) < 200000 else "Medium"
            st.metric("Confidence", confidence)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Please ensure all features match the training data")

# Add information section
with st.expander("ℹ️ About this Application"):
    st.write("""
    This application uses a Random Forest Regressor to predict house prices.
    
    **Model Performance:**
    - Algorithm: Random Forest
    - Evaluation Metrics: R², MAE, RMSE
    
    **Note:** Predictions are based on historical data and should be used as estimates only.
    """)
