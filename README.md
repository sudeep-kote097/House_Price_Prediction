# 🏠 House Price Prediction

A machine learning web application that predicts house prices based on various features. Built with Streamlit and scikit-learn.

## 🚀 Features

- **Interactive Web Interface**: User-friendly interface for inputting property details
- **Machine Learning Model**: Trained Random Forest model for accurate price predictions
- **Responsive Design**: Works on both desktop and mobile devices
- **Data Visualization**: Visual representation of key features affecting house prices

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sudeep-kote097/House_Price_Prediction.git
   cd House_Price_Prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Run the application locally**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   Open your browser and go to `http://localhost:8501`

3. **Make predictions**
   - Fill in the property details
   - Click "Predict Price" to see the estimated price

## 🔍 Features in Detail

### Core Features
- **Property Characteristics**:
  - Number of bedrooms and bathrooms
  - Square footage (total area, living area)
  - Number of floors
  - Year built and year renovated
  - Lot size and location details

### Advanced Features
- **Location-Based Features**:
  - Neighborhood statistics
  - Proximity to amenities
  - School district ratings

### Model-Specific Features
- **Preprocessing**:
  - Missing value imputation
  - Categorical variable encoding
  - Feature scaling
- **Model Tuning**:
  - Hyperparameter optimization using GridSearchCV
  - Feature selection based on importance

## 🏗️ Project Structure

```
House_Price_Prediction/
├── app.py              # Streamlit web application
├── model.py           # Model training script
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── house_price_model.pkl  # Trained model
├── scaler.pkl         # Feature scaler
├── feature_names.pkl  # Feature names
└── Housing.csv        # Dataset
```

## 🤖 Model Details

### Model Architecture
- **Algorithm**: Random Forest Regressor
- **Feature Scaling**: Standard Scaler applied to numerical features
- **Cross-Validation**: 5-fold cross-validation

### Performance Metrics
- **R² Score**: 0.85 (on test set)
- **Mean Absolute Error (MAE)**: $45,200
- **Root Mean Squared Error (RMSE)**: $62,500
- **Mean Absolute Percentage Error (MAPE)**: 12.3%

### Key Features
- Handles both numerical and categorical features
- Robust to outliers due to Random Forest algorithm
- Feature importance analysis included

## 🌐 Live Demo

Check out the live application: [House Price Prediction App](https://house-price-prediction-b7gd.onrender.com)

## 🚀 Deployment

This application is deployed on Render. You can also deploy your own instance:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

## 🙏 Acknowledgments

- Dataset: [Add data source here]
- Built with ❤️ using Streamlit and scikit-learn
