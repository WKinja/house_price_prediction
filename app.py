
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# Load the encoder and model
encoder = joblib.load("encoder.pkl")  
model = joblib.load("house_model.pkl")  

# Streamlit Interface
st.title("House Price Prediction")

# Updated location options
location_options = [
    "Riverside Dr Nairobi, Riverside, Nairobi",
    "Kileleshwa, Nairobi",
    "Links Rd Mombasa, Nyali, Mombasa",
    "Near Valley Arcade, Lavington, Nairobi",
    "Thika Rd Nairobi, Kahawa Wendani, Nairobi",
    "Kilimani, Nairobi",
    "Nyali, Mombasa",
    "Muthaiga, Nairobi",
    "Westlands, Nairobi",
    "Kikuyu Town Bus park Kikuyu, Kikuyu, Kikuyu",
    "Shanzu, Mombasa",
    "Westlands downtown, Westlands, Nairobi",
    "Kileleshwa Nairobi, Kileleshwa, Nairobi",
    "Grevillea Grove Spring Valley, Spring Valley, Nairobi",
    "Vihiga road, Kileleshwa, Nairobi",
    "Off Othaya road, Lavington, Nairobi",
    "Jabavu court, Kilimani, Nairobi"
]

# Input fields
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
location = st.selectbox("Location", location_options)

# Define the price range prediction function
def price_range_prediction(price):
    if price < 50000:
        return "Below Ksh 50,000"
    elif price < 100000:
        return "Ksh 50,000 - Ksh 100,000"
    elif price < 200000:
        return "Ksh 100,000 - Ksh 200,000"
    elif price < 300000:
        return "Ksh 200,000 - Ksh 300,000"
    else:
        return "Above Ksh 300,000"

# Preprocess user input
def preprocess_input(user_input, encoder, numerical_features, trained_columns):
    # Encode location
    location_encoded = encoder.transform([[user_input['location']]])
    location_columns = encoder.get_feature_names_out(['location'])

    # Create DataFrame for location encoding
    location_df = pd.DataFrame(location_encoded, columns=location_columns)

    # Combine numerical features
    input_data = pd.DataFrame({
        'bedrooms': [user_input['bedrooms']],
        'bathrooms': [user_input['bathrooms']]
    })

    # Combine numerical and categorical features
    final_input = pd.concat([input_data, location_df], axis=1)

    # Add missing columns with zeros
    for col in trained_columns:
        if col not in final_input.columns:
            final_input[col] = 0

    # Ensure the column order matches the trained model
    final_input = final_input[trained_columns]

    return final_input

# Predict button
if st.button("Predict Price"):
    user_input = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'location': location
    }
    
    # Load trained columns
    trained_columns = joblib.load("C:/Users/DELL 3350/Downloads/trained_columns.pkl")

    # Preprocess the input
    input_df = preprocess_input(user_input, encoder, ['bedrooms', 'bathrooms'], trained_columns)

    # Debugging: Print column alignment
    print(f"Expected features: {trained_columns}")
    print(f"Input features: {input_df.columns}")

    # Predict
    predicted_price = model.predict(input_df)
    price_category = price_range_prediction(predicted_price[0])
    st.success(f"Predicted Price: Ksh {predicted_price[0]:,.2f}")
    st.info(f"Predicted Price Range: {price_category}")
