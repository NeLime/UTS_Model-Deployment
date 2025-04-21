import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('xgboost_model.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

st.title("Hotel Booking Status Prediction")
st.write("Fill in the booking details below and click **Predict** to see whether the booking is likely to be canceled.")

# Input fields
no_of_adults = st.number_input("Number of Adults", min_value=0, value=1)
no_of_children = st.number_input("Number of Children", min_value=0, value=0)
no_of_weekend_nights = st.number_input("Weekend Nights", min_value=0, value=0)
no_of_week_nights = st.number_input("Week Nights", min_value=0, value=1)
type_of_meal_plan = st.selectbox("Meal Plan", ["Not Selected", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"])
required_car_parking_space = st.selectbox("Requires How Many Parking Space?", [0, 1])
room_type_reserved = st.selectbox("Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
lead_time = st.number_input("Lead Time (days)", min_value=0, value=0)
arrival_year = st.number_input("Arrival Year", min_value=1, max_value=99999)
arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=1)
arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=1)
market_segment_type = st.selectbox("Market Segment", ["Offline", "Online", "Corporate", "Aviation", "Complementary"])
repeated_guest = st.selectbox("Repeated Guest?", [0, 1])
no_of_previous_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Previous Non-Canceled Bookings", min_value=0, value=0)
avg_price_per_room = st.number_input("Average Price per Room", min_value=0.0, value=100.0)
no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, value=0)

if st.button("Predict"):
    input_data = {
        "no_of_adults": no_of_adults,
        "no_of_children": no_of_children,
        "no_of_weekend_nights": no_of_weekend_nights,
        "no_of_week_nights": no_of_week_nights,
        "type_of_meal_plan": type_of_meal_plan,
        "required_car_parking_space": required_car_parking_space,
        "room_type_reserved": room_type_reserved,
        "lead_time": lead_time,
        "arrival_year": arrival_year,
        "arrival_month": arrival_month,
        "arrival_date": arrival_date,
        "market_segment_type": market_segment_type,
        "repeated_guest": repeated_guest,
        "no_of_previous_cancellations": no_of_previous_cancellations,
        "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
        "avg_price_per_room": avg_price_per_room,
        "no_of_special_requests": no_of_special_requests
    }

    df_input = pd.DataFrame([input_data])
    df_input['type_of_meal_plan'].fillna('Not Selected', inplace=True)
    df_input['required_car_parking_space'].fillna(0, inplace=True)
    df_input['avg_price_per_room'].fillna(df_input['avg_price_per_room'].median(), inplace=True)
    df_input = pd.get_dummies(df_input, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], drop_first=True)
    df_input = df_input.reindex(columns=features, fill_value=0)

    probabilities = model.predict_proba(df_input)[0]
    prediction = "Canceled" if probabilities[1] >= 0.5 else "Not Canceled"
    confidence = probabilities[1] if prediction == "Canceled" else probabilities[0]

    st.subheader(f"Prediction: **{prediction}**")
    st.write(f"Probability: **{confidence * 100:.2f}%**")

