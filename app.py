
import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('xgboost_model.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

st.title("Prediksi Status Booking Hotel")
st.write("Masukkan detail pemesanan dan klik **Prediksi** untuk mengetahui apakah booking akan dibatalkan.")

no_of_adults = st.number_input("Jumlah Dewasa", min_value=0, value=1)
no_of_children = st.number_input("Jumlah Anak", min_value=0, value=0)
no_of_weekend_nights = st.number_input("Malam Akhir Pekan", min_value=0, value=0)
no_of_week_nights = st.number_input("Malam Hari Kerja", min_value=0, value=1)
type_of_meal_plan = st.selectbox("Tipe Paket Makan", ["Not Selected", "Meal Plan 1", "Meal Plan 2", "Meal Plan 3"])
required_car_parking_space = st.selectbox("Memerlukan Tempat Parkir?", [0, 1])
room_type_reserved = st.selectbox("Tipe Kamar", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
lead_time = st.number_input("Lead Time (hari)", min_value=0, value=0)
arrival_year = st.selectbox("Tahun Kedatangan", [2017, 2018])
arrival_month = st.number_input("Bulan Kedatangan", min_value=1, max_value=12, value=1)
arrival_date = st.number_input("Tanggal Kedatangan", min_value=1, max_value=31, value=1)
market_segment_type = st.selectbox("Segmen Pasar", ["Offline", "Online", "Corporate", "Aviation", "Complementary"])
repeated_guest = st.selectbox("Tamu Berulang?", [0, 1])
no_of_previous_cancellations = st.number_input("Jumlah Pembatalan Sebelumnya", min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Jumlah Booking Tidak Dibatalkan", min_value=0, value=0)
avg_price_per_room = st.number_input("Harga Rata-rata per Kamar", min_value=0.0, value=100.0)
no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0, value=0)

if st.button("Prediksi"):
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
    result = model.predict(df_input)
    status = "Dibatalkan" if result[0] == 1 else "Tidak Dibatalkan"
    st.subheader(f"Hasil Prediksi: **{status}**")
