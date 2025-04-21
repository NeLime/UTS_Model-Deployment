
import pandas as pd
import pickle

class HotelBookingModelPredictor:
    def __init__(self, model_path, features_path=None):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
        if features_path:
            with open(features_path, 'rb') as file:
                self.features = pickle.load(file)
        else:
            self.features = None

    def predict(self, input_data):
        df_input = pd.DataFrame([input_data])
        df_input['type_of_meal_plan'].fillna('Not Selected', inplace=True)
        df_input['required_car_parking_space'].fillna(0, inplace=True)
        df_input['avg_price_per_room'].fillna(df_input['avg_price_per_room'].median(), inplace=True)
        df_input = pd.get_dummies(df_input, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], drop_first=True)
        if self.features:
            df_input = df_input.reindex(columns=self.features, fill_value=0)
        result = self.model.predict(df_input)
        return "Canceled" if result[0] == 1 else "Not_Canceled"

if __name__ == "__main__":
    sample_input = {
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 224,
        "arrival_year": 2017,
        "arrival_month": 10,
        "arrival_date": 2,
        "market_segment_type": "Offline",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 65.0,
        "no_of_special_requests": 0
    }
    predictor = HotelBookingModelPredictor("xgboost_model.pkl", "features.pkl")
    prediction = predictor.predict(sample_input)
    print("Prediction:", prediction)
