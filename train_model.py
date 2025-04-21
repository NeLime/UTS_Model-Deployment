
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

class HotelBookingModelTrainer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        self.features = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        return df

    def preprocess_data(self, df):
        data = df.copy()
        if 'Booking_ID' in data.columns:
            data.drop('Booking_ID', axis=1, inplace=True)
        data['type_of_meal_plan'].fillna('Not Selected', inplace=True)
        data['required_car_parking_space'].fillna(0, inplace=True)
        data['avg_price_per_room'].fillna(data['avg_price_per_room'].median(), inplace=True)
        data['required_car_parking_space'] = data['required_car_parking_space'].astype(int)
        data['booking_status'] = data['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)
        data = pd.get_dummies(data, columns=['type_of_meal_plan', 'room_type_reserved', 'market_segment_type'], drop_first=True)
        return data

    def split_data(self, data):
        X = data.drop('booking_status', axis=1)
        y = data['booking_status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.features = X_train.columns.tolist()
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_train, y_train, X_test, y_test):
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, target_names=["Not_Canceled", "Canceled"]))
        return acc_train, acc_test

    def save_model(self, model_path, features_path=None):
        with open(model_path, 'wb') as file:
            pickle.dump(self.model, file)
        if features_path:
            with open(features_path, 'wb') as file:
                pickle.dump(self.features, file)
        print(f"Model saved to {model_path}")
        if features_path:
            print(f"Features saved to {features_path}")

if __name__ == "__main__":
    trainer = HotelBookingModelTrainer("Dataset_B_hotel.csv")
    df = trainer.load_data()
    data = trainer.preprocess_data(df)
    X_train, X_test, y_train, y_test = trainer.split_data(data)
    trainer.train_model(X_train, y_train)
    trainer.evaluate_model(X_train, y_train, X_test, y_test)
    trainer.save_model("xgboost_model.pkl", "features.pkl")
