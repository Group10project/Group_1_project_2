import pandas as pd
import numpy as np
import requests
import joblib  
import os
from dotenv import load_dotenv
import time
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

load_dotenv()

class WaterShortagePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = [
            'dam_level', 
            'loadshedding_stage',
            'pollution_index',
            'water_usage',
            'precipitation',
            'temperature',
            'day_of_week',
            'month'
        ]
        self.target = 'shortage_occurred'
        self.fixed_water_usage = 2.8

    def fetch_with_retry(self, func, max_retries=3, delay=2):
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(delay)
        return None

    def fetch_dam_levels(self):
        """FIND OUT THE DAM LEVELS
        in the Dlangezwa area nearest"""
        def _fetch():
            url = "https://www.dws.gov.za/Hydrology/Weekly/ProvinceWeek.aspx?region=UM"
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            return float(soup.find_all('td')[10].text.strip('%'))
        
        try:
            return self.fetch_with_retry(_fetch) or 65.0
        except:
            return 65.0

    def fetch_loadshddingdata(self, target_date):
        """WHAT IS HAPPENING WITH LOAD-SHEDDING STAGES RIGHT NOW"""
        try:
            api_key = os.getenv("ESPUSH_API_KEY")
            
            if target_date.date() == datetime.now().date():
                url =   "https://developer.sepush.co.za/business/2.0/status"
            else:url = f"https://developer.sepush.co.za/business/2.0/history?date={target_date.strftime('%Y-%m-%d')}"
            
            response = requests.get(url, headers={"token": api_key}, timeout=5)
            return response.json()["eskom"]["stage"]
        except:
            return 0  # Fallback

    




    def fetch_weather_forecast(self, target_date):
        #WEATHER NJE
        try:
            is_past = target_date < datetime.now().date()
            base_url = ("https://archive-api.open-meteo.com/v1/archive" 
                    if is_past 
                    else "https://api.open-meteo.com/v1/forecast")
            
            params = {
                "latitude": -28.7167,
                "longitude": 31.9000,
                "start_date": target_date.strftime("%Y-%m-%d"),
                "end_date": target_date.strftime("%Y-%m-%d"),
                "daily": ["temperature_2m_max", "precipitation_sum"],
                "timezone": "Africa/Johannesburg"
            }
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            return {
                "temperature": data["daily"]["temperature_2m_max"][0],
                "precipitation": data["daily"]["precipitation_sum"][0]
            }
        except:
            return {"temperature": 25, "precipitation": 0}
    
    def fetch_pollution_data(self):
        #Fetch pollution data
        try:
            return np.random.randint(1, 5)
        except:
            return 3
        
    def get_water_usage(self, date):
        """Simplified version using fixed value"""
        return self.fixed_water_usage

    def load_data(self, filepath):
        dataset = pd.read_csv(filepath)
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset['day_of_week'] = dataset['date'].dt.dayofweek
        dataset['month'] = dataset['date'].dt.month
        dataset.fillna({
            'dam_level': dataset['dam_level'].median(),
            'loadshedding_stage': 0,
            'pollution_index': 3,
            'water_usage': dataset['water_usage'].median(),
            'precipitation': 0,
            'temperature': dataset['temperature'].median()
        }, inplace=True)
        return dataset



    def train(self, data_path):
        dataset = self.load_data(data_path)
        X = dataset[self.features]
        Y = dataset[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        self.model.fit(X_train, y_train)
        print(f"Model accuracy: {self.model.score(X_test, y_test):.2f}")
        joblib.dump(self.model, "water_model.joblib")





    def predict_shortage(self, input_data):
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:input_df = input_data.copy()
        
        for feat in self.features:
            if feat not in input_df.columns:
                input_df[feat] = 0
        
        return self.model.predict_proba(input_df[self.features])[0][1]

def generatertheone():
    dates = pd.date_range(start="2022-01-01", end="2023-12-31")
    data = {
        "date": dates,
        "dam_level": np.random.uniform(30,100,  len(dates)),
        "loadshedding_stage": np.random.randint(0,6, len(dates)),
        "pollution_index": np.random.randint(1, 6, len(dates)),
        "water_usage": np.random.uniform(1.5, 5.0, len(dates)),
        "precipitation": np.random.uniform(0,20, len(dates)),
        "temperature": np.random.uniform(15, 35,len(dates)),
        "shortage_occurred": np.random.choice([0,1], len(dates), p=[0.7,0.3])
    }
    pd.DataFrame(data).to_csv("historical_water_data.csv", index=False)

if __name__ == "__main__":
    if not os.path.exists("historical_water_data.csv"):
        generatertheone()
    
    predictor = WaterShortagePredictor()
    predictor.train("historical_water_data.csv")
