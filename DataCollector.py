#Horse race predictor AI



# Dataset try just using dooban race results first
# and see how it goes might need to do indervidual race horses 



import requests
import xgboost as xgb
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from bs4 import BeautifulSoup

BASE_URL = "https://www.racingaustralia.horse/"

class DataCollector:
# [[{}, {}, ...], [{}, {}, ...], [{}, {}, ...], ...]
    def __init__(self):
        test = self.get_next_day_track_urls()
        self.data = []
        for url in test:
            race_stats = self.get_form_data(url)
            self.data.append(race_stats)



    def get_next_day_track_urls(self):
        response = requests.get(BASE_URL)
        soup = BeautifulSoup(response.text, "html.parser")

        track_urls = []
        for links in soup.find_all("a", href=True):
            if "FreeFields/Form.aspx" in links["href"]:  # Filter links related to race tracks
                track_urls.append(BASE_URL + links["href"])
        print(track_urls)
        return track_urls


    def get_future_race_data(self):
        #gets the data of up and coming races, including which horses are racing in it
        race_data = []
        
        return race_data


    def get_form_data(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        
        # All form data 
        race_data = []
        for table in soup.find_all("table", class_="horse-form-table"):
            perious = table.find_all("table", class_= "horse-last-start")
            
            if len(perious) == 0:
                continue
            perious_list = table.find_all("span", class_="horse-name")
            horse_name = None
            # Get the first element if it exists
            if perious_list:
                horse_name = perious_list[0].find("a").text
            for row in perious[0].find_all("tr"):
                cols = row.find_all("td")
                if len(cols) > 1:
                    win = 0
                    position = cols[0].text.strip()  # Extract race position
                    
                    race_details = cols[1]
                    details_text = race_details.text
                    if position.split("\xa0")[0] == "1":
                        win = 1
                    parts = details_text.split()
                    distance = parts[2]
                    track_condition = parts[3]
                    race_class = parts[4]
                    jockey_link = race_details.find_all("a", class_="GreenLink")
                    jockey_name = jockey_link[1].text.strip()
                    weight = 0
                    barrier = None
                    num = 0
                    for item in parts:
                        
                        if item.endswith("kg"):
                            weight = item 
                            
                        elif item == "Barrier":
                            barrier = parts[num+1]
                            if barrier.endswith("st") or barrier.endswith("nd") or barrier.endswith("rd"):
                                barrier = barrier[:-3]
                        num += 1

                    if weight == "0kg":
                        continue


                    race_data_ind = {
                        "horse": horse_name,
                        "position": position,
                        "jockey_name": jockey_name,
                        "distance": distance,
                        "track_condition": track_condition,
                        "race_class": race_class,
                        "weight": weight,
                        "barrier": barrier,
                        "win": win
                        # "raceName": raceName,
                        # "age": horseAge
                    }
                    race_data.append(race_data_ind)




        return race_data
    
    def processed_data(self):
        sameList = [entry for race in self.get_data() for entry in race]

        df = pd.DataFrame(sameList)
        
        # 2️⃣ Extract Numeric Position
        def extract_position(pos):
            match = re.match(r"(\d+)\xa0of\xa0(\d+)", pos)
            return int(match.group(1)) if match else None

        df["position"] = df["position"].apply(extract_position)

        # 3️⃣ Convert Numeric Features
        def extract_numeric(value):
            return float(re.sub(r"[^\d.]", "", value)) if value else None

        df["distance"] = df["distance"].apply(extract_numeric)
        df["weight"] = df["weight"].apply(extract_numeric)
        df = df.dropna(subset=["weight"])
        df = df[pd.to_numeric(df["barrier"], errors="coerce").notna()]
        df["barrier"] = df["barrier"].astype(int)  # Now safely convert to int
        print(df)
        # 4️⃣ Encode Categorical Features
        for col in ["horse", "jockey_name", "track_condition", "race_class"]:
            df[col] = LabelEncoder().fit_transform(df[col])

        # 5️⃣ Define Features & Target
        X = df.drop(columns=["position"])
        y = df["position"]
        

        valid_idx = X.index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        scaler = MinMaxScaler()

        # List of numerical columns to normalize
        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

        # Apply the scaler to the numerical columns
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
        
        return [X, y]
    
    def get_data(self):
        return self.data
    
    
test = DataCollector()