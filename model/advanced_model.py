import argparse
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pickle
import datetime
from calorie_calculator import calculate_calories_burned


# Argumente verarbeiten
parser = argparse.ArgumentParser(description='Create Model')
parser.add_argument('-u', '--uri', required=True, help="mongodb uri with username/password")
args = parser.parse_args()

# MongoDB Verbindung herstellen
client = MongoClient(args.uri)
db = client["tracks"]
collection = db["tracks"]

# Daten abfragen
values = [track for track in collection.find(projection={"gpx": 0, "url": 0, "bounds": 0, "name": 0})]
df = pd.DataFrame(values).set_index("_id")

# Daten vorbereiten
df['avg_speed'] = df['length_3d'] / df['moving_time']
df['difficulty_num'] = df['difficulty'].apply(lambda x: int(x[1])).astype('int32')
df.dropna(inplace=True)
df = df[(df['avg_speed'] < 2) & (df['min_elevation'] > 0) & (df['length_2d'] < 100000)]

# o Berechne Kalorienverbrauch für jede Route und füge diesen als neue Spalte hinzu
df['calories_burned'] = df.apply(lambda row: calculate_calories_burned(row['uphill'], row['downhill'], row['length_3d'], row['moving_time']), axis=1)

# Korrelationen analysieren
corr = df.corr(numeric_only=True)
print(corr)
sn.heatmap(corr, annot=True)
plt.show()

# Daten für das Training vorbereiten
y = df['moving_time']
X = df[['downhill', 'uphill', 'length_3d', 'max_elevation', 'calories_burned']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Lineares Regressionsmodell
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Linear Regression - r2: {}\nMSE: {}".format(r2_score(y_test, y_pred_lr), mean_squared_error(y_test, y_pred_lr)))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=50, random_state=9000)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
print("Gradient Boosting Regressor - r2: {}\nMSE: {}".format(r2_score(y_test, y_pred_gbr), mean_squared_error(y_test, y_pred_gbr)))

# Modell speichern
with open('GradientBoostingRegressor.pkl', 'wb') as fid:
    pickle.dump(gbr, fid)

# Modell laden und eine Demo-Vorhersage durchführen
with open('GradientBoostingRegressor.pkl', 'rb') as fid:
    gbr_loaded = pickle.load(fid)
    demo_output = gbr_loaded.predict([[300, 700, 10000, 1200]])[0]
    print("Our Model: " + str(datetime.timedelta(seconds=demo_output)))
