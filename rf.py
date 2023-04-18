import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('finaldata.csv')


X = df.drop(['date', 'energy_kWh'], axis=1)
y = df['energy_kWh after normalisation']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


score = rf.score(X_test, y_test)
print("R-squared score on the testing data with random forest:", score)
