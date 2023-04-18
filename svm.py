import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score

data = pd.read_csv('finaldata.csv')

X_train, X_test, y_train, y_test = train_test_split(data.drop(['energy_kWh', 'date'], axis=1), data['energy_kWh'], test_size=0.3, random_state=42)

svm_model = SVR(kernel='linear', C=1.0, epsilon=0.1)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("R-squared score of SVR: ", r2)
