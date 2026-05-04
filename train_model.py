import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('data/pharma_sales.csv')

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Encode categorical columns
encoders = {}

for col in ['Product', 'Region', 'Season']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features and target
X = df[['Product', 'Region', 'Price',
        'Marketing_Spend',
        'Inventory_Level',
        'Season',
        'Year',
        'Month',
        'Day']]

y = df['Units_Sold']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("R2 Score:", r2)

# Save model
joblib.dump(model, 'models/sales_model.pkl')

print("Model saved successfully!")