
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading movie dataset...")

# 1. Load dataset
df = pd.read_csv("data/movies.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select input features
X = df[
    ["Genre", "Duration_Min",
     "Budget_Crore_INR", "Popularity_Index"]
]

# 3. Select target
y = df["Rating_10"]

# 4. Define columns
categorical = ["Genre"]

numerical = [
    "Duration_Min",
    "Budget_Crore_INR",
    "Popularity_Index"
]

# 5. Handle missing numerical values
num_process = Pipeline([
    ("imputer", SimpleImputer(strategy="mean"))
])

# 6. Handle categorical values
cat_process = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", num_process, numerical),
    ("cat", cat_process, categorical)
])

# 8. Create Random Forest model
model = Pipeline([
    ("preprocessing", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# 9. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training model...")

# 10. Train model
model.fit(X_train, y_train)

print("Model training completed!")

# 11. Evaluate model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", round(mae, 3))
print("R2 Score:", round(r2, 3))

# 12. Save model
joblib.dump(model, "movie_rating_model.pkl")

print("Model saved successfully!")
print("Training process completed!")
