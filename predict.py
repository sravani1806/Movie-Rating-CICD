
import joblib
import pandas as pd

# Load trained model
model = joblib.load("movie_rating_model.pkl")

def predict_rating(genre, duration, budget, popularity):

    movie = pd.DataFrame([{
        "Genre": genre,
        "Duration_Min": duration,
        "Budget_Crore_INR": budget,
        "Popularity_Index": popularity
    }])

    rating = model.predict(movie)[0]

    return round(float(rating), 2)


if __name__ == "__main__":

    result = predict_rating(
        "Action", 150, 100, 8.5
    )

    print("Predicted Movie Rating:", result)
