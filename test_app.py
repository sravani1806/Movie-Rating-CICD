
from predict import predict_rating

def test_movie_prediction():

    rating = predict_rating(
        "Action", 150, 100, 8.5
    )

    assert isinstance(rating, float)
    assert 0 <= rating <= 10

def test_different_genre():

    rating = predict_rating(
        "Comedy", 120, 50, 7.5
    )

    assert isinstance(rating, float)
    assert 0 <= rating <= 10
