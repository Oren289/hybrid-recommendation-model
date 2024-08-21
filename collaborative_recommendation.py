import pandas as pd
import numpy as np
import ast
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
from pymongo import MongoClient
from pymongo.server_api import ServerApi


uri = "mongodb+srv://ilhanmahardikap:xxxxxxxx@cluster0.6phgimn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

db = client.movie_recommender
ratings = pd.DataFrame(list(db.ratings.find()))
ratings.rename(columns={'movieId': 'id'}, inplace=True)
movies = pd.read_csv('./processed_movies.csv', low_memory=False)
movies = movies[movies['id'].apply(str).str.contains('-') == False]
movies['id'] = movies['id'].astype('int')
ratings = pd.merge(movies, ratings)
ratings = ratings[['id', 'title', 'userId', 'rating', 'poster_path']]

# pivoting the table
user_ratings = ratings.pivot_table(index=['userId'], columns=[
                                   'title'], values='rating')
user_ratings = user_ratings.dropna(thresh=5, axis=1).fillna(0)

# pearson correlation
item_similarity_df = user_ratings.corr(method='pearson')


def get_similar_movies(movie_name, user_rating):
    similar_score = item_similarity_df[movie_name]*(user_rating-2.5)
    similar_score = similar_score.sort_values(ascending=False)
    return similar_score

def get_id_from_title(title):
    return ratings[ratings['title'] == title].id.values[0]

def get_poster_from_title(title):
    return ratings[ratings['title'] == title].poster_path.values[0]


def recommend_collaborative(movies):
    similar_movies = pd.DataFrame()
    response_series = pd.Series(movies)
    response_list = response_series.tolist()
    converted_movies = [ast.literal_eval(item) for item in response_list]

    for movie, rating in converted_movies:
        if movie in user_ratings:
            similar_movies = similar_movies._append(
                get_similar_movies(movie, rating), ignore_index=True)

    similar_movies = similar_movies.mean().sort_values(ascending=False).head(50)

    similar_movies_array = []

    for title, value in similar_movies.items():
        similar_movies_array.append((get_id_from_title(title), title, value, get_poster_from_title(title)))

    return similar_movies_array
