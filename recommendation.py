import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import pymongo
from pymongo import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://ilhanmahardikap:xxxxxxxx@cluster0.6phgimn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))

db = client.movie_recommender
df = pd.DataFrame(list(db.combined_features.find()))
df = df.sort_values(by='index')
df.drop_duplicates(subset=['title'], inplace=True)
df.reset_index(drop=True, inplace=True)
df['index'] = df.index

# create count matrix
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(df['combined_features'])

# compute cosine similarity
cos_sim_val = cosine_similarity(matrix)

# get index from title


def get_index_from_title(title):
    indices = df[df['title'] == title].index.values
    if len(indices) > 0:
        return indices[0]
    else:
        return None

# get title from index


def get_title_from_index(index):
    return df[df['index'] == index].title.values[0]

def get_id_from_index(index):
    return df[df['index'] == index].id.values[0]

def get_poster_from_index(index):
    return df[df['index'] == index].poster_path.values[0]


def get_similar_movies(movie_index):
    similar_score = list(enumerate(cos_sim_val[movie_index]))
    # similar_score = similar_score.sort_values(ascending=False)
    sorted_similar_movies = sorted(
        similar_score, key=lambda x: x[1], reverse=True)
    return sorted_similar_movies


# def recommend(movie_user_likes):
#     similar_movies = list()
#     for movie in movie_user_likes:
#         movie_index = get_index_from_title(movie)
#         similar_movies += get_similar_movies(movie_index)
#         sorted_similar_movies = sorted(
#             similar_movies, key=lambda x: x[1], reverse=True)
#     # Print title of first 50 movies
#     i = 0
#     for movie in sorted_similar_movies:
#         if i > 50:
#             break
#         print(get_title_from_index(movie[0]))
#         i = i+1

def recommend(movie_user_likes):
    # movie_user_likes_array = movie_user_likes.split(",")
    similar_movies = list()
    for movie in movie_user_likes:
        movie_index = get_index_from_title(movie)
        if movie_index is not None:
            similar_movies += get_similar_movies(movie_index)
        sorted_similar_movies = sorted(
            similar_movies, key=lambda x: x[1], reverse=True)
    # Print title of first 50 movies

    recommended_movies = []

    i = 0
    for movie in sorted_similar_movies:
        if i > 50:
            break
        recommended_movies.append((str(get_id_from_index(movie[0])), get_title_from_index(movie[0]), movie[1], get_poster_from_index(movie[0])))
        # recommended_movies.append(movie)
        i = i+1

    return recommended_movies


# movie_user_likes = ["Insidious"]

# array = recommend(movie_user_likes)
# print(array)
