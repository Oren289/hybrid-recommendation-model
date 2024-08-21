import pandas as pd
import ast
import recommendation
import collaborative_recommendation


def recommend_hybrid(movies):
    title__array = []
    response_series = pd.Series(movies)
    response_list = response_series.tolist()
    converted_movies = [ast.literal_eval(item) for item in response_list]

    for title, rating in converted_movies:
        title__array.append(title)

   # content filtering score
    content_filtering_score = recommendation.recommend(title__array)

   # collaborative filtering score
    collaborative_filtering_score = collaborative_recommendation.recommend_collaborative(
        movies)

   #  hybbrid code

    # Normalization function
    def normalize(scores):
        if not scores:
            normalized_scores = [
                [item[0], item[1], 0, item[3]] for item in scores
            ]
            return normalized_scores
        min_score = min(score[2] for score in scores)
        max_score = max(score[2] for score in scores)
        normalized_scores = [
            [item[0], item[1], (item[2] - min_score) / (max_score - min_score), item[3]] for item in scores
        ]
        return normalized_scores

    def combine_scores(cosine_scores, pearson_scores, weight_cosine=0.5, weight_pearson=0.5):
        # Normalize scores
        normalized_cosine_scores = normalize(cosine_scores)
        normalized_pearson_scores = normalize(pearson_scores)

        # Convert lists to dictionaries for efficient lookups
        cosine_dict = {item[1]: item for item in normalized_cosine_scores}
        pearson_dict = {item[1]: item for item in normalized_pearson_scores}

        # Combine scores using weighted sum
        combined_scores = {}
        for title, (id, _, cosine_score, poster_path) in cosine_dict.items():
            if title not in combined_scores:
                combined_scores[title] = [id, title, 0, poster_path]
            combined_scores[title][2] += weight_cosine * cosine_score

        for title, (id, _, pearson_score, poster_path) in pearson_dict.items():
            if title not in combined_scores:
                combined_scores[title] = [id, title, 0, poster_path]
            combined_scores[title][2] += weight_pearson * pearson_score

        # Convert the combined scores back to a sorted list
        combined_scores_list = sorted(
            combined_scores.values(), key=lambda x: x[2], reverse=True)

        return combined_scores_list

   # Combine the scores
    combined_scores = combine_scores(
        content_filtering_score, collaborative_filtering_score, weight_cosine=0.3, weight_pearson=0.7)

    return combined_scores
