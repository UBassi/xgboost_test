from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from string import punctuation
from nltk.corpus import stopwords
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib


if __name__ == "__main__":

    data = pd.read_csv("https://raw.githubusercontent.com/algorithmiaio/model-deployment/master/xgboost_notebook_to_algorithmia/data/amazon_musical_reviews/Musical_instruments_reviews.csv")
    data.head()

    data["reviewText"].iloc[1]

    import nltk
    nltk.download('stopwords')

    def threshold_ratings(data):
        def threshold_overall_rating(rating):
            return 0 if int(rating)<=3 else 1
        data["overall"] = data["overall"].apply(threshold_overall_rating)

    def remove_stopwords_punctuation(data):
        data["review"] = data["reviewText"] + data["summary"]

        puncs = list(punctuation)
        stops = stopwords.words("english")

        def remove_stopwords_in_str(input_str):
            filtered = [char for char in str(input_str).split() if char not in stops]
            return ' '.join(filtered)

        def remove_punc_in_str(input_str):
            filtered = [char for char in input_str if char not in puncs]
            return ''.join(filtered)

        def remove_stopwords_in_series(input_series):
            text_clean = []
            for i in range(len(input_series)):
                text_clean.append(remove_stopwords_in_str(input_series[i]))
            return text_clean

        def remove_punc_in_series(input_series):
            text_clean = []
            for i in range(len(input_series)):
                text_clean.append(remove_punc_in_str(input_series[i]))
            return text_clean

        data["review"] = remove_stopwords_in_series(data["review"].str.lower())
        data["review"] = remove_punc_in_series(data["review"].str.lower())

    def drop_unused_colums(data):
        data.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime', "reviewText", "summary"], axis=1, inplace=True)

    def preprocess_reviews(data):
        remove_stopwords_punctuation(data)
        threshold_ratings(data)
        drop_unused_colums(data)

    preprocess_reviews(data)
    data.head()

    rand_seed = 42
    X = data["review"]
    y = data["overall"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_seed)

    params = {"max_depth": range(9,12), "min_child_weight": range(5,8)}
    rand_search_cv = RandomizedSearchCV(XGBClassifier(), param_distributions=params, n_iter=1)

    model  = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('model', rand_search_cv)
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {round(acc * 100, 2)}")

    joblib.dump(model, "model.pkl", compress=True)
