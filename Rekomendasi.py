import numpy as np
import pandas as pd
import surprise
import json
from surprise import *
import pickle
from surprise import accuracy
from surprise.model_selection import train_test_split
import seaborn as sns
# import cinemagoer
from imdb import Cinemagoer
import collections

df = pd.read_csv('ratings.csv')
df.drop(['timestamp'], axis=1, inplace=True)

# resu = df.to_json(orient="split")
# parsed = json.loads(resu)
# cod = json.dumps(parsed, indent=4)

# print(cod)


# js = df.to_json(orient="split")
# parsed = json.loads(js)
# json.dumps(parsed, indent=4)

class RecommenderSystem:
    def __init__(self, data):
        self.df = pd.read_csv(data)
        self.all_movie = self.df.movieId.unique()
        self.data = None

    def fit(self):
        data = Dataset.load_from_df(self.df[['userId', 'movieId', 'rating']], Reader(rating_scale=(1, 5)))
        self.data = data.build_full_trainset()
        # pickle.dump(self.data, open('model.pkl', 'wb'))
        # self.pickled_model = pickle.load(open('model.pkl', 'rb'))
        self.model = SVD()
        self.model.fit(self.data)

    def recommend(self, userId, top=1):
        watch_movie = self.df[df.userId == userId].movieId
        not_watch_movie = [movieId for movieId in self.all_movie if movieId not in watch_movie]
        score = [self.model.predict(userId, movieId).est for movieId in not_watch_movie]

        result = pd.DataFrame({'movieId': not_watch_movie, 'score': score})
        result.sort_values('score', ascending=False, inplace=True)
        return result.head(top)

#%%
def results(userId):
    recsys = RecommenderSystem('ratings.csv')
    recsys.fit()
    return recsys.recommend(userId)

def hasil(results):
    object_list = []
    ia = Cinemagoer()

    for row in results.itertuples():
        movie = ia.get_movie(str(row.movieId))

        d = collections.OrderedDict()
        d["id"] = row[0]
        d["rating"] = movie.get('rating', 0.0)
        d["plot"] = movie['plot'][0] if movie.get('plot') else None
        d["genres"] = movie.get('genres', 'No genres')
        d["poster"] = movie['cover url'] if movie.get('cover url') else None
        # d["poster"] = movie.get('cover url', 'https://www.imdb.com/title/tt0111161/mediaviewer/rm1628034048')
        # d["cast"] = movie.get('cast', 'No cast')
        d["directors"] = [director['name'] for director in movie['directors']]
        # d["writers"] = [writer['name'] for writer in movie['writers']]
        # d["runtime"] = movie.get('runtime', 'No runtime')
        # d["countries"] = movie.get('countries', 'No countries')
        # d["languages"] = movie.get('languages', 'No languages')
        # d["release date"] = movie.get('release date', 'No release date')
        # d["votes"] = movie.get('votes', 'No votes')
        # d["imdbID"] = movie.get('imdbID', 'No imdbID')
        # d["imdbIndex"] = movie.get('imdbIndex', 'No imdbIndex')
        # d["imdbUrl"] = movie.get('imdbUrl', 'No imdbUrl')
        # d["kind"] = movie.get('kind', 'No kind')
        # d["plot outline"] = movie.get('plot outline', 'No plot outline')
        # d["tagline"] = movie.get('tagline', 'No tagline')
        # d["trivia"] = movie.get('trivia', 'No trivia')
        # d["sound mix"] = movie.get('sound mix', 'No sound mix')
        # d["aspect ratio"] = movie.get('aspect ratio', 'No aspect ratio')
        # d["color info"] = movie.get('color info', 'No color info')
        # d["company"] = movie.get('company', 'No company')
        # d["country codes"] = movie.get('country codes', 'No country codes')
        # d["language codes"] = movie.get('language codes', 'No language codes')
        d["title"] = movie.get('title', 'No title')
        d["year"] = movie.get('year', 'No year')

        object_list.append(d)

        # print(movie)
        # print(movie['title'])
        # print(movie['rating'])
        # print(movie['year'])
        # print(movie['plot'])
        # print(movie['genres'])
        # print(movie['poster'])
        # print(movie['cast'])
        # print(movie['directors'])
        # print(movie['writers'])
        # print(movie['runtime'])
        # print(movie['countries'])
        # print(movie['languages'])
        # print(movie['release date'])
        # print(movie['rating'])
        # print(movie['votes'])
        # print(movie['cover url'])
        # print(movie['imdbID'])
        # print(movie['imdbIndex'])
        # print(movie['imdbUrl'])
        # print(movie['kind'])
        # print(movie['plot outline'])
        # print(movie['tagline'])
        # print(movie['trivia'])
        # print(movie['sound mix'])
        # print(movie['aspect ratio'])
        # print(movie['color info'])
        # print(movie['company'])
        # print(movie['country codes'])
        # print(movie['language codes'])
        # print(movie['locations'])
        # print(movie['mpaa'])
        # print(movie['plot'])
        # print(movie['quotes'])
        # print(movie['sound clips'])
        # print(movie['soundtrack'])
        # print(movie['synopsis'])
        # print(movie['title'])
        # print(movie['top 250 rank'])
        # print(movie['votes'])
        # print(movie['write

    return object_list

# print(hasil(results(10)))
# print(results(1))

print(hasil(results(10)))
