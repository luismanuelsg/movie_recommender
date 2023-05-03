'''Implements functions for making predictions.'''
import pandas as pd
import numpy as np
import pickle

import warnings
warnings.filterwarnings(action='ignore')

#We should customize this functions to take a dictionary user_query as an argument
class Recommenders():
    def __init__(self):
        self.RATINGS = pd.read_csv('./data/ratings.csv', index_col= 0)
        with open('./data/factorizer_NMF.pkl', 'rb') as file_in:
            self.NMF_model = pickle.load(file_in)
        self.Q = self.NMF_model.components_
        self.MOVIES = self.RATINGS.columns
        self.ratings_mean = self.RATINGS.mean()

    def NMF_recommender(self, user_query, chart_len):
        user_query_list = list(user_query.keys())
        imputed_query = pd.DataFrame(user_query, index = ['new_user'], columns= self.MOVIES)
        imputed_query.fillna(value= self.ratings_mean, inplace= True)
        P_user = self.NMF_model.transform(imputed_query)
        R_user = np.dot(P_user, self.Q)
        R_user = pd.DataFrame(R_user, columns= self.MOVIES, index=['new_user'])
        R_user_Tsorted = R_user.T.sort_values(by='new_user', ascending=False)
        recommendables = list(R_user_Tsorted.index)
        user_recommendations = [movie for movie in recommendables if movie not in user_query_list]
        
        return user_recommendations[:chart_len]

    def cos_sim_recommender(self, user_query, chart_len):
        return list(self.MOVIES)
