import numpy as np
import pandas as pd
from base_recommender import BaseRecommender
from sklearn.metrics.pairwise import pairwise_distances

class ContentBasedRecommender(BaseRecommender):

    def __init__(self, df_content_dict,  category_factor=0.7, mechanic_factor=0.2, family_factor=0.1):
        """
        @param df_content_dict: Dictionary for sparse dataframes:df_rank,df_category,df_mechanic and df_family
        @param category_factor: weight of boardgamecategory in similarity calculation
        @param mechanic_factor: weight of boardgamemechanic in similarity calculation
        @param family_factor  : weight of boardgamefamily in similarity calculation
        """
        super().__init__()
        self.name = "content"
        self.ids = df_content_dict["rank"].index.unique().tolist()
        self.df_content_dict = df_content_dict
        self.factors = {"category": category_factor, "mechanic": mechanic_factor, "family": family_factor}

    def get_similarities(self, df, target_id, similarity_metric="jaccard"):
        return 1-pairwise_distances(np.asarray(df,dtype=bool), np.asarray(df.loc[target_id],dtype=bool).reshape(1, -1), metric=similarity_metric)

    def rescale_factors(self, target_id):
        """
        rescales factors if mechanic or family columns are empty
        For example if family columns are empty and fc=0.6,fm=0.2 and ff=0.2
        rescaled factors will be fc=0.75, fm=0.25, ff=0
        :param target_id: target_game_id
        """
        if (self.df_content_dict["mechanic"].loc[target_id] == 0).all():
            self.factors["mechanic"] = 0
        if (self.df_content_dict["family"].loc[target_id] == 0).all():
            self.factors["family"] = 0
        factors_sum = sum(self.factors.values())
        self.factors["category"] = self.factors["category"]/ factors_sum
        self.factors["mechanic"] = self.factors["mechanic"]/ factors_sum
        self.factors["family"] = self.factors["family"] / factors_sum

    def get_recommendations(self, target_ids, num_recommendations=10):
        target_game_id = target_ids[0]
        self.rescale_factors(target_game_id)
        # Content based (df_category and df_mechanic have common index(filtered ids), df_id_name contains all game ids
        df_similarities = self.df_content_dict["rank"]
        df_similarities["similarity"] = 0

        for key in ["category", "mechanic", "family"]:
            df_similarities[["similarity"]] += self.factors[key]*self.get_similarities(self.df_content_dict[key], target_game_id)
        df_similarities = df_similarities.sort_values(by=["similarity", "Board_Game_Rank"], ascending=[False,True])
        recommended_game_ids = df_similarities.iloc[:num_recommendations].index
        df_recommendations = pd.concat((self.df_id2name["content"].loc[recommended_game_ids], df_similarities.iloc[:num_recommendations]), axis=1)
        df_recommendations.index.name = "game_id"
        return df_recommendations
