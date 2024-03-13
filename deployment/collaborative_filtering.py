from abc import ABC
from base_recommender import BaseRecommender
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from stqdm import stqdm
import numpy as np


class CollaborativeFiltering(BaseRecommender, ABC):
    df_ratings = None
    df_ratings_top_users = None
    def __init__(self, threshold_in_common):
        """
        @param threshold_in_common: minimum number of games in common with other users for user-based cf
                                    minimum number of user in common with other games for item-based cf
        """
        super().__init__()
        self.threshold_in_common = threshold_in_common
        self.bar = None

    @staticmethod
    def set_index(index_name):
        CollaborativeFiltering.df_ratings = CollaborativeFiltering.df_ratings.reset_index().set_index(index_name)
        CollaborativeFiltering.df_ratings_top_users = CollaborativeFiltering.df_ratings_top_users.reset_index().set_index(index_name)

    @staticmethod
    def set_top_users(num_top_users):
        top_user_ids = CollaborativeFiltering.df_ratings.index.value_counts()[:num_top_users].index
        CollaborativeFiltering.df_ratings_top_users = CollaborativeFiltering.df_ratings.loc[top_user_ids]
        CollaborativeFiltering.df_ratings_top_users.index.name = "user_id"

    def create_pivot_table(self, df_ratings_for_similarity_calculation, target_id, slice_length):
        """
        @param df_ratings_top_users    : the most active n users (n is selected by user, default 10000)
        @param target_id               : target user id for user-based cf and target game for item-based cf
        @param slice_length            : batch size for generator
                                       : pivot table for batch(index:user id column:game id for user-based cf)
                                                              (index:game id column:user id for item-based cf)
        """
        ids_rows = df_ratings_for_similarity_calculation.index.unique().tolist()
        df_ratings_target = CollaborativeFiltering.df_ratings.loc[target_id]
        ids_cols = df_ratings_target.iloc[:, 0]  # df_ratings_target["game_id"] # game_ids_played_by_target_user

        for i in range(0,1+ len(ids_rows)//slice_length):
            ids_rows_subset = set(ids_rows[i*slice_length:(i +1)* slice_length]) # select a batch of users
            print("target_id",target_id,"ids_rows",ids_rows_subset)
            ids_rows_subset.discard(target_id)  # exclude the target user if it is in the user-batch
            # only users in the user batch(slice)
            df_ratings_subset = df_ratings_for_similarity_calculation[df_ratings_for_similarity_calculation.index.isin(ids_rows_subset)]
            # filter ratings: select the ratings of the games played by the target user
            # item-based  cf: select the ratings of the games played by the users who played the target game
            df_ratings_subset = df_ratings_subset[df_ratings_subset.iloc[:, 0].isin(ids_cols)]  # df_ratings_subset[df_ratings_subset["game_id"].isin(ids_cols)]
            # only users who have games in common (with the target user) more than the threshold
            df_counts = df_ratings_subset.index.value_counts()
            df_ratings_subset = df_ratings_subset.loc[df_counts[df_counts > self.threshold_in_common].index]
            df_ratings_subset = pd.concat((df_ratings_subset, df_ratings_target), axis=0)
            yield df_ratings_subset.pivot_table(index=df_ratings_subset.index, columns=df_ratings_subset.columns[0], values="rating")  # column is game_id

    def get_similarities(self, df_ratings_for_similarity_calculation,target_id, slice_length=2000, similarity_metric=None):
        df_similarities = pd.DataFrame()

        if len(self.df_ratings.loc[target_id]) < self.threshold_in_common: # If the target user did not rate enough games we do not have to proceed any more.
            print("The number of games rated by the target user is less than threshold for number of games in common.")
        else:
            total_steps = 1 + len(df_ratings_for_similarity_calculation.index.unique()) // slice_length
            pivot_generator = self.create_pivot_table(df_ratings_for_similarity_calculation, target_id, slice_length=slice_length)
            for _ in stqdm(range(total_steps), st_container=self.bar): #pivot table including users who have at least threshold games in common with the target user
                df_pivot_filtered_slice = next(pivot_generator)
                df_pivot_target = df_pivot_filtered_slice.loc[target_id]               #row of the pivot table related to the target user
                df_pivot_others_slice = df_pivot_filtered_slice.loc[df_pivot_filtered_slice.index != target_id] # pivot table where index includes users other than target user

                if not df_pivot_others_slice.empty:
                    # calculate  correlations of the target user with other (filtered) users in pivot table slice
                    if similarity_metric == None:
                        df_similarities_slice = df_pivot_others_slice.corrwith(df_pivot_target, axis=1, numeric_only=True)
                    # calculate distances of the target user with other (filtered) users in pivot table slice
                    else:
                        df_similarities_slice = pd.DataFrame(pairwise_distances(df_pivot_others_slice,df_pivot_target.to_numpy().reshape(1,-1), metric=similarity_metric),index=df_pivot_others_slice.index )

                    if len(df_pivot_others_slice) > 0:
                        # add "number of games in common" column as a new column to df_similarities_slice
                        df_num_in_common = df_pivot_others_slice.notna().sum(axis=1)
                        df_similarities_slice = pd.concat((df_similarities_slice, df_num_in_common), axis=1)
                        # save similarities and "number of games in common" in  df_similarities
                        df_similarities = pd.concat((df_similarities, df_similarities_slice))
            if df_similarities.empty or df_similarities.iloc[0,0]==np.nan:
                print("No recommendations found")
            else:
                name_rows_pivot = self.df_ratings.index.name[:4]  # user
                name_cols_pivot = self.df_ratings.iloc[:, 0].name[:4]  # game
                similarity_column = similarity_metric if similarity_metric else "Correlation_with_the_selected_"+name_rows_pivot
                df_similarities.columns = [similarity_column, "num_of_"+name_cols_pivot+"_in_common"]
                df_similarities.sort_values(by=similarity_column, ascending=False if similarity_metric is None else True, inplace=True)

                df_similarities[name_rows_pivot+"_name"] = self.df_id2name[self.name].loc[df_similarities.index]
                df_similarities.index.name = name_rows_pivot+"_id"

        return df_similarities
