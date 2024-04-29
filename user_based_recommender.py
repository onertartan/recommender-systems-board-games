from collaborative_filtering import CollaborativeFiltering
import pandas as pd


class UserBasedCF(CollaborativeFiltering):
    def __init__(self, threshold_in_common,k_neighbors):
        super().__init__(threshold_in_common,k_neighbors)
        self.name = "user"
        self.ids = CollaborativeFiltering.df_ratings.index.unique().tolist()


    def get_recommendations(self, target_ids, num_recommendations=10):
        target_user_id = target_ids[0]
        if CollaborativeFiltering.df_ratings.index.name == "game_id":
            CollaborativeFiltering.set_index("user_id")
        # ratings given by the target user
        df_target_user = self.df_ratings.loc[target_user_id]
        # similarities between the target user and the other users(for only games in common with the target user)
        df_similarities = self.get_similarities(self.df_ratings_top_users, target_user_id, similarity_metric=None)
        # K-most similar user ids(K-nearest neighbors)
        similar_user_ids = df_similarities.index[:self.k_neighbors]
        # Get ratings  of top k similar users
        df_top_k_similar_users = self.df_ratings_top_users.loc[similar_user_ids]
        # Exclude ratings of the games played by the target user
        df_top_k_similar_users = df_top_k_similar_users[~df_top_k_similar_users.isin(df_target_user["game_id"].tolist())]
        # Find counts of the games in common among top-k neighbors
        df_game_counts = df_top_k_similar_users.groupby('game_id')['game_id'].count().to_frame("count")
        # Find average ratings of the games
        df_game_average_ratings = df_top_k_similar_users.groupby("game_id")["rating"].mean().to_frame("average_rating")
        df_recommended_games = pd.merge(df_game_counts, df_game_average_ratings, left_index=True, right_index=True)
        # index of the dataframe(user-ids) is float due to the merge, convert to integer
        df_recommended_games.index = df_recommended_games.index.astype("uint32")
        # Sort the games by (user) count first, then by average rating
        df_recommended_games.sort_values(by=["count", "average_rating"], ascending=False, inplace=True)
        # Add game names as a new column
        df_recommended_games["game_name"] = self.df_id2name["game"].loc[df_recommended_games.index]
        df_recommended_games = df_recommended_games.round(2)
        return df_recommended_games[:num_recommendations]
