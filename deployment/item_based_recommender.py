from collaborative_filtering import CollaborativeFiltering

class ItemBasedCF(CollaborativeFiltering):
    def __init__(self,  threshold_in_common):
        super().__init__(threshold_in_common)
        self.name = "game"
        self.ids = CollaborativeFiltering.df_ratings["game_id"].unique().tolist()

    def get_recommendations(self, target_ids,  num_recommendations=10):
        if CollaborativeFiltering.df_ratings.index.name == "user_id":
            CollaborativeFiltering.set_index("game_id")

        candidate_ids = set()
        K = num_recommendations

        for target_id in target_ids:
            # for each target game id calculate similarities with all games among top users
            df_similarities_all_games = self.get_similarities(CollaborativeFiltering.df_ratings_top_users, target_id, slice_length=2000, similarity_metric=None)
            # exclude target ids and get K top games
            candidate_ids = candidate_ids.union(df_similarities_all_games.loc[~df_similarities_all_games.index.isin(target_ids)].iloc[:K].index.tolist())
            print("target_id",target_id)
        # Obtain ratings given to candidate games
        df_ratings_candidates = CollaborativeFiltering.df_ratings_top_users.loc[list(candidate_ids)]
        print("******************************")
        print("df_ratings_candidates.head", df_ratings_candidates.head())
        # Calculate similarities of candidates with the first target game(There must be at least one target game)
        df_similarities_candidates = self.get_similarities( df_ratings_candidates, target_ids[0],
                                                            slice_length=2000, similarity_metric=None)
        print(">##################################################################")
        print("df_similarities_candidates.head()",df_similarities_candidates.head())
        # Calculate similarities of candidates with the other target games (if they are given)
        for i in range(1, len(target_ids)):
            df_similarities_candidates += self.get_similarities(df_ratings_candidates, target_ids[i],
                                                          slice_length=2000, similarity_metric=None)
        print("FINAL",df_similarities_candidates.head())
        print("type",type(df_similarities_candidates))
        return df_similarities_candidates.sort_values(by=df_similarities_candidates.columns[0], ascending=False)