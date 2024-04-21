import streamlit as st
from base_recommender import BaseRecommender
from collaborative_filtering import CollaborativeFiltering
from content_based_recommender import ContentBasedRecommender
from user_based_recommender import UserBasedCF
from item_based_recommender import ItemBasedCF
from helper import *

st.set_page_config(page_title="Game Board Recommender", layout="wide")
st.image('images/top_image.jpg',  use_column_width=None, clamp=False, channels="RGB", output_format="auto")

st_tabs, st_buttons, entered_ids = crete_ui(st)

df_ratings, df_content_dict, df_id2game, df_game2id, df_id2user, df_user2id = get_data()

if 'df_recommendations' not in st.session_state:
    df_ratings, df_content, df_id2game, df_game2id, df_id2user, df_user2id = get_data()

    BaseRecommender.df_id2name = {"game": df_id2game, "user": df_id2user, "content": df_id2game}
    BaseRecommender.df_name2id = {"game": df_game2id, "user": df_user2id, "content": df_game2id}
    # User-based and item-based CF
    CollaborativeFiltering.df_ratings = df_ratings
    CollaborativeFiltering.set_top_users(num_top_users=10000)
    st.session_state['recommenders'] = dict()
    st.session_state['recommenders']["user"] = UserBasedCF(threshold_in_common=10)
    st.session_state['recommenders']["item"] = ItemBasedCF(threshold_in_common=10)
    # Content-based recommender
    st.session_state['recommenders']["content"] = ContentBasedRecommender(df_content_dict)
    # Empty dataframes for recommendations
    st.session_state['df_recommendations'] = {"content": pd.DataFrame(), "item": pd.DataFrame(), "user": pd.DataFrame()}
    st.session_state['target_ids'] = {}


if __name__ == '__main__':
    for method_name in st_buttons:
        if st_buttons[method_name]:
            recommender = st.session_state['recommenders'][method_name]
            target_id = recommender.get_target_id(entered_ids[method_name])
            if entered_ids[method_name] != str(target_id):
                st_tabs[method_name].write("Generating random id.")
            st_tabs[method_name].write(f"Target id is {target_id} ")
            if method_name == "user":
                recommender.bar = st_tabs["user"].progress(0, text="Generating recommendations using more than 169000 users. This might take 1 min.")

            st.session_state['df_recommendations'][method_name] = recommender.get_recommendations([target_id], num_recommendations=10 )
            #st_cols[method_name].write(method_name, "Name:", st.session_state['recommenders'][method_name].df_id2name["game"].loc[target_id].item())

        if not st.session_state['df_recommendations'][method_name].empty:
            st_tabs[method_name].dataframe(st.session_state['df_recommendations'][method_name])
