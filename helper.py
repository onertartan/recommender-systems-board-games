import base64
import streamlit as st
import pandas as pd
import pickle

@st.cache_data
def get_data():
    #  data cached and accessed through session_state
    df_id2game = pd.read_csv("./data/df_id2game.csv", index_col=0)  # index: game id column:game name
    df_game2id = pd.read_csv("./data/df_game2id.csv", index_col=0)  # index: game name column:game id
    df_id2user = pd.read_csv("./data/df_id2user.csv", index_col=0)  # index: user id column:user name
    df_user2id = pd.read_csv("./data/df_user2id.csv", index_col=0)  # index: username column:user id
    # data for user-based and item-based collaborative filtering
    df_ratings = pd.read_csv("./data/df_ratings_10.csv", index_col=0, dtype={"gameId": "uint32", "rating": "int8"})
    df_ratings.index = df_ratings.index.astype("uint32")
    df_descriptions = pd.read_csv("./data/df_descriptions.csv", index_col=0)
    # data for content-based recommendation
    with open('./data/df_content_dict.pkl', 'rb') as file:
        df_content_dict = pickle.load(file)
    return df_ratings, df_content_dict, df_id2game, df_game2id, df_id2user, df_user2id, df_descriptions


@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_png_as_page_bg(png_file):
    main_bg = get_base64_of_bin_file(png_file)
    return f""" <style>.stApp {{ background: url(data:image/png;base64,{main_bg}); background-size: cover}} </style>"""


def create_ui(st):
    st.set_page_config(page_title="Game Board Recommender", layout="wide")
    st.image('images/top_image.jpg', use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    # CSS styling
    with open('./style/my_style.css') as f:
        css = f.read()

    st.markdown("<h1>BOARD GAME RECOMMENDER</h1>", unsafe_allow_html=True)
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

    st_sliders, st_tabs, st_tab_cols, st_buttons, entered_ids = {}, {}, {}, {},{}
    st_sliders["num_recommendations"] = st.slider("Number of recommendations", min_value=1, max_value=20, value=10, step=1,  key="num_recommendations",  label_visibility="visible")

    st_tabs["content"], st_tabs["user"], st_tabs["item"] = st.tabs(["Content-Based Recommender", "User-Based Collaborative Filtering", "Item-Based Collaborative Filtering"])
    st_tab_cols={"content": st_tabs["content"].columns(3), "user": st_tabs["user"].columns(3), "item": st_tabs["item"].columns(3)}

    st_sliders["content"] = {"category_factor": st_tab_cols["content"][0].slider("Category factor", min_value=.0, max_value=1., value=.7, step=.1),
                              "mechanic_factor": st_tab_cols["content"][0].slider("Mechanic factor", min_value=.0, max_value=1., value=.2, step=.1),
                              "family_factor": st_tab_cols["content"][0].slider("Category family factor", min_value=.0, max_value=1., value=.1, step=.1)
                             }

    st_sliders["user"] = {"threshold_in_common": st_tab_cols["user"][1].slider("Number of users in common", min_value=1, max_value=20, value=10),
                          "k_neighbors": st_tab_cols["user"][1].slider("K neighbors", min_value=1, max_value=20, value=3)}
    st_sliders["item"] = {"threshold_in_common": st_tab_cols["item"][2].slider("Number of games in common", min_value=1, max_value=20, value=10),
                          "k_neighbors": st_tab_cols["item"][2].slider("K neighbors", min_value=1, max_value=20, value=10)}

    for tab_name in st_tabs:
        st_buttons[tab_name] = st_tabs[tab_name].button("Get recommendations", key=tab_name)
        message = "user name or user id" if tab_name == "user" else "game name or game id"
        entered_ids[tab_name] = st_tabs[tab_name].text_input(f"Enter  {message}  (Skip for random {message})", key="target_" + tab_name)

    return  st_tabs, st_buttons, entered_ids, st_sliders
