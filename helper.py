import base64
import streamlit as st
import pandas as pd
import pickle
@st.cache_data
def get_data():
    #  data cached and accessed through session_state
    df_id2game = pd.read_csv("df_id2game.csv", index_col=0)  # index: game id column:game name
    df_game2id = pd.read_csv("df_game2id.csv", index_col=0)  # index: game name column:game id
    df_id2user = pd.read_csv("df_id2user.csv", index_col=0)  # index: user id column:user name
    df_user2id = pd.read_csv("df_user2id.csv", index_col=0)  # index: username column:user id
    # data for user-based and item-based collaborative filtering
    df_ratings = pd.read_csv("df_ratings_10.csv", index_col=0, dtype={"gameId": "uint32", "rating": "int8"})
    df_ratings.index = df_ratings.index.astype("uint32")
    # data for content-based recommendation
    with open('df_content_dict.pkl', 'rb') as file:
        df_content_dict = pickle.load(file)

    return df_ratings, df_content_dict, df_id2game, df_game2id, df_id2user, df_user2id

@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def get_png_as_page_bg(png_file):
    main_bg = get_base64_of_bin_file(png_file)
    return f"""
            <style>
            .stApp {{
                background: url(data:image/png;base64,{main_bg});
                background-size: cover
            }}
            </style>
            """


def crete_ui(st):
    st.markdown("<h1 style='text-align: center; '>BOARD GAME RECOMMENDER</h1>", unsafe_allow_html=True)

    st_tabs, st_buttons, entered_ids = {}, {}, {}
    st_tabs["content"], st_tabs["user"], st_tabs["item"] = st.tabs(["Content-Based Recommender", "User-Based Collaborative Filtering", "Item-Based Collaborative Filtering"])
    st_tabs["content"].markdown("""
      <style>
      .css-16idsys.e16nr0p34 {
        background-color: green;
      }
      </style>
    """, unsafe_allow_html=True)
    #st_tabs["content"].markdown("<h2 style='text-align:center; color:red;'>Content-Based Recommendation</h2>",  unsafe_allow_html=True)
    st_tabs["user"].markdown("<h2 style='text-align:center; color:green;'>User-Based <br> Collaborative Filtering</h2>",
                             unsafe_allow_html=True)
    st_tabs["item"].markdown("<h2 style='text-align:center; color:blue;'>Item-Based <br> Collaborative Filtering</h2>",
                             unsafe_allow_html=True)

    for col_name in st_tabs:
        st_buttons[col_name] = st_tabs[col_name].button("Get recommendations", key=col_name)
        message = "user name or user id" if col_name == "user" else "game name or game id"
        entered_ids[col_name] = st_tabs[col_name].text_input(f"Enter  {message}  (Skip for random {message})",
                                                             key="target_" + col_name)
    return st_tabs, st_buttons, entered_ids
