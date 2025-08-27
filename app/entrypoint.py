import os

import streamlit as st
from dotenv import load_dotenv
from src.utils import preload_table_content

load_dotenv()


def app():

    st.set_page_config(
        page_title="Analysing Bundestag Speeches",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state for analysis
    if "data_loaded" not in st.session_state:
        st.session_state["data_loaded"] = False

    if not st.session_state["data_loaded"]:
        st.session_state["db_tables"] = [
            "speeches",
            "contributions",
            "politicians",
            "electoral_terms",
            "factions",
        ]

        # Load data
        for table in st.session_state["db_tables"]:
            st.session_state[table] = preload_table_content(
                os.path.join("data", f"{table}.parquet")
            )

        st.session_state["data_loaded"] = True

    # init session state for chatbot
    if "chatbot_messages" not in st.session_state:
        st.session_state["chatbot_messages"] = [
            {
                "role": "assistant",
                "content": "Hallo! Ich bin dein virtueller Assistent.",
            }
        ]

    with st.sidebar:

        pg = st.navigation(
            {
                "Home": [
                    st.Page(os.path.join("pages", "home.py"), title="Home", icon="🏠")
                ],
                "Tools": [
                    st.Page(
                        os.path.join("pages", "analysis.py"),
                        title="Analysis",
                        icon="📊",
                    ),
                    st.Page(
                        os.path.join("pages", "chatbot.py"), title="Chatbot", icon="🤖"
                    ),
                ],
            }
        )

    pg.run()


if __name__ == "__main__":
    app()
