import streamlit as st
import os
from src.Database import Database


def main():

    st.title("Open Discourse")

    db = Database(
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
        host=os.getenv("DB_HOST", "database"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "next"),
    )
    query = "SELECT count(*) FROM open_discourse.speeches WHERE electoral_term >= 17"
    data = db.fetch_data(query)
    st.write(data)


if __name__ == "__main__":
    main()
