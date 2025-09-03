import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Bundestag Speech Analysis", layout="wide")

st.title("Bundestag Speech Analysis")
st.divider()

# Main Features Overview
st.header("Available Tools")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
    ### 📊 Analysis
    
    **Interactive data analysis and visualizations**
    
    **Features:**
    - Statistical analysis of political data
    - Interactive charts and visualizations
    - Data exploration and filtering tools
    - Quantitative insights and trends
    - Export capabilities for further analysis
    
    **Available Analysis:**
    - Speech frequency and patterns
    - Political party comparisons  
    - Temporal trends and changes
    - Speaker activity and engagement
    """
    )

with col2:
    st.markdown(
        """
    ### 🧠 Chatbot*
    
    **Ask questions about parliamentary data using natural language**
    
    **Features:**
    - Natural language queries about speeches and politics
    - Semantic search through parliamentary documents
    - Source citations for all answers
    - AI-powered responses with context
    - Query German political data in your own words
    
    """
    )
    with st.expander("ℹ️ *About the Chatbot"):
        st.markdown(
            """
            This chatbot is intended as an additional feature, with the aim of testing the newly released Ollama Docker images.
            The chatbot does not have high expectations for high-quality answers, since both compute resources and methods were only available to a limited extent.
            The most challenging aspect of this chatbot was dealing with limited memory and the vast amount of data. If you encounter memory issues, restrict loading data in the entrypoint.py file
            """
        )


btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    if st.button("🚀 Start Analysis", use_container_width=True):
        st.switch_page("pages/analysis.py")

with btn_col2:
    if st.button("🚀 Try Chatbot", use_container_width=True):
        st.switch_page("pages/chatbot.py")
