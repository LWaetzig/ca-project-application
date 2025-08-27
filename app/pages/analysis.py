import streamlit as st

from src.analysis import (
    table_explorer,
    variable_analysis,
    advanced_analysis,
    dashboard_view,
)

st.title("Bundestag Speech Analysis")

# Create main tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Overview (Dashboard)",
        "Table Explorer",
        "Variable Analysis",
        "Advanced Analysis",
    ]
)

with tab1:
    dashboard_view()

with tab2:
    table_explorer()

with tab3:
    variable_analysis()

with tab4:
    advanced_analysis()
