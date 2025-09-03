import logging
import os

import pandas as pd
import psutil
import streamlit as st
from src.utils import preload_table_content

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

        # Load data with memory limitations
        data_limits = {
            "speeches": None,  # limit to 5000 rows or less if docker exits with code 137 -> OOM error
            "contributions": None,  # limit to 5000 or less rows if docker exits with code 137 -> OOM error
            "politicians": None,
            "electoral_terms": None,
            "factions": None,
        }

        for table in st.session_state["db_tables"]:
            logger.info(f"Loading table: {table}")
            try:
                max_rows = data_limits.get(table)
                if max_rows:
                    logger.info(f"Limiting {table} to {max_rows} rows")
                st.session_state[table] = preload_table_content(
                    os.path.join("data", f"{table}.parquet"), max_rows=max_rows
                )
                logger.info(f"Loaded {table}: {len(st.session_state[table])} rows")
            except Exception as e:
                logger.error(f"Error loading {table}: {e}")
                st.session_state[table] = pd.DataFrame()

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
        # Memory usage monitor
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            sys_memory = psutil.virtual_memory()

            with st.expander("📊 Memory Monitor", expanded=False):
                st.metric("Process Memory", f"{memory_mb:.1f} MB")
                st.metric("System Memory Usage", f"{sys_memory.percent:.1f}%")
                if sys_memory.percent > 85:
                    st.warning("⚠️ High memory usage detected!")

                # Memory cleanup button
                if st.button(
                    "🧹 Clear Memory Cache", help="Clear cached data to free memory"
                ):
                    try:
                        from src.rag import cleanup_memory

                        cleanup_memory()
                        st.success("Memory cache cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing cache: {e}")

        except Exception:
            # Silently fail if memory monitoring is not available
            pass

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
                        os.path.join("pages", "chatbot.py"), title="Chatbot", icon="🧠"
                    ),
                ],
            }
        )

    pg.run()


if __name__ == "__main__":
    app()
