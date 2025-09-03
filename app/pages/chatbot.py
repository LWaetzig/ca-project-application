import gc
import logging
import time

import psutil
import streamlit as st
from src.rag import answer_with_retrieval, cleanup_memory
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def log_memory_usage(stage: str) -> tuple[Any, Any]:
    """Log current memory usage

    Args:
        stage (str): stage in which the response generation is

    Returns:
        tuple[Any, Any]: Memory usage information
    """
    try:
        # Force garbage collection before measuring
        gc.collect()

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Chatbot memory usage at {stage}: {memory_mb:.2f} MB")

        # Log system memory
        sys_memory = psutil.virtual_memory()
        logger.info(
            f"System memory at {stage}: {sys_memory.percent:.1f}% used, {sys_memory.available / 1024 / 1024 / 1024:.2f} GB available"
        )

        # Return values for UI display
        return memory_mb, sys_memory.percent
    except Exception as e:
        logger.error(f"Error logging memory usage: {e}")
        return 0, 0


st.title("🧠 Chatbot")

# display message history
if "chatbot_messages" not in st.session_state:
    st.session_state["chatbot_messages"] = [
        {
            "role": "assistant",
            "content": "Hallo! Ich bin dein virtueller Assistent.",
        }
    ]

for message in st.session_state["chatbot_messages"]:
    avatar = "ai" if message["role"] == "assistant" else "user"
    with st.chat_message(name=message["role"], avatar=avatar):
        st.write(message["content"])


prompt = st.chat_input(placeholder="Ask me something...", accept_file=False)
if prompt:
    start_time = time.time()
    log_memory_usage("before_chat_processing")
    logger.info(f"Processing chat input: {prompt[:100]}...")

    # add message to message history
    st.session_state["chatbot_messages"].append({"role": "user", "content": prompt})

    # display message in chat interface
    with st.chat_message(name="user", avatar="user"):
        st.write(prompt)

    with st.chat_message(name="assistant", avatar="ai"):
        # Create placeholder for streaming response
        response_placeholder = st.empty()

        with st.spinner("Antwort wird generiert...", show_time=True):
            log_memory_usage("before_answer_generation")
            answer_start = time.time()
            try:
                answer, sources = answer_with_retrieval(prompt)
                answer_time = time.time() - answer_start

                log_memory_usage("after_answer_generation")
                logger.info(f"Answer generated successfully in {answer_time:.2f}s")
            except Exception as e:
                answer_time = time.time() - answer_start
                logger.error(f"Error generating answer: {e}")
                log_memory_usage("after_answer_error")
                answer = f"Es ist ein Fehler aufgetreten: {str(e)}"
                sources = []

        # Optimized typing effect (faster for better performance)
        message_placeholder = st.empty()
        full_response = ""

        # Faster typing effect - show chunks instead of character by character
        words = answer.split()
        chunk_size = max(1, len(words) // 20)

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "█", unsafe_allow_html=True)
            time.sleep(0.1)

        # Final response without cursor
        message_placeholder.markdown(answer, unsafe_allow_html=True)

        if sources:
            with st.expander(f"📚 Quellen ({len(sources)})", expanded=False):
                for s in sources:
                    similarity_color = (
                        "🟢"
                        if s.get("similarity", 0) > 0.7
                        else "🟡" if s.get("similarity", 0) > 0.5 else "🔴"
                    )
                    st.markdown(
                        f"{similarity_color} **[{s.get('date','')}]** {s.get('speaker', 'Unknown')}\n\n"
                        f"_{s.get('preview','')}_ "
                        f"(similarity: {s.get('similarity', 0):.3f})"
                    )
                    st.divider()

        st.session_state["chatbot_messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

    # Force aggressive garbage collection after processing
    cleanup_memory()
    log_memory_usage("after_chat_complete")
