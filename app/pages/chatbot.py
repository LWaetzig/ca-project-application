import time

import streamlit as st
from src.rag import answer_with_retrieval

st.title("Chatbot")

# display message history
for message in st.session_state["chatbot_messages"]:
    avatar = "ai" if message["role"] == "assistant" else "user"
    with st.chat_message(name=message["role"], avatar=avatar):
        st.write(message["content"])


prompt = st.chat_input(placeholder="Frag mich etwas...", accept_file=False)
if prompt:
    # add message to message history
    st.session_state["chatbot_messages"].append({"role": "user", "content": prompt})

    # display message in chat interface
    with st.chat_message(name="user", avatar="user"):
        st.write(prompt)

    with st.chat_message(name="assistant", avatar="ai"):
        with st.spinner("Antwort wird generiert...", show_time=True):

            answer, sources = answer_with_retrieval(prompt)

        message_placeholder = st.empty()
        full_response = str()

        for char in answer:
            full_response += char
            message_placeholder.markdown(full_response + "█ ", unsafe_allow_html=True)
            time.sleep(0.05)  # simulate typing effect

        message_placeholder.markdown(answer, unsafe_allow_html=True)

        if sources:
            with st.expander("Quellen", expanded=False):
                for s in sources:
                    st.markdown(
                        f"- [{s.get('date', '')}] {s.get('preview', '')} (sim={s.get('similarity', 0)})"
                    )

        st.session_state["chatbot_messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
