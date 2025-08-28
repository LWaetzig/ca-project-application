import time

import streamlit as st

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

            # get context for user query
            context = "Retrieving context for user query..."
            time.sleep(2)  # simulate context retrieval

            # generate response
            response = f"Generating response for user query: {context}"
            time.sleep(2)  # simulate response generation

            st.session_state["chatbot_messages"].append(
                {"role": "assistant", "content": response, "sources": context}
            )

        message_placeholder = st.empty()
        full_response = str()

        for char in response:
            full_response += char
            message_placeholder.markdown(full_response + "█ ", unsafe_allow_html=True)
            time.sleep(0.05)  # simulate typing effect

        message_placeholder.markdown(response, unsafe_allow_html=True)

        with st.expander("Quellen", expanded=False):
            st.markdown(context, unsafe_allow_html=True)
