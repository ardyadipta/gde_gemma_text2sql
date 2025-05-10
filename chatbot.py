import streamlit as st
import requests

# Streamlit UI setup
st.title("💬 Gemma Chatbot")

# Sidebar for server configuration
st.sidebar.header("Server Configuration")
external_ip = st.sidebar.text_input("External IP", value="34.124.232.240")
port = st.sidebar.text_input("Port", value="8000")

# Validate input
if external_ip == "":
    st.warning("Please input the External IP address in the sidebar.")
    st.stop()

base_url = f"http://{external_ip}:{port}"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

# User input
prompt = st.chat_input("Ask Gemma something...")

if prompt:
    # User message
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call Gemma API on provided server
    try:
        response = requests.post(
            f"{base_url}/gemma_request",
            json={"text": prompt},
            timeout=1000
        )
        response.raise_for_status()
        result = response.json()
        print(result)

        answer = result.get("text", "Sorry, I didn't understand.")
    except requests.exceptions.RequestException as e:
        answer = f"Error: {e}"

    # Gemma's reply
    st.session_state.messages.append({"role": "assistant", "text": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
