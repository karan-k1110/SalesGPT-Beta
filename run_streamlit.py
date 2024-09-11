import streamlit as st
import json
import logging
import os
import warnings

from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM
from salesgpt_beta.agents import SalesGPT

load_dotenv()  # loads .env file

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress logging
logging.getLogger().setLevel(logging.CRITICAL)

# LangSmith settings
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_SMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = "sales-gpt-beta1"  # insert your project name here

# Initialize the agent and LLM if not already in session_state
if "sales_agent" not in st.session_state:
    config_path = "/Users/karan/Developer/Personal/projects/SalesGPT-Beta/examples/angel_one_setup.json"

    llm = ChatLiteLLM(temperature=0.2, model_name="gpt-3.5-turbo")

    try:
        with open(config_path, "r", encoding="UTF-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        st.error(f"Config file {config_path} not found.")
        st.stop()
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON from the config file {config_path}.")
        st.stop()

    sales_agent = SalesGPT.from_llm(llm, verbose=True, **config)
    sales_agent.seed_agent()

    # Run determine_conversation_stage once
    sales_agent.determine_conversation_stage()

    # Store in session_state
    st.session_state.sales_agent = sales_agent
    st.session_state.cnt = 0

# Access the agent and the counter
sales_agent = st.session_state.sales_agent
cnt = st.session_state.cnt

# Display chat history from session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": sales_agent.step()["output"]}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Your response:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    sales_agent.human_step(prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Increment the conversation counter
    st.session_state.cnt += 1
    cnt = st.session_state.cnt

    if cnt == 60:
        st.write("Maximum number of turns reached - ending the conversation.")
    else:
        response = sales_agent.step()["output"]
        # End conversation if the agent decides so
        with st.chat_message("assistant"):
            if "<END_OF_TURN>" in response:
                response = response.replace("<END_OF_TURN>", "")
            if "<END_OF_CALL>" in response:
                response = response.replace("<END_OF_CALL>", "")
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)