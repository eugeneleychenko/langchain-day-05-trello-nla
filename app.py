import streamlit as st
import os
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents import AgentType
from langchain.utilities.zapier import ZapierNLAWrapper

# Running dotenv
load_dotenv(find_dotenv())

zapier_nla_api_key=os.getenv("ZAPIER_NLA_API_KEY")
openai_api_key=os.getenv("openai_api_key")

llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k")
zapier = ZapierNLAWrapper(zapier_nla_api_key=zapier_nla_api_key)
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
agent = initialize_agent(
    toolkit.get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

st.title('NLA Trello test')

input_text = st.text_input('Enter your command:')
if st.button('Run'):
    result = agent.run(input_text)
    st.write(result)