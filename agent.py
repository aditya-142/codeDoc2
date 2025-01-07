import os
import streamlit as st
from llama_index.llms.groq import Groq
from llama_index.agent.groq import GroqAgent
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec

def return_agent():
    key = st.secrets['GROQ_API_KEY']

    code_spec = CodeInterpreterToolSpec()
    tools = code_spec.to_tool_list()
    agent = GroqAgent.from_tools(
        tools,
        llm=Groq(temperature=0, model="mixtral-8x7b"),
        api_key=key,
        verbose=True
    )
    return agent
