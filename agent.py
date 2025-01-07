import os
import streamlit as st
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
from groq import Groq  # Import Groq client

def groq_chat_completion(prompt: str, api_key: str, model: str = "mixtral-8x7b-32768") -> str:
    """Call Groq API"""
    client = Groq(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

def return_agent():
    """Creates an AI agent using the Groq API instead of OpenAI."""
    key = st.secrets["GROQ_API_KEY"]  # Ensure this key is set in Streamlit secrets

    code_spec = CodeInterpreterToolSpec()
    tools = code_spec.to_tool_list()

    # Define a simple wrapper function for the agent to use Groq
    def llm(prompt: str) -> str:
        return groq_chat_completion(prompt, key)

    agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)
    return agent
