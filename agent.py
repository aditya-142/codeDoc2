import os
import streamlit as st
from llama_index.llms.custom import CustomLLM
from llama_index.agent.openai import OpenAIAgent
from llama_index.tools.code_interpreter.base import CodeInterpreterToolSpec
from groq import Groq  

class GroqLLM(CustomLLM):
    """Custom LLM class for using Groq API with llama-index."""

    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768", temperature: float = 0):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        super().__init__()

    def complete(self, prompt: str) -> str:
        """Generate a response from Groq API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature
        )
        return response.choices[0].message.content.strip()

def return_agent():
    """AI agent using the Groq API"""
    key = st.secrets["GROQ_API_KEY"]  # Ensure this key is set in Streamlit secrets

    code_spec = CodeInterpreterToolSpec()
    tools = code_spec.to_tool_list()

    llm = GroqLLM(api_key=key, model="mixtral-8x7b-32768")  # Use Groq model

    agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True)
    return agent
