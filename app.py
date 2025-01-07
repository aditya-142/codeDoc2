import subprocess
import sys
import streamlit as st
import os
import ast
import logging
import tempfile
import shutil
from git import Repo
from urllib.parse import urlparse
from typing import List, Optional, Tuple
from dataclasses import dataclass
# install_groq()
from groq import GroqClient
from prompts import GRADING_PROMPT
from agent import return_agent

MAX_DIRECTORY_DEPTH = 5
API_KEY = st.secrets['GROQ_API_KEY']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Code Product Documentation", layout="wide")

st.title("Code Product Documentation Generator")
st.markdown("Generate comprehensive documentation for your Python projects with ease. Support for both local directories and GitHub repositories.")

@dataclass
class FileInfo:
    file_path: str
    content: str
    module_docstring: Optional[str]
    functions: List[tuple]
    classes: List[tuple]

def is_github_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.netloc == "github.com" and len(parsed.path.strip("/").split("/")) >= 2
    except:
        return False

def clone_github_repo(url: str) -> Tuple[str, tempfile.TemporaryDirectory]:
    try:
        temp_dir = tempfile.TemporaryDirectory()
        Repo.clone_from(url, temp_dir.name)
        return temp_dir.name, temp_dir
    except Exception as e:
        logger.error(f"Error cloning repository: {str(e)}")
        raise Exception(f"Failed to clone repository: {str(e)}")

@st.cache_data
def get_python_files(directory: str, max_depth: int = MAX_DIRECTORY_DEPTH) -> List[str]:
    python_files = []
    for root, _, files in os.walk(directory):
        if root[len(directory):].count(os.sep) < max_depth:
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
    return python_files

@st.cache_data
def extract_info(file_path: str) -> Optional[FileInfo]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            tree = ast.parse(content)

        module_docstring = ast.get_docstring(tree)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        return FileInfo(
            file_path=file_path,
            content=content,
            module_docstring=module_docstring,
            functions=[(f.name, ast.get_docstring(f)) for f in functions],
            classes=[(c.name, ast.get_docstring(c)) for c in classes],
        )
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {str(e)}")
        return None

def generate_file_summary(file_info: FileInfo) -> str:
    return f"""
        File: {file_info.file_path}
        Module Docstring: {file_info.module_docstring}
        Functions: {file_info.functions}
        Classes: {file_info.classes}
    """

@st.cache_data
def generate_holistic_documentation(project_info: List[FileInfo], template: str) -> Optional[str]:
    file_summaries = "\n".join(generate_file_summary(file_info) for file_info in project_info)
    structure = ", ".join(file_info.file_path for file_info in project_info)

    prompt = f"""
    Generate comprehensive documentation for the following Python project:

    Project Structure:
    {structure}

    File Summaries:
    {file_summaries}

    Please provide a detailed overview of the project, following this template structure:
    {template}

    Ensure that you cover the project's purpose, main components, and how they interact.
    Include all the sections specified in the template, and add any additional relevant information.
    """

    try:
        client = GroqClient(api_key=API_KEY)
        completion = client.chat.completions.create(
            model="mixtral-8x7b",
            messages=[
                {"role": "system", "content": "You are an expert software documentation generator."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating holistic documentation: {str(e)}")
        return None

@st.cache_data
def evaluate_documentation(original_doc: str, generated_doc: str) -> Optional[str]:
    evaluate_prompt = GRADING_PROMPT.replace("{original}", original_doc)
    prompt = evaluate_prompt.replace("{generated}", generated_doc)

    agent = return_agent()
    response = agent(prompt)
    return response

def process_input_path(input_path: str) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
    temp_dir = None
    if is_github_url(input_path):
        try:
            directory_path, temp_dir = clone_github_repo(input_path)
        except Exception as e:
            st.error(f"Error processing GitHub repository: {str(e)}")
            return None, None
    else:
        directory_path = input_path
        if not os.path.exists(directory_path):
            st.error("Invalid directory path")
            return None, None
    
    return directory_path, temp_dir

def main():
    input_path = st.text_input(
        "Enter the directory path or GitHub repository URL:",
        help="You can enter either a local directory path or a GitHub repository URL.",
        key="input_path"
    )

    template = st.text_area(
        "Enter the documentation template structure (leave empty for default):",
        value="# Project Overview\n## Installation\n## Usage\n## API Reference\n## Contributing",
        help="Enter the desired structure for your documentation in Markdown format."
    )

    if input_path and st.button("Generate Documentation", key="generate_button"):
        with st.spinner("Processing input..."):
            directory_path, temp_dir = process_input_path(input_path)
            
            if directory_path:
                try:
                    file_paths = get_python_files(directory_path)
                    project_info = [extract_info(file) for file in file_paths if extract_info(file)]
                    documentation = generate_holistic_documentation(project_info, template)
                    st.session_state.documentation = documentation
                finally:
                    if temp_dir:
                        temp_dir.cleanup()

    if 'documentation' in st.session_state:
        st.markdown("## Generated Documentation")
        st.markdown(st.session_state.documentation)

        if st.button("Report", key="evaluate_button"):
            st.session_state.show_evaluation = True

    if st.session_state.get('show_evaluation', False):
        st.markdown("## Evaluation")
        eval_choice = st.selectbox("Choose original document source:", ["Upload", "Paste", "Attach"])

        original_document = None
        if eval_choice == "Upload":
            uploaded_file = st.file_uploader("Upload a .txt or .md file", type=["txt", "md"])
            if uploaded_file:
                original_document = uploaded_file.read().decode("utf-8")

        elif eval_choice == "Paste":
            original_document = st.text_area("Paste the reference document:")

        elif eval_choice == "Attach":
            file_path = st.text_input("Enter the path where the original document is:")
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    original_document = file.read()
            else:
                st.error("Invalid file path or file does not exist.")

        if original_document and st.button("Generate Report", key="generate_report_button"):
            with st.spinner("Evaluating... might take some time"):
                eval_report = evaluate_documentation(original_document, st.session_state.documentation)

            st.markdown("## Evaluation Report")
            st.markdown(eval_report)

if __name__ == "__main__":
    main()

"""def install_groq():
    """Checks if the 'groq' module is installed; if not, installs it."""
    try:
        import groq  # Try importing groq
    except ModuleNotFoundError:
        print("Module 'groq' not found. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "groq"])
        print("Installation complete. Restarting...")
        import groq  # Try importing again after installation """




