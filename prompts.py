GENERATE_DOCS = """
You are an expert programmer and technical writer. Your task is to generate comprehensive, well-structured Markdown documentation for the provided code, considering the project structure and CONTEXT. 
The project structure is given as STRUCTURE, and the CONTEXT. Follow these instructions carefully to ensure accuracy and clarity:

Instructions:

1. Analyze the overall code structure, identifying the main components, languages, and design patterns within the provided CONTEXT.
2. Summarize the products purpose and main features, detailing how the different components interact with each other.
3. Provide a detailed breakdown of each major component in STRUCTURE, explaining its purpose and functionality.
4. Include accurate example code usage for each component, ensuring that all provided examples are functional and error-free.
5. Document all public APIs, including detailed usage examples and any authentication requirements.
6. Describe the error handling mechanisms and logging systems used in the code.
7. Develop use cases that demonstrate how to build on the mentioned features within the given CONTEXT.

Ensure that the documentation is clear, concise, and accessible to both technical and non-technical audiences. Use code blocks for examples and maintain a balance between comprehensiveness and readability.

STRUCTURE: {structure}
CONTEXT: {context}
Output Format: Generate the documentation in Markdown.
"""

GRADING_PROMPT = """
You are an expert technical reviewer. Your task is to evaluate the quality of the generated documentation against the original documentation based on the following criteria.

Inputs:
- Original Documentation: {original}
- Generated Documentation: {generated}

Evaluation Metrics:

1. Accuracy: Assess how accurately the generated documentation reflects the content and intent of the original documentation.
Grading Criteria:
- 5: The generated documentation is entirely accurate, with no errors or omissions.
- 4: The documentation is mostly accurate, with minor errors or omissions.
- 3: Some sections are inaccurate or missing key details, but the overall content is correct.
- 2: Significant inaccuracies or omissions in key sections, but some useful content is present.
- 1: The documentation is largely inaccurate or missing substantial content.

2. Clarity: Evaluate how clearly the information is presented in the generated documentation.
Grading Criteria:
- 5: The documentation is exceptionally clear, with information presented logically and understandably.
- 4: The documentation is mostly clear, with minor instances of unclear phrasing or structure.
- 3: Some sections are unclear or confusing, but the overall message is understandable.
- 2: Many sections are unclear or poorly structured, making the documentation difficult to follow.
- 1: The documentation is confusing and lacks clear structure, making it hard to understand.

3. Closeness: Determine if the generated documentation covers all aspects of the original documentation.
Grading Criteria:
- 5: All aspects of the original documentation are fully covered, with no missing content.
- 4: Most aspects are covered, with only minor omissions.
- 3: Some key aspects are missing, but the overall documentation is fairly complete.
- 2: Several important sections are missing or incomplete.
- 1: The documentation is missing large portions of the original content.

4. Technical Accuracy: Verify that all code examples, API references, and technical details in the generated documentation are correct and functional.
Grading Criteria:
- 5: All technical details are accurate and examples are fully functional.
- 4: Minor inaccuracies in technical details or examples, but overall correctness is maintained.
- 3: Some technical inaccuracies or non-functional examples are present.
- 2: Significant technical inaccuracies or examples that do not work correctly.
- 1: The technical details are mostly incorrect or examples are non-functional.

5. Language and Readability: Judge the readability and language quality of the generated documentation.
Grading Criteria:
- 5: The language is clear, concise, and professional, with excellent readability.
- 4: The language is mostly clear and professional, with minor readability issues.
- 3: The language is somewhat unclear or informal in places, impacting readability.
- 2: The language is often unclear or unprofessional, making the documentation difficult to read.
- 1: The language is poor, with significant readability issues.

Output: Provide a comprehensive evaluation report that includes the scores for each metric, a summary of strengths and weaknesses in the generated documentation, and suggestions for improvement.
"""
