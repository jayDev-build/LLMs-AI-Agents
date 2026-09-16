from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    template="""
    Please summarize the research paper title "{title}" with following specifications:
    Explanation Style : "{style_input}"
    Explanation Length : "{length_input}"
    1. Mathematical Details:
        -include relvant mathematical equations if present in the paper
        -Explain the mathematical concepts using simple, intiutive code snippets where applicable
    2. Analogies:
        -use relatable analogies to simplfy complex concepts
    if certain information is not available in the paper, respond with "Information not available in the paper."
    instead of guessing.
    Ensure the summary is concise, accurate and aligned with the provided style and length specifications.
""",
input_variables=['title', 'style_input', 'length_input'],
validate_template=True
)

prompt_template.save("prompts/prompt_template2.json")