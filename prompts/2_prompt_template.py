from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
st.header("LangChain Google Generative AI Chat")


#creating template for prompt
# prompt_template = PromptTemplate(
#     template="""
#     Please summarize the research paper title "{title}" with following specifications:
#     Explanation Style : "{style_input}"
#     Explanation Length : "{length_input}"
#     1. Mathematical Details:
#         -include relvant mathematical equations if present in the paper
#         -Explain the mathematical concepts using simple, intiutive code snippets where applicable
#     2. Analogies:
#         -use relatable analogies to simplfy complex concepts
#     if certain information is not available in the paper, respond with "Information not available in the paper."
#     instead of guessing.
#     Ensure the summary is concise, accurate and aligned with the provided style and length specifications.
# """
# # input_variables=['title', 'style_input', 'length_input']
# )


#loading template from json file
prompt_template = load_prompt("prompts/prompt_template.json")

title = st.selectbox(
    "Select the research paper title:",
    ["Attention Is All You Need", "Deep Residual Learning for Image Recognition", "Mastering the Game of Go with Deep Neural Networks and Tree Search"]
)

style_input = st.selectbox(
    "Select the explanation style:",
    ["Begineer-friendly", "Detailed", "Technical", "Code-oriented", "Mathematical", "Analogy-based"]
)

length_input = st.selectbox(
    "Select the explanation length:",
    ["Brief", "Moderate", "Comprehensive"]
)


# if st.button("Summarize"):
#     prompt = prompt_template.invoke({
#         'title': title, 
#         'style_input': style_input, 
#         'length_input': length_input
#         })
#     res = model.invoke(prompt)
#     st.write(res.content[0]["text"])


#Now doing chain

if st.button("Summarize"):
    chain = prompt_template | model
    res = chain.invoke({
        'title': title, 
        'style_input': style_input, 
        'length_input': length_input
        })
    # res = model.invoke(prompt)
    st.write(res.content[0]["text"])

