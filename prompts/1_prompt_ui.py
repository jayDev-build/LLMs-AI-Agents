from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

st.header("LangChain Google Generative AI Chat")

input = st.text_input("Enter your message:")

if st.button("Summarize"):
    res = model.invoke(input)
    st.write(res.content[0]["text"])



