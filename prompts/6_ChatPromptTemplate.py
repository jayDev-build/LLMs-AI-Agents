from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import  AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

mssgs = ChatPromptTemplate([
    ("system", "You are a helpful assistant that provides information about research papers."),
    ("human", "Please summarize the research paper titled '{title}' ")
    ])

prompt = mssgs.invoke({"title": "Attention Is All You Need"})

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

res = model.invoke(prompt)

print(res.content[0]["text"])