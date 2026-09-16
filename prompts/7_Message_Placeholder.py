from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import  AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

mssgs = ChatPromptTemplate([
    ("system", "You are a helpful customer support assistant "),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", '{query}' )
    ])

chat_history = []

with open("prompts/chat_History.txt", "r") as f:
    chat_history.extend(f.readlines())

prompt = mssgs.invoke({"query": "where is the refund for my order", 
                       "chat_history": chat_history})

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

res = model.invoke(prompt)

print(res.content[0]["text"])