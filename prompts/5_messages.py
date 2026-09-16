from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

mssgs = []
while True:
    user = input("User: ")
    if user.lower() == "exit":
        break
    mssgs.append(HumanMessage(content=user))
    res = model.invoke(mssgs)
    print("Assistant: ", res.content[0]["text"])
    mssgs.append(AIMessage(content=res.content))

print(mssgs)