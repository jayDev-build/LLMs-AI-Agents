'''
This is Chat Model it return an AI message.
'''

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

res = model.invoke("What is the capital of India?")

print(res.content[0]["text"])