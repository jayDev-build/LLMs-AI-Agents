'''
This is LLM it return a text message. It is a simple wrapper around the Google Generative AI API.
'''

from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2)

res = llm.invoke("Write a poem about the beauty of nature.")

print(res)