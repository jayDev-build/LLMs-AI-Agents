from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How do I bake a cake?",
]

model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2" , output_dimensionality=32)

result = model.embed_documents(docs)

print(str(result))