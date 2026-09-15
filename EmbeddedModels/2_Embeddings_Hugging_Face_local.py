from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    "What is the meaning of life?",
    "What is the purpose of existence?",    
    "How do I bake a cake?",
]

model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

result = model.embed_documents(docs)

print(str(result))