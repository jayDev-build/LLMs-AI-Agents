from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1", 
    temperature= 0.2,
    task = "text-generation")

model = ChatHuggingFace(llm = llm)

res = model.invoke("Generate a story about 5 super power children born in 1800s who are trying to save the world from an alien invasion.")

print(res.content)