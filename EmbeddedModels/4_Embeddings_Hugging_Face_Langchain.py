from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-base-en-v1.5",
    task="feature-extraction"
    )

result = embeddings.embed_query(
    "What is the meaning of life?"
)

print(result)