from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

client = InferenceClient()

embedding = client.feature_extraction(
    "What is the meaning of life?",
    model="BAAI/bge-base-en-v1.5"
)

print(embedding)