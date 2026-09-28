from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

text_structure_sample = """# Microservices Architecture Overview

Microservices structure an application as a collection of loosely coupled, independently deployable services. Each service handles a discrete business domain and communicates over lightweight protocols like gRPC or REST.

Key benefits of microservices:
- Independent scaling based on service-specific load.
- Fault isolation to prevent single points of failure.
- Polyglot persistence, enabling teams to pick the right database.

However, microservices introduce substantial operational complexity. Distributed tracing, network latency, eventual consistency, and complex CI/CD pipelines require mature platform engineering before adoption.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 50,
    chunk_overlap = 10
)

res = splitter.split_text(text_structure_sample)

i = 0
for chunk in res:
    print(i, chunk)
    i += 1
# print(res)