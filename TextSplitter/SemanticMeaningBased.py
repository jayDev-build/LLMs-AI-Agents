from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv

load_dotenv()

semantic_sample = (
    "The central bank announced a 25 basis point reduction in the benchmark interest rate. "
    "Bond yields dropped immediately following the monetary policy announcement. "
    "Equities gained momentum as lower borrowing costs are expected to boost corporate earnings. "
    "Photosynthesis occurs within the chloroplasts of plant leaves. "
    "Chlorophyll absorbs light energy primarily from the blue and red wavelengths of the spectrum. "
    "Carbon dioxide and water are converted into glucose and oxygen through this metabolic pathway. "
    "Kubernetes manages containerized workloads across a distributed cluster of nodes. "
    "The control plane coordinates scheduling and handles worker node health checks. "
    "Pods represent the smallest deployable compute units in the cluster ecosystem."
)

embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

splitter = SemanticChunker(
    embeddings = embedder,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80
)

res = splitter.split_text(semantic_sample)

i = 0
for chunk in res:
    print(i, chunk, end = '\n\n')
    i += 1