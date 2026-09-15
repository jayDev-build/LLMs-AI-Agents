from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

docs = [
    "Virat Kohli is indian cricketer known for his agressive batting and leadership",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing abilities",
    "Sachin Tendulkar, also knows as the 'God of Cricket', holds many batting records ",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries ",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers",
]

query = "tell me about Virat Kohli"

embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2" , output_dimensionality=32)

docs_embeddings = embedder.embed_documents(docs)
query_embedding = embedder.embed_query(query)

similarities = cosine_similarity([query_embedding], docs_embeddings)[0]
similarities = list(enumerate(similarities))
# print(similarities)
sorted_similarities = sorted(similarities, key=lambda x: x[1])
# print(sorted_similarities)

index, score = sorted_similarities[-1]

print(docs[index])
print ("Similarity Score: ", score)