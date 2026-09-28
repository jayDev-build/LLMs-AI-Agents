from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model = 'gemini-3.1-flash-lite')

loader = PyPDFLoader(file_path='DocumentLoaders\PyPDFLoader\pypdfloader_test.pdf')

docs = loader.load()

prompt = PromptTemplate(
    template = 'what is the techstack mentioned here: {content}',
    input_variables=['content']
)

chain = prompt | model | parser

res = chain.invoke({"content":docs})

print(res)
