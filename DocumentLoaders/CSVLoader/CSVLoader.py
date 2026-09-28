from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, PyMuPDFLoader
from dotenv import load_dotenv


load_dotenv()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model = 'gemini-3.1-flash-lite')

loader = CSVLoader(file_path='DocumentLoaders\CSVLoader\csvloader_test_data.csv')

docs = loader.load()

prompt = PromptTemplate(
    template = 'find the Net sales from the given data: {data}',
    input_variables=['data']
)

chain = prompt | model | parser

res = chain.invoke({'data':docs})

print(res)