from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyMuPDFLoader
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model = 'gemini-3.1-flash-lite')

loader = DirectoryLoader(
    path = r'DocumentLoaders\DirectoryLoader\SamplePdfs',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()
prompt = PromptTemplate(
    template = 'explain about computer Netwrking from the given text : {content}',
    input_variables=['content']
)

chain = prompt | model | parser

res = chain.invoke({'content':docs})
print(res)
chain.get_graph().print_ascii()