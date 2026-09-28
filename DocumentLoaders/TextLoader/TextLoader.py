from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()
model = ChatGoogleGenerativeAI(model = 'gemini-3.1-flash-lite')

loader = TextLoader(file_path='DocumentLoaders\TextLoader\sample.txt', encoding='utf-8')

docs = loader.load()

prompt = PromptTemplate(
    template = 'Summarize the given text: {text}',
    input_variables=['text']
)

chain = prompt | model | parser

res = chain.invoke({'text':docs[0].page_content})

print(res)
# print(docs)