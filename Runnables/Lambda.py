from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'write a summary on topic : {topic}',
    input_variables=['topic']
)

def count_words(text):
    return len(text.split())

# chain = RunnableSequence(prompt1, model, parser, RunnableLambda(count_words))
chain = RunnableSequence(prompt1, model, parser, lambda x : len(x.split()))

res = chain.invoke({"topic" : "allegations on Gyanesh Kumar"})

print(res)