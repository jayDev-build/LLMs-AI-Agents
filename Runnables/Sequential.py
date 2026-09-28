from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')

prompt1 = PromptTemplate(
    template='write a joke on the given topic {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the given joke {joke}',
    input_variables=['joke']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)
joke_explain_chain = RunnableSequence(prompt2, model, parser)

chain = RunnableSequence(joke_gen_chain, joke_explain_chain)

print(chain.invoke({'topic':"cricket"}))
