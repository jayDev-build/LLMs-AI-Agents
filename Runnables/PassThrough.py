from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": joke_explain_chain
})

chain = RunnableSequence(joke_gen_chain, parallel_chain)

# print(chain.invoke({'topic':"cricket"}))

chain.get_graph().print_ascii()
