from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableBranch
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
prompt2 = PromptTemplate(
    template = 'Summarize the given report : {report}',
    input_variables=['report']
)


def count_words(text):
    return len(text.split())

report_gen = RunnableSequence(prompt1, model, parser)
branch_chain = RunnableBranch(
    (lambda x : len(x.split()) > 1000, RunnableSequence(prompt2, model, parser)),
    (RunnablePassthrough())
)

chain = RunnableSequence(report_gen, branch_chain)

res = chain.invoke({"topic" : "allegations on Gyanesh Kumar"})

print(res)

chain.get_graph().print_ascii()