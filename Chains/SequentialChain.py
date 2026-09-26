from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write a concise 3-paragraph overview on the topic: {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Write 5 pointer summary of the given report {report}',
    input_variables=['report']
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    timeout=120.0  # seconds
)
chain = prompt1 | model | parser | prompt2 | model | parser

chain.get_graph().print_ascii()

res = chain.invoke({'topic' : "Mahabharat"})

print(res)