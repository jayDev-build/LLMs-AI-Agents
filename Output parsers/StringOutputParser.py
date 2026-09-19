from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash")

parser = StrOutputParser()

prompt = PromptTemplate(
    template=""" write a detailed report on topic {topic} """
    # input_variables=["topic"]
    # partial_variables={'format_instruction' : parser.get_format_instructions()}
)

chain = prompt | model | parser
response = chain.invoke({"topic" : 'black hole'})

print(response)

