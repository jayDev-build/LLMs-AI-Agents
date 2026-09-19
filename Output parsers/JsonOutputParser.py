from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash")

parser = JsonOutputParser()

prompt = PromptTemplate(
    template=""" write a detailed report on topic {topic} \n{format_instructions}""",
    # input_variables=["topic"]
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template=""" Summarize the given detailed report into 5 pointers {report} \n{format_instructions}""",
    # input_variables=["topic"]
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

chain = prompt | model | prompt2 | model | parser
response = chain.invoke({"topic" : 'black hole'})

print(response)