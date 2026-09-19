from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate 
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash")

class Person(BaseModel):
    name:str = Field(description="Name of the person")
    age:int = Field(description="Age of the person", ge=0)
    city:str = Field(description="City person resided in")
    story:Optional[str] = Field(description="Story about the character")



parser = PydanticOutputParser(pydantic_object = Person)

prompt = PromptTemplate(
    template=""" Generate Name, Age and City of fictional place {place} Person \n{format_instructions}""",
    # input_variables=["topic"]
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template=""" Write the summarized story of the given fictional character {character} \n{format_instructions}""",
    # input_variables=["topic"]
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

chain = prompt | model | prompt2 | model | parser
response = chain.invoke({"place" : 'India'})

print(response)




