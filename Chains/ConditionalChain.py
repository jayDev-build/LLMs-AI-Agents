from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal, Optional

load_dotenv()

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field("Map the sentiment of the given review")
    feedback : Optional[str] = Field("Feedback given by user")

model = ChatGoogleGenerativeAI(
    model = 'gemini-3.5-flash'
)

parser = PydanticOutputParser(pydantic_object = Feedback)
str_parser = StrOutputParser()

# strucutred_model = model.with_structured_output(Feedback)

prompt1 = PromptTemplate(
    template='Analayze and return the sentiment for given feedback: {text} in the following instructions\n {format_instructions}',
    input_variables=['text'],
    partial_variables={'format_instructions' : parser.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='Write a 2 line appropriate response to user thanking him/her for positive reply : {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write a 2 line appropriate response to user apologising for his/her bad experience : {feedback}',
    input_variables=['feedback']
)

classifier_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', prompt2 | model | str_parser),
    (lambda x: x.sentiment == 'Negative', prompt3| model | str_parser),
    RunnableLambda(lambda x : "Could not find sentiment")
)

classifier = classifier_chain | branch_chain

# res = classifier.invoke({'text':"It was a Good phone the camera quality is great "})

# print(res)

classifier.get_graph().print_ascii()
branch_chain.get_graph().print_ascii()