from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')

tweet_prompt = PromptTemplate(
    template = "write a tweet for the given topic : {topic}",
    input_variables=['topic']
)
Linkedin_prompt = PromptTemplate(
    template = "write a LinkedIn for the given topic : {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

chain = RunnableParallel({
    "tweet": RunnableSequence(tweet_prompt, model, parser),
    "linkedin": RunnableSequence(Linkedin_prompt, model, parser)
    }
)

res = chain.invoke({"topic" : "CJP i.i cockrpoach janta party"})

print(res)
