from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Write short Notes from the text: {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Write 5 Questions / Answers from the following text: {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the given Notes and Question/Answers in a single document: notes -> {Notes} ans Quiz -> {Quiz}',
    input_variables=['Notes', "Quiz"]
)

parser = StrOutputParser()

model1 = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash"
)

model2 = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash",
)

parallel_chain = RunnableParallel({
    "Notes": prompt1 | model1 | parser,
    "Quiz": prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

res = final_chain.invoke({
    "text" : 
    """
    The Mahabharata is an ancient Indian epic poem that tells the story of a great dynastic war between two groups 
    of paternal cousins, the Pandavas and the Kauravas, over the throne of Hastinapura.The Royal LineageThe Kuru 
    Dynasty: The story begins in the kingdom of Hastinapura. King Shantanu's lineage leads to two princes:
      Dhritarashtra (who is blind) and Pandu.The Pandavas: Because Dhritarashtra was blind, Pandu initially 
      took the throne. Pandu’s five sons—Yudhishthira, Bhima, Arjuna, Nakula, and Sahadeva—are known for their 
      virtue and divine parentage.The Kauravas: Dhritarashtra’s eldest son, Duryodhana, leads his 99 brothers. 
      They grow greedy, jealous, and resentful of the Pandavas.The Conflict and ExileThe Rigged Game: Duryodhana 
      schemes to cheat the Pandavas out of their share of the kingdom. He invites Yudhishthira to a rigged game 
      of dice. Yudhishthira loses everything, including their shared wife, Draupadi, and the Pandavas are forced 
      into a 12-year exile plus one year in disguise.The Failed Peace: After completing their exile, the Pandavas 
      demand their kingdom back. Duryodhana refuses, setting the stage for total war.The Kurukshetra War and the 
      Bhagavad GitaThe 18-Day Battle: The two armies meet on the battlefield of Kurukshetra. Lord Krishna serves 
      as Arjuna's charioteer and guide.The Bhagavad Gita: Just before the battle begins, Arjuna suffers a crisis 
      of conscience about fighting his own kin. Krishna counsels him on dharma (duty and righteous action), forming 
      the famous philosophical text known as the Bhagavad Gita.Pandava Victory: The brutal 18-day war ends in the 
      total annihilation of the Kaurava army. The Pandavas win, but the victory comes at a devastating cost of human 
      life.Aftermath and AscensionThe Rule and Departure: Yudhishthira rules justly for many years. Eventually, 
      realizing the transient nature of earthly life, the brothers abdicate the throne and journey toward the 
      Himalayas to find spiritual liberation.
    """
})


print(res)

final_chain.get_graph().print_ascii()