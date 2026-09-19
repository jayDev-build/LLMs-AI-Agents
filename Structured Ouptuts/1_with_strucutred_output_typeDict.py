from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

user_prompte = HumanMessage(
    content=
    """
Kousalya Wundavalli
5 out of 5 starsExcellent Senior-Friendly Laptop for Home Entertainment and Internet Access!
Reviewed in India on 31 August 2026
Colour: Gray
Verified Purchase
I recently purchased this laptop for domestic use, and it has been an absolute delight. If you are looking for a reliable, simple, and visually clear device for the household—especially for senior citizens—this laptop is a fantastic option.

Display and Entertainment (OTT Streaming):

The standout feature is its gorgeous 14.1-inch Full HD display. It offers fantastic, crisp picture quality with wide viewing angles. Watching online content on various OTT streaming applications is an immersive experience. The anti-glare screen is also incredibly gentle on the eyes, making long viewing sessions comfortable without causing eye strain.

Senior-Citizen Friendly:

The operating system is highly intuitive. Because it translates a familiar, smartphone-style application ecosystem into a clean desktop format, senior citizens can navigate it with ease. The app icons are straightforward, the internet browser is easy to manage, and they don't have to deal with the complex configuration menus often found on traditional operating systems.

Performance for Everyday Domestic Tasks:

Equipped with plenty of memory and a reliable processor, the everyday performance is remarkably fluid. Browsing the internet, logging onto social media, reading news, and streaming high-definition media happen without any stuttering or lag. Additionally, the battery life is strong enough to last through multiple movies before needing a charge.

Verdict:

This device perfectly balances simplicity and modern utility. It delivers exactly what it promises: seamless internet access, a beautiful entertainment display, and an uncomplicated user interface that anyone in the family can master. Highly recommended for domestic environments! """)

#schema
class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[int, "The sentiment of the review, where 1 is positive and -1 is negative"]
    key_themes: Annotated[list[str], "A list of key themes mentioned in the review"]
    pros: Annotated[Optional[list[str]], "A list of pros mentioned in the review, if any"]
    cons: Annotated[Optional[list[str]], "A list of cons mentioned in the review, if any"]
    name: Annotated[Optional[str], "The name of the reviewer, if mentioned"]


structured_model = model.with_structured_output(Review)

# response = structured_model.invoke("""Ordered this keyboard from Ant for my person use both for gaming and office work. First time I ordered, the product came out as defective, enter key was not working so I placed a replacement for it.

# The very next I got the replacement and gave the defective product to the delivery man.


# The new keyboard works well, It is stylish and unique. The build quality is quite good and all the button works well. The wire although could be braided but again in this price range, it is ok. It is light weight and economic.


# Over-all a great keyboard and I am satisfied with my purchase.""")

response = structured_model.invoke(user_prompte.content)

print(response)