from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "summary": {
        "description": "A brief summary of the review",
        "title": "Summary",
        "type": "string"
        },
        "sentiment": {
        "description": "The sentiment of the review, where 1 is positive and -1 is negative",
        "title": "Sentiment",
        "type": "integer"
        },
        "key_themes": {
        "description": "A list of key themes mentioned in the review",
        "items": {
            "type": "string"
        },
        "title": "Key Themes",
        "type": "array"
        },
        "pros": {
        "anyOf": [
            {
            "items": {
                "type": "string"
            },
            "type": "array"
            },
            {
            "type": "null"
            }
        ],
        "default": None,
        "description": "A list of pros mentioned in the review, if any",
        "title": "Pros"
        },
        "cons": {
        "anyOf": [
            {
            "items": {
                "type": "string"
            },
            "type": "array"
            },
            {
            "type": "null"
            }
        ],
        "default": None,
        "description": "A list of cons mentioned in the review, if any",
        "title": "Cons"
        },
        "name": {
        "anyOf": [
            {
            "type": "string"
            },
            {
            "type": "null"
            }
        ],
        "default": None,
        "description": "The name of the reviewer, if mentioned",
        "title": "Name"
        }
    },
    "required": [
        "summary",
        "sentiment",
        "key_themes"
    ],
}


model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

structured_model = model.with_structured_output(json_schema)

user_prompts = """
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

This device perfectly balances simplicity and modern utility. It delivers exactly what it promises: seamless internet access, a beautiful entertainment display, and an uncomplicated user interface that anyone in the family can master. Highly recommended for domestic environments! 

"""
response = structured_model.invoke(user_prompts)

print(response)