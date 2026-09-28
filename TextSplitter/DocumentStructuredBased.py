from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from dotenv import load_dotenv

load_dotenv()

markdown_sample = """# Payment Gateway Integration Guide

## Setup and Credentials
Before making API calls, generate your API key pair from the dashboard.
Store the secret key in an environment variable named PAYMENT_SECRET_KEY.

## Webhook Configuration
Webhooks notify your application when asynchronous payment events occur.

### Signature Verification
Every webhook header contains an X-Signature hash.
Verify this hash using HMAC-SHA256 to protect against spoofing attempts.

### Retry Logic
If your webhook endpoint returns a 5xx status code, the gateway retries delivery up to 5 times.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,
    chunk_size = 100,
    chunk_overlap = 20
)

res = splitter.split_text(markdown_sample)

i = 0
for chunk in res:
    print(i, chunk, end = '\n\n')
    i += 1