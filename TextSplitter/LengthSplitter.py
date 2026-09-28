from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import TokenTextSplitter
from dotenv import load_dotenv

load_dotenv()

length_sample = (
    "2026-09-28 10:00:01 INFO [AuthService] User login initiated: user_id=1042.\n"
    "2026-09-28 10:00:02 INFO [AuthService] Token generated: session_id=sess_9823.\n"
    "2026-09-28 10:00:03 WARN [DBService] Connection pool latency high: 142ms.\n"
    "2026-09-28 10:00:04 INFO [PaymentService] Webhook received: charge.success.\n"
    "2026-09-28 10:00:05 ERROR [PaymentService] Duplicate idempotency key: ord_1928.\n"
    "2026-09-28 10:00:06 INFO [AlertService] PagerDuty alert triggered for on-call."
)

# model = ChatGoogleGenerativeAI(model = 'gemini-3.5-flash-lite')

splitter = TokenTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

res = splitter.split_text(length_sample)

print(res)