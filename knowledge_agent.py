import os
from dotenv import load_dotenv
from langchain_cerebras import ChatCerebras

load_dotenv()
AGENTBEATS = os.getenv("AGENT_BEATS")
if not AGENTBEATS:
    raise ValueError("Api key not loaded")

llm = ChatCerebras(
    model="llama-3.3-70b",
    temperature=0.7,
    api_key=AGENTBEATS)