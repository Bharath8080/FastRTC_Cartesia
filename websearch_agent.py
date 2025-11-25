from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from loguru import logger
import os
from dotenv import load_dotenv
load_dotenv()

# =======================
# 1. LLM MODEL (Groq) - Optimized for Low Latency
# =======================
model = ChatGroq(
    model="openai/gpt-oss-20b",  # Faster than gpt-oss-20b
    max_tokens=256,  # Reduced from 512 for faster responses
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
)

# =======================
# 2. TAVILY SEARCH TOOL - Optimized for Speed
# =======================
tavily_tool = TavilySearch(
    max_results=2,  # Reduced from 5 for faster responses
    topic="general",
    api_key=os.getenv("TAVILY_API_KEY")
)

tools = [tavily_tool]   # 🔥 Replace math tools with Tavily

# =======================
# 3. SYSTEM PROMPT - Optimized for Speed
# =======================
system_prompt = """
You are Samantha, a helpful assistant. Use Tavily for factual or current information.
Keep responses brief and conversational for audio playback.
"""

# =======================
# 4. MEMORY
# =======================
memory = InMemorySaver()

# =======================
# 5. BUILD THE AGENT
# =======================
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

agent_config = {
    "configurable": {
        "thread_id": "default_user"
    }
}
