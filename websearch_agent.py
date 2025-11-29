import os
from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
from loguru import logger

# LLM + Tools
from langchain_cerebras import ChatCerebras
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import WeatherDataLoader
from typing import Optional
from serpapi import GoogleSearch
from langchain.tools import tool

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError("❌ SERPAPI_API_KEY is missing! Add it to your .env file.")

@tool
def search_flights(departure_id: str, arrival_id: str, outbound_date: str,
                   return_date: str = None, currency: str = "USD") -> str:
    """
    Search flights using SerpAPI and return clean text.
    """

    params = {
        "engine": "google_flights",
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": outbound_date,
        "currency": currency,
        "api_key": SERPAPI_API_KEY,
        "hl": "en",
        "gl": "us",
    }

    if return_date:
        params["return_date"] = return_date

    search = GoogleSearch(params)
    results = search.get_dict()

    flights = results.get("best_flights", [])

    if not flights:
        return "❌ No flights found."

    output = "✈️ **Best Flight Options**\n\n"

    for f in flights:
        seg = f["flights"][0]

        airline = seg.get("airline", "Unknown")
        flight_no = seg.get("flight_number", "N/A")
        dep_airport = seg["departure_airport"]["id"]
        arr_airport = seg["arrival_airport"]["id"]
        dep_time = seg["departure_airport"]["time"]
        arr_time = seg["arrival_airport"]["time"]
        duration = seg.get("duration", "N/A")
        price = f.get("price", "N/A")

        output += (
            f"🛫 **{airline}** ({flight_no})\n"
            f"• {dep_airport} → {arr_airport}\n"
            f"• ⏱ Duration: {duration} mins\n"
            f"• 🕒 {dep_time} → {arr_time}\n"
            f"• 💵 Price: {price} {currency}\n"
            f"---------------------------\n"
        )

    return output


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city using OpenWeatherMap."""
    try:
        loader = WeatherDataLoader.from_params(
            [city],
            openweathermap_api_key=os.getenv("OPENWEATHERMAP_API_KEY")
        )

        docs = loader.load()  # returns list of Document objects
        if not docs:
            return f"No weather data found for {city}."

        data = docs[0].page_content  # raw text

        return f"🌤 Weather in {city.title()}:\n{data}"

    except Exception as e:
        return f"Error fetching weather: {str(e)}"



# ==========================
# 1. LLM MODEL (CEREBRAS)
# ==========================
model = ChatCerebras(
    model="gpt-oss-120b",      # Low latency, strong model
    max_tokens=512,
    api_key=os.getenv("CEREBRAS_API_KEY"),
    temperature=0.7,
)


# ==========================
# 2. TAVILY SEARCH TOOL
# ==========================
tavily_tool = TavilySearch(
    max_results=2,
    topic="general",
    api_key=os.getenv("TAVILY_API_KEY")
)


# ==========================
# 3. YFINANCE TOOLS (DECORATOR)
# ==========================

@tool
def get_stock_price(ticker: str) -> str:
    """Get the latest stock price for a ticker symbol like AAPL or TSLA."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.history(period="1d")

        if info.empty:
            return f"No stock data found for '{ticker}'."

        price = info["Close"].iloc[-1]
        return f"📈 {ticker.upper()} Current Price: {price:.2f} USD"

    except Exception as e:
        return f"Error fetching stock price: {str(e)}"


@tool
def get_company_info(ticker: str) -> str:
    """Get company name, sector, and market cap for a given stock ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        name = info.get("longName", "Unknown")
        sector = info.get("sector", "Unknown")
        mc = info.get("marketCap", "N/A")

        return (
            f"🏢 {name}\n"
            f"• Sector: {sector}\n"
            f"• Market Cap: {mc}\n"
        )

    except Exception as e:
        return f"Error fetching company info: {str(e)}"


# ==========================
# 4. REGISTER ALL TOOLS
# ==========================
tools = [
    tavily_tool,
    get_stock_price,
    get_company_info,
    get_weather,
    search_flights
]


# ==========================
# 5. SYSTEM PROMPT
# ==========================
system_prompt = """
You are Samantha, a helpful AI agent.
Use TavilySearch for general or current information.
Use YFinance tools for stock prices and company financials.
Use Weather tool for current weather.
Use Flights tool for flight options with the help of SerpAPI.
Keep responses short, natural, and suitable for voice interaction.
"""


# ==========================
# 6. MEMORY
# ==========================
memory = InMemorySaver()


# ==========================
# 7. BUILD THE AGENT
# ==========================
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=system_prompt,
    checkpointer=memory,
)

# Config
agent_config = {
    "configurable": {
        "thread_id": "default_user"
    }
}
