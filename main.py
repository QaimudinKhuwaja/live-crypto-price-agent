import chainlit as cl
import requests
import os
from dotenv import load_dotenv
from agents import Runner, Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool

# 🔐 Load API key
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.") 

# 🌐 Gemini Model Config
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# 🛠️ Tool: Show top 10 crypto prices
@function_tool
def show_top_prices(dummy: str = "") -> str:
    """Returns top 10 cryptocurrency prices from Binance"""
    url = "https://api.binance.com/api/v3/ticker/price"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        top_10 = data[:10]
        result = "📊 Top 10 Cryptocurrency Prices:\n\n"
        for coin in top_10:
            result += f"- {coin['symbol']}: ${coin['price']}\n"
        return result

    except requests.exceptions.RequestException as e:
        return f"❌ An error occurred: {str(e)}"

# 🛠️ Tool: Show specific coin price
@function_tool
def show_specific_coin_price(symbol: str) -> str:
    """Returns price of a specific coin like BTCUSDT from Binance"""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return f"🔎 Current price of {symbol.upper()}: ${data['price']}"
        else:
            return f"❌ Coin {symbol.upper()} not found. Please check the symbol."

    except requests.exceptions.RequestException as e:
        return f"❌ An error occurred: {str(e)}"

# 🤖 Crypto Agent
crypto_agent = Agent(
    name="Crypto Agent",
    instructions="""
You are a cryptocurrency expert AI assistant. 
Your job is to respond to user queries about live crypto prices.
Use tools to:
- Show top 10 crypto prices when the user says things like "top", "show top coins", etc.
- Show price of a specific coin like BTCUSDT when user asks for it.
Always be clear, concise, and avoid unnecessary information.
""",
    tools=[show_top_prices, show_specific_coin_price]
)

# 💬 Handle messages
@cl.on_message
async def handle_message(message: cl.Message):
    result = Runner.run_sync(
        starting_agent=crypto_agent,
        input=message.content,
        run_config=config
    )
    await cl.Message(content=result.final_output).send()