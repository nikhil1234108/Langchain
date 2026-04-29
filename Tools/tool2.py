import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
import requests
from dotenv import load_dotenv

load_dotenv()

for proxy_var in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(proxy_var, None)

HF_TOKEN = (
    os.getenv("HUGGINGFACEHUB_API_TOKEN")
    or os.getenv("HF_TOKEN")
    or os.getenv("hugging_face_api_key")
)
EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")

if not HF_TOKEN:
    raise ValueError(
        "Missing Hugging Face token. Set HUGGINGFACEHUB_API_TOKEN, HF_TOKEN, "
        "or hugging_face_api_key in your .env file."
    )

if not EXCHANGE_RATE_API_KEY:
    raise ValueError("Missing EXCHANGE_RATE_API_KEY in your .env file.")

CURRENCY_ALIASES = {
    "USD": "USD",
    "US DOLLAR": "USD",
    "US DOLLARS": "USD",
    "UNITED STATES DOLLAR": "USD",
    "UNITED STATES DOLLARS": "USD",
    "INR": "INR",
    "INDIAN RUPEE": "INR",
    "INDIAN RUPEES": "INR",
    "RUPEE": "INR",
    "RUPEES": "INR",
    "CNY": "CNY",
    "RMB": "CNY",
    "YUAN": "CNY",
    "CHINESE YUAN": "CNY",
    "RENMINBI": "CNY",
}


def normalize_currency_code(currency: str) -> str:
    normalized = currency.strip().upper()
    return CURRENCY_ALIASES.get(normalized, normalized)

@tool
def get_currency_exchange_rate(base_currency:str, target_currency:str) -> float:
    """Get the exchange rate between two currencies using ISO codes or common names."""
    base_currency = normalize_currency_code(base_currency)
    target_currency = normalize_currency_code(target_currency)
    url = (
        f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}/pair/"
        f"{base_currency}/{target_currency}"
    )
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    return data["conversion_rate"]

@tool
def rate_conversion(base_currency_amount:int, conversion_rate:float) -> float:
    """Convert a currency amount to the target currency."""
    return base_currency_amount * conversion_rate

conversion_rate = get_currency_exchange_rate.invoke({'base_currency':'USD', 'target_currency':'INR'})
print(conversion_rate)

conversion_amount = rate_conversion.invoke({'base_currency_amount':100, 'conversion_rate':conversion_rate})
print(conversion_amount)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    provider="together",
    task="conversational",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
    temperature=0.1,
)
model = ChatHuggingFace(llm=llm)

llm_with_tools = model.bind_tools([get_currency_exchange_rate, rate_conversion])
query = HumanMessage(content="Convert 1000 Indian rupees to Chinese currency.")
messages = [query]
last_conversion_result = None

while True:
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        if response.content:
            print(response.content)
        elif last_conversion_result is not None:
            print(f"Converted amount: {last_conversion_result:.4f}")
        break

    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_currency_exchange_rate":
            tool_output = get_currency_exchange_rate.invoke(tool_call["args"])
        elif tool_call["name"] == "rate_conversion":
            tool_output = rate_conversion.invoke(tool_call["args"])
            last_conversion_result = tool_output
        else:
            continue

        messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
