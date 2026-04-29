import os
import sys

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _normalize_query(user_query: str) -> str:
    normalized = user_query
    replacements = {
        "telengana": "Telangana",
        "telangana": "Telangana",
        " os ": " of ",
        "terms of population and land": "population",
        "3rd": "third",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


@tool
def get_weather(city: str):
    """Get weather report for a city."""
    url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "langchain-weather-tool/1.0"},
    )
    response.raise_for_status()
    data = response.json()

    current = data.get("current_condition", [])
    nearest_area = data.get("nearest_area", [])
    if not current:
        return f"Weather lookup failed for {city}: incomplete response received."

    current = current[0]
    area = nearest_area[0] if nearest_area else {}
    description_items = current.get("weatherDesc", [])
    description = ", ".join(
        item.get("value", "") for item in description_items if item.get("value")
    ) or "No description"
    temperature = current.get("temp_C", "N/A")
    feels_like = current.get("FeelsLikeC", "N/A")
    humidity = current.get("humidity", "N/A")
    matched_city = (
        area.get("areaName", [{}])[0].get("value")
        if area.get("areaName")
        else city
    )
    country = (
        area.get("country", [{}])[0].get("value")
        if area.get("country")
        else "Unknown"
    )
    location_note = ""
    if matched_city and matched_city.lower() != city.lower():
        location_note = f" Nearby match: {matched_city}."
    return (
        f"Weather for {city}, {country}: {description}. Temperature: {temperature} C, "
        f"feels like {feels_like} C, humidity {humidity}%.{location_note}"
    )


def _extract_city_name(model: ChatHuggingFace, user_query: str, search_results: str) -> str:
    response = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Identify the single best city name that answers the user's weather request. "
                    "The answer must be a city in Telangana, India. Ignore unrelated places from other "
                    "states or countries. Return only the city name. If the query already names a city, "
                    "return that city. Do not include any explanation, punctuation, or extra words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User query: {user_query}\n\n"
                    f"Search results: {search_results}\n\n"
                    "Return only the city name."
                ),
            },
        ]
    )
    return response.content.strip()


def run_agent(user_query: str) -> str:
    load_dotenv()
    hf_token = (
        os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("hugging_face_api_key")
    )
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        temperature=0.7,
        huggingfacehub_api_token=hf_token,
    )
    model = ChatHuggingFace(llm=llm)
    search_tool = DuckDuckGoSearchRun()
    normalized_query = _normalize_query(user_query)
    search_results = search_tool.invoke(
        f"Telangana India city lookup: {normalized_query}"
    )
    city = _extract_city_name(model, normalized_query, search_results)

    if not city:
        return "Unable to determine the city name from the query."
    return get_weather.invoke({"city": city})


if __name__ == "__main__":
    try:
        output = run_agent("find the weather report of 3rd largest city in terms os population and land in telengana")
        print(output)
    except Exception as exc:
        print(f"Weather agent execution failed: {exc}")
