import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# =========================================================
# 1. TOOLS
# =========================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
    }

    return movies.get(
        genre.lower(),
        "No movies found for that genre"
    )


@tool
def change__to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return temp_c * 1.8 + 32


@tool
def get_weather(city: str) -> str:
    """Get current temperature for a given city name."""

    # Geocoding API
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"

    geo_params = {
        "name": city,
        "count": 1
    }

    geo_response = requests.get(
        geo_url,
        params=geo_params,
        timeout=15
    ).json()

    if "results" not in geo_response:
        return f"Could not find weather data for city: {city}"

    location = geo_response["results"][0]

    latitude = location["latitude"]
    longitude = location["longitude"]

    # Weather API
    weather_url = "https://api.open-meteo.com/v1/forecast"

    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code",
        "temperature_unit": "celsius"
    }

    weather_response = requests.get(
        weather_url,
        params=weather_params,
        timeout=15
    ).json()

    current = weather_response["current"]

    result = {
        "resolved_city": location["name"],
        "temperature_celsius": current["temperature_2m"],
        "weather_code": current["weather_code"]
    }

    return json.dumps(result)


# List of tools
tools = [
    get_weather,
    search_movies,
    change__to_f
]


# =========================================================
# 2. GEMINI API KEY
# =========================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# =========================================================
# 3. INITIALIZE MODEL
# =========================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# =========================================================
# 4. CREATE AGENT
# =========================================================

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to "
        "Indian weather and Indian cinema. "

        "You can answer questions about weather in Indian cities "
        "and Indian movies. "

        "For any other topic, question, role, or general knowledge "
        "outside Indian weather and Indian cinema, you must say exactly: "

        "'I am not authorized to answer questions outside of "
        "Indian weather and cinema.' "

        "Never show internal thinking or reasoning. "
        "Return only the final answer to the user."
    )
)


# =========================================================
# 5. INPUT MODEL
# =========================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the agent"
    )


# =========================================================
# 6. FORMAT INPUT
# =========================================================

def format_for_agent(x) -> dict:

    if isinstance(x, dict):
        user_input = x["input"]
    else:
        user_input = x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# =========================================================
# 7. EXTRACT ONLY FINAL ANSWER
# =========================================================

def extract_text_response(agent_output) -> str:

    # If output is already a string
    if isinstance(agent_output, str):
        return agent_output

    # Make sure output is a dictionary
    if not isinstance(agent_output, dict):
        return str(agent_output)

    # Get messages
    messages = agent_output.get("messages")

    # Check nested dictionaries
    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict):

                if "messages" in value:
                    messages = value["messages"]
                    break

    if not messages:
        return str(agent_output)

    # Find the final AI message
    for message in reversed(messages):

        content = getattr(
            message,
            "content",
            None
        )

        if content is None:
            continue

        # Normal string response
        if isinstance(content, str):

            if content.strip():
                return content.strip()

        # Gemini content blocks
        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):

                    # Ignore thinking/reasoning
                    if item.get("type") == "thinking":
                        continue

                    # Extract text
                    if item.get("type") == "text":

                        text = item.get("text", "")

                        if text:
                            text_parts.append(text)

                    elif "text" in item:

                        text = item.get("text", "")

                        if text:
                            text_parts.append(text)

                elif isinstance(item, str):

                    text_parts.append(item)

            if text_parts:
                return "".join(text_parts).strip()

    return "No final response was generated."


# =========================================================
# 8. CREATE LANGCHAIN CHAIN
# =========================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# =========================================================
# 9. FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    version="1.0.0"
)


# =========================================================
# 10. LANGSERVE ROUTE
# =========================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# =========================================================
# 11. HEALTH CHECK
# =========================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "message": "Indian Weather and Cinema Agent is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# 12. START SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
