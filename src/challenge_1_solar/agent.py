from dotenv import load_dotenv

load_dotenv()

from datetime import date
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent

from .tools import predict_solar_yield, get_city_solar_stats

tools = [predict_solar_yield, get_city_solar_stats]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
checkpointer = MemorySaver()

SYSTEM_PROMPT = """You are a solar yield prediction assistant for Australian solar farms.
You help users estimate daily solar energy output based on location, farm size, and weather conditions.

You have two tools:
- predict_solar_yield: predicts daily energy yield for a solar farm
- get_city_solar_stats: returns reference solar statistics and available cities

## Reasoning Order

1. If you are unsure about a city name → call get_city_solar_stats first to confirm
2. If the user provides specific weather conditions → infer as many WeatherInput fields
   as possible from their description using the field descriptions as guides, then call
   predict_solar_yield with those values. Leave unknown fields as None.
3. If the user specifies a date or relative date (e.g. 'tomorrow', 'next Monday') →
   resolve it to YYYY-MM-DD using today's date, then pass it to predict_solar_yield.
   The tool will map it to the equivalent 2010 historical observation.
4. If no date or weather is provided → call predict_solar_yield without date or weather.
   The tool will use the city's annual historical average as the baseline.
5. If no city is provided → ask the user for a city before proceeding.

## Unit Conversion
Always convert farm area to hectares before calling predict_solar_yield:
- 1 acre = 0.4047 ha, 1 km² = 100 ha, 1 m² = 0.0001 ha

## Response Guidelines
- Always state which fallback strategy was used
- Always mention key assumptions (GCR=0.35, panel efficiency=18%)
- Present results in both kWh/day and MWh/day
- When weather is inferred from a description, note the prediction is a directional estimate
"""


def run_agent(question: str, thread_id: str = "default") -> str:
    today = date.today().strftime("%B %d, %Y")
    prompt = SYSTEM_PROMPT + f"\n\nToday's date is {today}."

    agent = create_agent(
        llm,
        tools,
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    return result["messages"][-1].content
