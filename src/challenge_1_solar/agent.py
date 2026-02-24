from dotenv import load_dotenv

load_dotenv()

from datetime import date
from gradio import ChatMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
import json


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

## Yield Formula
When asked how yield is calculated, explain:
    Panel area (m²)    = farm_area_ha × 10,000 × GCR
    Installed kWp      = Panel area × panel_efficiency
    Daily yield (kWh)  = Installed kWp × PVOUT (kWh/kWp/day)

Where PVOUT (kWh/kWp/day) is predicted by XGBoost from weather features,
sourced from the Global Solar Atlas yearly raster for each Australian city.

## Unit Conversion
Always convert farm area to hectares before calling predict_solar_yield:
- 1 acre = 0.4047 ha, 1 km² = 100 ha, 1 m² = 0.0001 ha

## Weather Inference from Natural Language
When the user describes weather qualitatively, think about the physical correlations
between weather variables and infer ALL fields that are consistent with the description.
For example, a rainy day implies not just rainfall but also high humidity, heavy cloud
cover, low sunshine hours and likely no wind. Use your world knowledge of meteorology
to fill in as many WeatherInput fields as possible before calling the tool.

## Response Guidelines
- Always state which fallback strategy was used
- Always mention GCR and panel efficiency values used
- Present results in both kWh/day and MWh/day
- When weather is inferred from a qualitative description, note that the prediction
  is a directional estimate and not a precise forecast
- Always show the assumptions list returned by the tool
"""


def stream_agent(question: str, thread_id: str = "default"):
    """Yields ChatMessage objects for Gradio observability."""
    today = date.today().strftime("%B %d, %Y")
    prompt = SYSTEM_PROMPT + f"\n\nToday's date is {today}."

    agent = create_agent(
        llm,
        tools,
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": thread_id}}
    messages = []

    for event in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config,
        stream_mode="updates",
    ):
        for node, update in event.items():
            for msg in update.get("messages", []):

                # LLM made tool calls — show as pending accordion
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        messages.append(
                            ChatMessage(
                                role="assistant",
                                content=f"**Args:** `{tc['args']}`",
                                metadata={
                                    "title": f"🔧 Calling `{tc['name']}`",
                                    "status": "pending",
                                    "id": tc["id"],
                                },
                            )
                        )
                    yield messages

                # Tool returned a result — mark parent done, show result
                elif hasattr(msg, "tool_call_id"):
                    for m in messages:
                        if m.metadata and m.metadata.get("id") == msg.tool_call_id:
                            m.metadata["status"] = "done"

                    # Parse content and format as markdown
                    try:
                        data = json.loads(msg.content)
                        content = "\n".join(f"- **{k}:** {v}" for k, v in data.items())
                    except (json.JSONDecodeError, AttributeError):
                        content = str(msg.content)

                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=content,
                            metadata={
                                "title": "✅ Tool result",
                                "parent_id": msg.tool_call_id,
                            },
                        )
                    )
                    yield messages

                # Final text response — no tool calls
                elif (
                    hasattr(msg, "content")
                    and msg.content
                    and not getattr(msg, "tool_calls", None)
                ):
                    messages.append(
                        ChatMessage(
                            role="assistant",
                            content=msg.content,
                        )
                    )
                    yield messages
