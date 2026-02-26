from dotenv import load_dotenv

load_dotenv()

from datetime import date
from gradio import ChatMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
import json

from .tools import predict_solar_yield, get_city_weather_stats

tools = [predict_solar_yield, get_city_weather_stats]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
checkpointer = MemorySaver()

SYSTEM_PROMPT = """You are a solar yield prediction assistant for Australian solar farms.
You help users estimate daily solar energy output based on location, farm size, and weather conditions.

You have two tools:
- get_city_weather_stats: fetches weather baseline for a city (yearly, monthly, or specific date)
- predict_solar_yield: predicts daily energy yield given explicit weather conditions

## Reasoning Order

1. If no city is provided → ask the user for a city before proceeding.

2. Determine if get_city_weather_stats needs to be called:
   - City changed → call it
   - Period changed (different date or month) → call it
   - No weather context in conversation yet → call it
   - Only farm size, GCR, efficiency, or qualitative weather changed → skip it

3. When calling get_city_weather_stats:
   - Date provided (e.g. 'July 15', 'tomorrow') → resolve to YYYY-MM-DD, pass as date=
   - Month provided (e.g. 'in July', 'next month') → pass as month=
   - No date or month → call with city only (returns yearly average)

4. If the user describes weather qualitatively (e.g. 'hot and clear', 'stormy'):
   - Use the baseline as the foundation and override fields consistent with the description
   - Use meteorological reasoning: a rainy day implies high humidity, heavy cloud,
     low sunshine, likely rainfall > 0. A clear hot day implies low cloud, high
     sunshine, low humidity, high max temp.
   - Override as many fields as the description supports.

5. Call predict_solar_yield with the WeatherInput (baseline + any overrides).

## Yield Formula
When asked how yield is calculated, explain:
    Panel area (m²)    = farm_area_ha × 10,000 × GCR
    Installed kWp      = Panel area × panel_efficiency
    Daily yield (kWh)  = Installed kWp × PVOUT (kWh/kWp/day)

Where PVOUT (kWh/kWp/day) is predicted by XGBoost from weather features.

## Unit Conversion
Always convert farm area to hectares before calling predict_solar_yield:
- 1 acre = 0.4047 ha, 1 km² = 100 ha, 1 m² = 0.0001 ha

## Response Guidelines
- Always state the weather baseline source (yearly average, monthly average, or specific date)
- Always mention GCR and panel efficiency values used
- Present results in both kWh/day and MWh/day
- When weather is inferred from a qualitative description, note that the prediction
  is a directional estimate and not a precise forecast
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

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            for msg in update.get("messages", []):

                # Tool call — pending accordion
                if getattr(msg, "tool_calls", None):
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

                # Tool result — close parent, nest result
                elif getattr(msg, "tool_call_id", None):
                    for m in messages:
                        if m.metadata and m.metadata.get("id") == msg.tool_call_id:
                            m.metadata["status"] = "done"
                    try:
                        content = "\n".join(
                            f"- **{k}:** {v}"
                            for k, v in json.loads(msg.content).items()
                        )
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

                # Final response
                elif getattr(msg, "content", None) and not getattr(
                    msg, "tool_calls", None
                ):
                    messages.append(ChatMessage(role="assistant", content=msg.content))

                yield messages
