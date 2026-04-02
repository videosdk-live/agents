import aiohttp
from videosdk.agents import Agent, AgentSession, Pipeline, function_tool, JobContext, RoomOptions, WorkerJob, ContextWindow
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.cartesia import CartesiaTTS
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

from dotenv import load_dotenv
load_dotenv(override=True)
pre_download_model()

@function_tool
async def get_weather(city: str) -> dict:
    """Get the current weather temperature for a given city.

    Args:
        city: The name of the city (e.g. "Dubai", "Mumbai", "New York")
    """
    # Map common cities to lat/lon
    city_coords = {
        "dubai": (25.2048, 55.2708),
        "mumbai": (19.0760, 72.8777),
        "new york": (40.7128, -74.0060),
        "london": (51.5074, -0.1278),
        "tokyo": (35.6762, 139.6503),
        "paris": (48.8566, 2.3522),
        "sydney": (-33.8688, 151.2093),
    }

    coords = city_coords.get(city.lower(), (25.2048, 55.2708))
    lat, lon = coords

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                temp = data["current"]["temperature_2m"]
                print(f"  [Tool 1] get_weather({city}) → {temp}°C")
                return {"city": city, "temperature": temp, "unit": "Celsius"}
            else:
                return {"city": city, "temperature": 25, "unit": "Celsius", "note": "fallback"}


@function_tool
async def get_clothing_advice(temperature: float) -> dict:
    """Get clothing recommendation based on the current temperature.
    Call this AFTER getting the weather temperature.

    Args:
        temperature: The temperature in Celsius
    """
    if temperature > 35:
        advice = "Very light breathable clothes, hat, and sunscreen. Stay hydrated."
    elif temperature > 25:
        advice = "Light clothes like t-shirt and shorts. Sunglasses recommended."
    elif temperature > 15:
        advice = "Light jacket or sweater with comfortable pants."
    elif temperature > 5:
        advice = "Warm coat, scarf, and layered clothing."
    else:
        advice = "Heavy winter coat, gloves, hat, and thermal layers."

    print(f"  [Tool 2] get_clothing_advice({temperature}°C) → {advice}")
    return {"temperature": temperature, "clothing_advice": advice}


@function_tool
async def get_activity_suggestion(temperature: float, clothing: str) -> dict:
    """Suggest an outdoor activity based on temperature and what the person is wearing.
    Call this AFTER getting both the weather AND the clothing advice.

    Args:
        temperature: The temperature in Celsius
        clothing: What the person is wearing (from get_clothing_advice)
    """
    if temperature > 35:
        activity = "Visit an indoor mall or aqua park to stay cool."
    elif temperature > 25:
        activity = "Go to the beach, have a picnic in the park, or try outdoor dining."
    elif temperature > 15:
        activity = "Go for a hike, visit a botanical garden, or explore the city on foot."
    elif temperature > 5:
        activity = "Visit a museum, try a cozy cafe, or go for a brisk walk in the park."
    else:
        activity = "Go ice skating, visit a winter market, or enjoy hot chocolate at a cafe."

    print(f"  [Tool 3] get_activity_suggestion({temperature}°C, '{clothing[:30]}...') → {activity}")
    return {
        "temperature": temperature,
        "clothing": clothing,
        "activity_suggestion": activity,
    }


class ToolChainingAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a helpful travel assistant. When a user asks what to do in a city:\n"
                "1. FIRST call get_weather to get the temperature\n"
                "2. THEN call get_clothing_advice with that temperature\n"
                "3. THEN call get_activity_suggestion with the temperature AND clothing advice\n"
                "4. Finally, combine all three results into a natural spoken response.\n\n"
                "You MUST call all three tools in sequence — do NOT skip any step.\n"
                "Keep your final response concise and conversational (2-3 sentences max)."
            ),
            tools=[get_weather, get_clothing_advice, get_activity_suggestion],
        )

    async def on_enter(self) -> None:
        await self.session.say(
            "Hi! I'm your travel assistant. Ask me what to do in any city "
            "and I'll check the weather, suggest what to wear, and recommend an activity!"
        )

    async def on_exit(self) -> None:
        pass



async def start_session(context: JobContext):
    agent = ToolChainingAgent()

    pipeline = Pipeline(
        stt=DeepgramSTT(),
        llm=GoogleLLM(),
        tts=CartesiaTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector(),

        # ── Context Window ──────────────────────────────────────────
        #
        # ContextWindow bundles all context management into one object:
        #   - max_tokens: Token budget for the entire conversation history.
        #     When exceeded, old turns are compressed then truncated.
        #     With 3 tools per city, each city adds ~200 tokens.
        #     4000 tokens fits ~5 city plans + conversation.
        #
        #   - max_context_items: Maximum number of items (messages + tool calls).
        #     Either limit can trigger compression/truncation.
        #
        #   - keep_recent_turns: Number of recent user-assistant exchanges
        #     kept verbatim. Everything older gets summarized by the LLM.
        #
        #   - summary_llm: Optional separate LLM for generating summaries.
        #     If not set, the agent's main LLM is used automatically.
        #     Example: summary_llm=OpenAILLM(model="gpt-4o-mini")
        
        # ── Tool Execution ──────────────────────────────────────────
        #
        # max_tool_calls_per_turn: Maximum number of tool calls allowed in a single
        # user turn. This is a safety limit to prevent infinite loops where
        # the LLM keeps requesting tools without ever producing a text response.
        #
        # How tool chaining works:
        #   1. User says "Plan for Dubai" → LLM returns get_weather(Dubai)
        #   2. Tool executes → result added to context → LLM called again
        #   3. LLM returns get_clothing_advice(22°C) → execute → call LLM again
        #   4. LLM returns get_activity_suggestion(22°C, "jacket") → execute → call LLM
        #   5. LLM returns text "Dubai is 22°C, wear a jacket, go hiking!" → spoken by TTS
        #   That's 3 tool calls + 1 text response = 4 rounds, well within the limit.
        #
        # For multi-city queries ("Plan for Dubai AND Mumbai"):
        #   Dubai: 3 tools + Mumbai: 3 tools = 6 tool calls minimum.
        #   Some LLMs may re-call tools redundantly, so 10 gives headroom.
        #

        # How parallel tool calls work:
        #   Some LLMs (Anthropic Claude, OpenAI GPT-4o) can return multiple
        #   tool calls in a single response. For example, Claude might return:
        #     [get_weather(London), get_clothing_advice(14°C), get_activity(14°C)]
        #   all at once. These are collected and executed in parallel using
        #   asyncio.gather, then all results are added to context before the
        #   next LLM call. This is faster than sequential execution.
        #   Google Gemini sends one tool call at a time (always sequential).
        context_window=ContextWindow(
            max_tokens=4000,
            max_context_items=20,
            keep_recent_turns=3,
            max_tool_calls_per_turn=10,
        ),
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)


def make_context() -> JobContext:
    room_options = RoomOptions(
        room_id="<room_id>",
        name="Tool Chaining Agent",
        playground=True,
    )
    return JobContext(room_options=room_options)


if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()
