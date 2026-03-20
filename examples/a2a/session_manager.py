# Realtime pipeline for customer agent, cascading LLM-only for specialist

from videosdk.agents import AgentSession, Pipeline
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
import os
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

def create_pipeline(agent_type: str) -> Pipeline:
    if agent_type == "customer":
        # Customer agent: Realtime model for voice interaction
        return Pipeline(
            llm=GeminiRealtime(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                config=GeminiLiveConfig(
                    voice="Leda",
                    response_modalities=["AUDIO"]
                )
            )
        )
    else:
        # Specialist agent: Text-only LLM for background processing
        return Pipeline(
            llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")),
        )


def create_session(agent, pipeline) -> AgentSession:
    return AgentSession(
        agent=agent,
        pipeline=pipeline,
    )


### Alternative: Both agents using Cascading pipeline

# from videosdk.agents import AgentSession, Pipeline
# from videosdk.plugins.google import GoogleSTT, GoogleLLM, GoogleTTS
# from videosdk.plugins.openai import OpenAILLM
# from videosdk.plugins.deepgram import DeepgramSTT
# from videosdk.plugins.silero import SileroVAD
# from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
# import os
# import logging 
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])


# pre_download_model()

# def create_pipeline(agent_type: str) -> Pipeline:
#     if agent_type == "customer":
#         return Pipeline(
#             # stt=GoogleSTT(model="latest_long"),
#             stt=DeepgramSTT(),
#             llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
#             tts=GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY")),
#             vad=SileroVAD(),
#             turn_detector=TurnDetector(),
#         )
#     else:
#         return Pipeline(
#             llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")),
#         )

# def create_session(agent, pipeline) -> AgentSession:
#     return AgentSession(agent=agent, pipeline=pipeline)
