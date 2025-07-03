
# Real time pipeline main ----- And cacading for text ....

from videosdk.agents import AgentSession, CascadingPipeline, RealTimePipeline, ConversationFlow
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
import os

class MyConversationFlow(ConversationFlow):
    async def on_turn_start(self, transcript: str) -> None:
        pass

    async def on_turn_end(self) -> None:
        pass

def create_pipeline(agent_type: str):
    if agent_type == "customer":
        return RealTimePipeline(
            model=GeminiRealtime(
                model="gemini-2.0-flash-live-001",
                config=GeminiLiveConfig(
                    voice="Leda", 
                    response_modalities=["AUDIO"]
                )
            )
        )
    else:
        return CascadingPipeline(
            llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")),
        )

def create_session(agent, pipeline) -> AgentSession:
    return AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=MyConversationFlow(agent=agent),
    )

### RealTime Pipline Example 

# from videosdk.agents import AgentSession, RealTimePipeline
# from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
# from openai.types.beta.realtime.session import TurnDetection
# from videosdk.plugins.google import GeminiRealtime, GeminiLiveConfig
# from typing import Dict

# def create_pipeline(agent_type: str) -> RealTimePipeline:
#     if agent_type == "customer":
#         model = GeminiRealtime(
#         model="gemini-2.0-flash-live-001",
#         config=GeminiLiveConfig(
#             voice="Leda",
#             response_modalities=["AUDIO"]
#         )
#     )
#     else:
#         model = GeminiRealtime(
#             model="gemini-2.0-flash-live-001",
#             config=GeminiLiveConfig(response_modalities=["TEXT"])
#         )

#     return RealTimePipeline(model=model)


# def create_session(agent, pipeline) -> AgentSession:
#     return AgentSession(agent=agent, pipeline=pipeline)



### Cascading Pipeline  Example 

# from videosdk.agents import AgentSession, CascadingPipeline, ConversationFlow
# from videosdk.plugins.google import GoogleSTT, GoogleLLM, GoogleTTS
# from videosdk.plugins.openai import OpenAILLM
# from videosdk.plugins.silero import SileroVAD
# from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
# import os

# pre_download_model()

# class MyConversationFlow(ConversationFlow):
#     async def on_turn_start(self, transcript: str) -> None:
#         pass

#     async def on_turn_end(self) -> None:
#         pass

# def create_pipeline(agent_type: str) -> CascadingPipeline:
#     if agent_type == "customer":
#         return CascadingPipeline(
#             stt=GoogleSTT( model="latest_long"),
#             llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
#             tts=GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY")),
#             vad=SileroVAD(),
#             turn_detector=TurnDetector(),
#         )
#     else:
#         return CascadingPipeline(
#             llm=OpenAILLM(api_key=os.getenv("OPENAI_API_KEY")),
#         )


# def create_session(agent, pipeline) -> AgentSession:
#     return AgentSession(
#         agent=agent,
#         pipeline=pipeline,
#         conversation_flow=MyConversationFlow(agent=agent),
#     )



### Cascading pipeline main ----- And realtime pipeline for text ....

# from videosdk.agents import AgentSession, CascadingPipeline, RealTimePipeline, ConversationFlow
# from videosdk.plugins.google import GoogleSTT, GoogleLLM, GoogleTTS, GeminiRealtime, GeminiLiveConfig
# from videosdk.plugins.openai import OpenAIRealtime, OpenAIRealtimeConfig
# from videosdk.plugins.silero import SileroVAD
# from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
# import os

# pre_download_model()

# class MyConversationFlow(ConversationFlow):
#     async def on_turn_start(self, transcript: str) -> None:
#         pass

#     async def on_turn_end(self) -> None:
#         pass

# def create_pipeline(agent_type: str):
#     if agent_type == "customer":
#         return CascadingPipeline(
#             stt=GoogleSTT(model="latest_long"),
#             llm=GoogleLLM(api_key=os.getenv("GOOGLE_API_KEY")),
#             tts=GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY")),
#             vad=SileroVAD(),
#             turn_detector=TurnDetector(),
#         )
#     else:
#         return RealTimePipeline(
#             model=OpenAIRealtime(
#             model="gpt-4o-realtime-preview",
#             config=OpenAIRealtimeConfig(
#                 modalities=["text"],
#                 tool_choice="auto"
#             )
#     )
#         )

# def create_session(agent, pipeline) -> AgentSession:
#     return AgentSession(
#         agent=agent,
#         pipeline=pipeline,
#         conversation_flow=MyConversationFlow(agent=agent),
#     )
