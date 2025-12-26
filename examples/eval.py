from videosdk.agents import Evaluation, Turn, Metric, JudgeMetric
from videosdk.agents.videosdk_eval.providers import STT, LLM, TTS, LLMJudge, STTEvalConfig, LLMEvalConfig, TTSEvalConfig
import logging
from videosdk.agents import function_tool
import aiohttp
logging.basicConfig(level=logging.INFO)


@function_tool
async def get_weather(
    latitude: str,
    longitude: str,
):
        """Called when the user asks about the weather. This function will return the weather for
        the given location. When given a location, please estimate the latitude and longitude of the
        location and do not ask the user for them.

        Args:
            latitude: The latitude of the location
            longitude: The longitude of the location
        """
        print("###Getting weather for", latitude, longitude)
        # logger.info(f"getting weather for {latitude}, {longitude}")
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m"
        weather_data = {}
        try:
             async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        print("Weather data", data)
                        weather_data = {
                            "temperature": data["current"]["temperature_2m"],
                            "temperature_unit": "Celsius",
                        }
                    else:
                        print(f"Failed to get weather data, status code: {response.status}")
                        raise Exception(
                            f"Failed to get weather data, status code: {response.status}"
                        )
        except Exception as e:
             print(f"Exception in get_weather tool: {e}")
             raise e

        return weather_data

eval = Evaluation(
    name="basic-agent-eval",
    metrics=[
	    Metric.STT_LATENCY,
        Metric.LLM_LATENCY,
        Metric.TTS_LATENCY,
    ]
)



# senario 1
eval.add_turn(
    Turn(
        stt=STT.deepgram(
            STTEvalConfig(file_path="./sample.wav")
        ),
        llm=LLM.google(
            LLMEvalConfig(
                model="gemini-2.5-flash-lite",
                use_stt_output=False,
                mock_input="write one paragraph about Water and get weather of Delhi",
                tools=[get_weather]
            )
        ),
        tts=TTS.google(
            TTSEvalConfig(
                model="en-US-Standard-A",
                use_llm_output=False,
                mock_input="Today is Thursday 25th december merry christmas"
            )
        ),
        judge=LLMJudge.google(model="gemini-2.5-flash-lite",prompt="Can you evaluate the agent's response based on the following criteria: Is it relevant to the user input",checks=[JudgeMetric.REASONING,JudgeMetric.CONCLUSION])
    )
)


eval.add_turn(
    Turn(
         tts=TTS.google(
            TTSEvalConfig(
                model="en-US-Standard-A",
                use_llm_output=False,
                mock_input="In 5 days, it will be new year"
            )
        )
    )
)

results = eval.run()
results.save("./reports")
