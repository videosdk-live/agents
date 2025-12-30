import logging
import aiohttp
from videosdk.agents import Evaluation, EvalTurn, EvalMetric, LLMAsJudgeMetric, LLMAsJudge, STTEvalConfig, LLMEvalConfig, TTSEvalConfig, STTComponent, LLMComponent, TTSComponent,function_tool
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
    include_context=False,
    metrics=[
	    EvalMetric.STT_LATENCY,
        EvalMetric.LLM_LATENCY,
        EvalMetric.TTS_LATENCY,
        EvalMetric.END_TO_END_LATENCY
    ],
    output_dir="./reports"
)



# senario 1
eval.add_turn(
    EvalTurn(
        stt=STTComponent.deepgramv2(
            STTEvalConfig(file_path="./sample.wav") 
        ),
        llm=LLMComponent.google(
            LLMEvalConfig(
                model="gemini-2.5-flash-lite",
                use_stt_output=False, 
                mock_input="write one paragraph about Water and get weather of Delhi",
                tools=[get_weather]
            )
        ),
        tts=TTSComponent.google(
            TTSEvalConfig(
                model="en-US-Standard-A",
                use_llm_output=False,
                mock_input="Peter Piper picked a peck of pickled peppers"  
            )
        ),
        judge=LLMAsJudge.google(model="gemini-2.5-flash-lite",prompt="Can you evaluate the agent's response based on the following criteria: Is it relevant to the user input",checks=[LLMAsJudgeMetric.REASONING,LLMAsJudgeMetric.SCORE])
    )
)
eval.add_turn(
    EvalTurn(
        stt=STTComponent.deepgram(
            STTEvalConfig(file_path="./Sports.wav") 
        ),
        llm=LLMComponent.google(
            LLMEvalConfig(
                model="gemini-2.5-flash-lite",
                use_stt_output=True, 
            )
        ),
        tts=TTSComponent.google(
            TTSEvalConfig(
                model="en-US-Standard-A",
                use_llm_output=True
            )
        ),
        judge=LLMAsJudge.google(model="gemini-2.5-flash-lite",prompt="Can you evaluate the agent's response based on the following criteria: Is it relevant to the user input",checks=[LLMAsJudgeMetric.REASONING,LLMAsJudgeMetric.SCORE])
    )
)


eval.add_turn(
    EvalTurn(
        stt=STTComponent.deepgram(
            STTEvalConfig(file_path="./Sports.wav") 
        )
    )
)

eval.add_turn(
    EvalTurn(
        llm=LLMComponent.google(
            LLMEvalConfig(
                model="gemini-2.5-flash-lite",
                use_stt_output=False, 
                mock_input="write one paragraph about trees",
            )
        ),
    )
)

eval.add_turn(
    EvalTurn(
         tts=TTSComponent.google(
            TTSEvalConfig(
                model="en-US-Standard-A",
                use_llm_output=False,
                mock_input="A big black bug bit a big black bear, made the big black bear bleed blood."
            )
        )
    )
)

results = eval.run()
results.save()
