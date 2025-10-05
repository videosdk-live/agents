import asyncio  
import os  
from videosdk.agents import Agent, AgentSession, CascadingPipeline, function_tool    
from videosdk.agents import ChatRole, ImageContent, JobContext, RoomOptions, WorkerJob, ConversationFlow    
from videosdk.plugins.openai import OpenAILLM, OpenAISTT, OpenAITTS  
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector
from videosdk.plugins.elevenlabs import ElevenLabsTTS  

import base64  
import mimetypes

import logging

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DocumentAnalysisAgent(Agent):    
    def __init__(self):    
        super().__init__(    
            instructions="You are a document analysis assistant. You can analyze images and answer questions about them. You can answer me about what you see in the document, the text content, or any details visible in the image. Always respond to user queries and be helpful.",
            tools=[self.analyze_document]
        )  
          
        # Convert local file to data URI  
        self.test_images = [    
            self._convert_file_to_data_uri("pan-sample.jpeg")  
        ]  

    def _convert_file_to_data_uri(self, file_path: str) -> str:  
        """Convert a local file to a data URI."""  
        try:  
            with open(file_path, 'rb') as file:  
                file_data = file.read()  
              
            # Get MIME type  
            mime_type, _ = mimetypes.guess_type(file_path)  
            if mime_type is None:  
                mime_type = 'image/jpeg'  # Default for images  
              
            # Encode to base64  
            base64_data = base64.b64encode(file_data).decode('utf-8')  
              
            # Create data URI  
            return f"data:{mime_type};base64,{base64_data}"  
        except FileNotFoundError:  
            print(f"Error: File {file_path} not found")  
            return ""  
        except Exception as e:  
            print(f"Error converting file to data URI: {e}")  
            return ""
    
    @function_tool    
    async def analyze_document(self, image_url: str) -> str:    
        """Analyze a document image and provide detailed insights."""    
        self.chat_context.add_message(    
            role=ChatRole.USER,    
            content=[ImageContent(    
                image=image_url,    
                inference_detail="high"    
            )]    
        )    
        return f"Document analysis added for: {image_url}"  
    
    async def on_enter(self):        
        await self.session.say("Hello! I'm analyzing the given image...")  
          
        for i, image_url in enumerate(self.test_images):    
            await self.session.say(f"Analyzing image {i+1}...")    
              
            # Add image with analysis request directly to chat context  
            self.chat_context.add_message(        
                role=ChatRole.USER,        
                content=[    
                    "Please analyze this image in detail. Tell me what information you can see, including any text content, personal details, and document structure:",  
                    ImageContent(        
                        image=image_url,        
                        inference_detail="high"        
                    )    
                ]        
            )  
              
            await self.session.say("Image added for analysis. You can now ask me questions about it!")
        
        await self.session.say("I've analyzed the given image. You can now ask me questions about what you see in the image, the text content, or any details visible in the image!")
    async def on_exit(self):  
        await self.session.say("Goodbye! Thanks for using the image analysis service.")  
    
async def start_image_analysis_session(ctx: JobContext):    
    logger.info("Starting image analysis session")
    agent = DocumentAnalysisAgent()
    conversation_flow = ConversationFlow(agent)
    
    # Configure components for cascading pipeline  
    logger.info("Setting up pipeline components")
    pipeline = CascadingPipeline(  
        stt=DeepgramSTT(),
        llm=OpenAILLM(model="gpt-4o"),  # Updated model with vision support  
        tts=ElevenLabsTTS(),
        vad=SileroVAD(),
        turn_detector=TurnDetector()
    )    
        
    logger.info("Creating agent session")
    session = AgentSession(    
        agent=agent,    
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )    
    
    async def cleanup_session():
        pass
    
    ctx.add_shutdown_callback(cleanup_session)
        
    try:  
        logger.info("Connecting to room")
        await ctx.connect()
        print("Waiting for participant...")
        await ctx.room.wait_for_participant()
        print("Participant joined")
        logger.info("Starting agent session")
        await session.start()  
        logger.info("Agent session started, waiting for events")
        # Keep running until interrupted  
        await asyncio.Event().wait()  
    except KeyboardInterrupt:  
        print("\nShutting down gracefully...")  
    finally:  
        logger.info("Closing session and shutting down")
        await session.close()  
        await ctx.shutdown()  
    
def make_context():    
    room_options = RoomOptions(
        room_id="your-room-id",
        # room_id="eddq-bfbp-v3t1", 
        name="Image Analysis Agent", 
        playground=True
    )
    
    return JobContext(    
        room_options=room_options
    )    
    
if __name__ == "__main__":    
    job = WorkerJob(entrypoint=start_image_analysis_session, jobctx=make_context)    
    job.start()