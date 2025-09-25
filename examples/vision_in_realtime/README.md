# VideoSDK Vision Agent with Google Gemini

This project demonstrates a vision-enabled AI agent built using the [VideoSDK Agents SDK for Python](https://github.com/videosdk-live/agents). The agent joins a VideoSDK meeting, uses Google's Gemini model to process the real-time video feed, and can verbally describe what it sees or answer questions about the visual content.

## Prerequisites

Before you begin, ensure you have the following:

- Python 3.12 or higher.  
- A [VideoSDK Account](https://app.videosdk.live/signup). You will need your **Auth Token**.  
- A [Google AI Studio](https://aistudio.google.com/apikey) with the "Vertex AI API" enabled. You will need your **Google API Key**.  

---

## 🚀 Getting Started

Follow these steps to set up and run the vision agent.

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the necessary library:

**`requirements.txt`**
```
videosdk-agents
videosdk-plugins-google  
av
videosdk-plugins-turn-detector  
videosdk-plugins-silero
python-dotenv
```

Now, install the requirements using pip:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

The application uses a `.env` file to manage secret keys.

First, create a `.env` file by copying the example:

```bash
cp .env.example .env
```

Now, open the `.env` file and add your credentials.

**`.env.example`**
```env
# VideoSDK Configuration
VIDEOSDK_AUTH_TOKEN=your_videosdk_auth_token_here

# Google API Configuration (for Gemini vision support)
GOOGLE_API_KEY=your_google_api_key_here
```

- `VIDEOSDK_AUTH_TOKEN`: Find this in your [VideoSDK Dashboard](https://app.videosdk.live/api-keys).  
- `GOOGLE_API_KEY`: Generate this from your [Google Cloud Console](https://console.cloud.google.com/apis/credentials) or [Google AI Studio](https://aistudio.google.com/apikey).  

### 5. Generate a Room ID

The agent needs a `roomId` to join a meeting. You can create a new room using the VideoSDK REST API.

Use your `VIDEOSDK_AUTH_TOKEN` to create a room. Here is an example using `curl`:

```bash
curl -X POST https://api.videosdk.live/v2/rooms \
  -H "Authorization: YOUR_VIDEOSDK_AUTH_TOKEN" \
  -H "Content-Type: application/json"
```

*(Replace `YOUR_VIDEOSDK_AUTH_TOKEN` with your actual token)*

```python
# In main.py

# ...
def make_context() -> JobContext:  
    room_options = RoomOptions(  
        room_id="<YOUR-NEW-ROOM-ID>",  # <--- PASTE YOUR NEW ROOM ID HERE
        name="Vision Test Agent",  
        playground=True,  
        vision=True,  
        recording=False  
    )  
# ...
```

For more details on the Create Room API, refer to the [VideoSDK documentation](https://docs.videosdk.live/api-reference/realtime-communication/create-room).

### 🚨 Important: A Video-Enabled Meeting UI is Required

The Vision Agent needs a video feed to analyze. The default agent playground UI is audio-only and **will not work** for this demo.

To test the agent, you must join the same meeting from a separate, video-enabled application.

1. **Get a Prebuilt Example:** Use any of the official VideoSDK quickstart repositories to quickly set up a video meeting UI. You can find them here:  
   - **[VideoSDK Quickstart Repositories](https://github.com/videosdk-live/quickstart)** (Available for JS, React, etc.)  

2. **Configure the UI:** Follow the instructions in the chosen quickstart repository to set up the project. You will need to configure it with the **same `roomId`** that you generated in the previous step.  

3. **Join the Meeting:** Start your quickstart application and join the meeting with your camera turned on.  

### 6. Run the Agent

Once your video meeting UI is ready and you have joined the room, you can start the Python agent.

```bash
python main.py
```

The agent will connect to the specified `roomId`. You will hear it introduce itself. You can then show it objects or scenes via your camera and ask it questions like "What do you see?" or "Describe the object in front of me."

*That's all, Happy coding!*
