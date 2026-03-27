# Slack Voice Assistant

A voice agent that controls Slack by voice — post to channels.

## Setup

### 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app
2. Under **OAuth & Permissions**, add these **Bot Token Scopes**:
   - `chat:write` — post messages to channels

3. Install the app to your workspace and copy the **Bot User OAuth Token**
4. Invite the bot to any channels you want it to post to (`/invite @YourBot`)

### 2. Environment variables

```
VIDEOSDK_AUTH_TOKEN=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
CARTESIA_API_KEY=...
SLACK_BOT_TOKEN=xoxb-...
```

### 3. Install and run

```bash
pip install -r requirements.txt
python agent.py
```