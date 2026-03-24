# Voice Blog Writer

A voice agent that writes full blog posts through a short conversation, then generates a titled markdown file using a sequential Gemini-powered LangGraph pipeline.

## Setup

### Environment variables

```
VIDEOSDK_AUTH_TOKEN=...
DEEPGRAM_API_KEY=...
GOOGLE_API_KEY=...
CARTESIA_API_KEY=...
```

### Install and run

```bash
pip install -r requirements.txt
python agent.py
```

## How it works

The graph asks 3 gathering questions before writing:
1. **Topic** — what to write about
2. **Audience** — who it's for (developers, beginners, etc.)
3. **Tone** — professional, casual, technical, etc.

Once all 3 are gathered:
```
coordinator → planner → write_sections (4 sequential Gemini calls) → compiler → synthesizer
```

Each section write is logged individually so you can watch the graph run step by step.
The blog is saved as `{title-slug}_{datetime}.md` in this folder.

`LangGraphLLM(output_node="synthesizer_node")` — only the final spoken line reaches TTS.
