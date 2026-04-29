# Turn / Ack Mismatch on Interrupt — Fix Notes

## What the bug looked like

In production lead-qualification calls (see [transcripts_turnwise.md](./transcripts_turnwise.md)),
the agent appeared to "hear" the user agreeing to questions the user never
actually heard. Two reproducible patterns:

**Session 69f080e10a4955798f2d41dd, turns 15–16**

| # | Speaker | Text |
|---|---------|------|
| 15 | Agent | "I understand. So you're using it for freelancing, primarily for quotations and payment receipts. **My next question is, is your business registered under GST?**" |
| 16 | User  | "yeah yeah." |
| 17 | Agent | "Great. Do you only need invoicing and billing…" *(treats turn 16 as a "yes" to GST)* |

The user only heard the freelancing acknowledgement and was confirming
*that*. They never heard the GST question. Yet the next LLM turn behaved
as if they had.

**Session 69f090f01d6e1660b518c714, turns 10–11** — same pattern: agent's
GST follow-up gets a phantom "exactly हां" because the user was
acknowledging the "I'm an AI assistant" sentence earlier in the same
agent turn.

Net effect: the call summary recorded GST registration where none was
confirmed. Bad data; framework-level, not a prompt or example issue.

## Root cause

In [pipeline_orchestrator.py](../videosdk-agents/videosdk/agents/pipeline_orchestrator.py),
the orchestrator wrote `self._partial_response` into chat context
whenever a turn was interrupted. `_partial_response` accumulates LLM
chunks as they arrive — not what TTS has actually played out.

Because LLM streaming finishes much faster than TTS playback, by the time
the user interrupts mid-utterance, `_partial_response` already contains
the **entire** generated response, including text the user never heard.
On interrupt:

1. STT final fires while agent is `SPEAKING` → orchestrator calls
   `_interrupt_pipeline()`.
2. `_interrupt_pipeline()` writes `_partial_response` to chat_context as
   the assistant message, marked `interrupted=True`.
3. The `interrupted=True` flag exists in
   [chat_context.py](../videosdk-agents/videosdk/agents/llm/chat_context.py)
   but the format converters don't trim the post-interrupt portion — the
   whole sentence ships to the LLM on the next turn.

Net: chat context holds the unspoken question; user's "yeah" attaches to
it on the next call.

## What we changed

A playback-aware truncation: at interrupt time, replace `_partial_response`
with a best-effort estimate of the text actually played out to the
listener. Two tiers, both TTS-agnostic at the call site.

### Tier 1 — Universal (works with any TTS)

All TTS plugins push audio through one chokepoint:
`audio_track.add_new_bytes(chunk)` on
[CustomAudioStreamTrack](../videosdk-agents/videosdk/agents/room/output_stream.py).
That same audio track now exposes:

| Field | Meaning |
|---|---|
| `_cumulative_input_samples` | Total samples ever pushed in via `add_new_bytes`. |
| `_samples_played` | Samples popped from the playback buffer in `recv()` — only ticks on real frames, not silence or pause. |
| `_synthesis_start_played` / `_synthesis_start_pushed` | Per-synthesis baselines, anchored by `mark_synthesis_start()`. |

`snapshot_playback()` returns the deltas. From those, `SpeechGeneration`
computes `played / pushed` as a playback fraction, multiplies by
`len(full_transcript)` to get a character cutoff, and snaps back to the
last whitespace so we don't cut mid-word.

### Tier 2 — Precise (Cartesia, ElevenLabs, any TTS with `supports_word_timestamps`)

These plugins emit `word_spoken` events whose payload carries
`cumulative_text` — the text whose audio the TTS *intends* to start
playing now, gated by word-level timestamps. We already subscribed to
that event for metrics; now we also stash the latest `cumulative_text`
in `_tier2_spoken_transcript`, and `compute_spoken_transcript()` prefers
it over the proportional Tier 1 estimate.

### How the truncation flows through

1. `SpeechGeneration.synthesize()` resets `full_transcript`,
   `spoken_transcript`, `_tier2_spoken_transcript` and calls
   `audio_track.mark_synthesis_start()` once `audio_track` is resolved.
2. As text streams in, `full_transcript` accumulates. As audio pushes
   to the track, `_cumulative_input_samples` grows. As `recv()` pops
   real frames, `_samples_played` grows. As Cartesia/ElevenLabs report
   words, `_tier2_spoken_transcript` updates.
3. User interrupts → `_interrupt_pipeline()` → `speech_generation.interrupt()`:
   - **Before** clearing audio buffers, calls
     `compute_spoken_transcript()` and stores the result in
     `self.spoken_transcript`.
   - Then `audio_track.interrupt()` clears buffers as before.
4. Back in `_interrupt_pipeline()`, the truncated message body is
   chosen as `speech_generation.spoken_transcript` if non-empty,
   else `_partial_response` (legacy fallback for text-only / no-TTS
   pipelines). That text is used for both
   `metrics_collector.set_agent_response(...)` and the
   `chat_context.add_message(role=ASSISTANT, …, interrupted=True)`
   call.

## Files changed

| File | Change |
|---|---|
| [videosdk-agents/videosdk/agents/room/output_stream.py](../videosdk-agents/videosdk/agents/room/output_stream.py) | Added `_samples_played`, `_cumulative_input_samples`, `_synthesis_start_*` fields; `mark_synthesis_start()` and `snapshot_playback()` methods; counter increments in `add_new_bytes()` and the popped-frame branch of `recv()`. Patched `MixingCustomAudioStreamTrack` overrides that bypass `super`. |
| [videosdk-agents/videosdk/agents/speech_generation.py](../videosdk-agents/videosdk/agents/speech_generation.py) | Added `spoken_transcript` and `_tier2_spoken_transcript` fields; reset them per `synthesize()`; new `compute_spoken_transcript()` method; `_on_tts_word_spoken` now stores `cumulative_text`; `interrupt()` captures `spoken_transcript` *before* clearing the audio track; calls `mark_synthesis_start()` in both hooks and non-hooks paths. |
| [videosdk-agents/videosdk/agents/pipeline_orchestrator.py](../videosdk-agents/videosdk/agents/pipeline_orchestrator.py) | `_interrupt_pipeline()` prefers `speech_generation.spoken_transcript` over `_partial_response`. Defensive reset of `spoken_transcript` at the top of `_generate_and_synthesize` so stale state from a prior turn can't bleed into the next interrupt. |
| [examples/cascade_basic.py](../examples/cascade_basic.py) | Added `chat_context_monitor` task that prints each new chat-context entry with an `[INTERRUPTED]` tag — useful for verifying truncation end-to-end. |

No public API changes. No new config flags. Nothing to touch in TTS
plugin code — they all already push through `audio_track.add_new_bytes`.

## Verification

Ran [examples/cascade_basic.py](../examples/cascade_basic.py) with Cartesia
TTS (Tier 2 active) and interrupted four agent turns mid-utterance. Each
`[INTERRUPTED]` entry stopped at roughly the last word actually played:

| Full LLM reply | Stored in chat context (`[INTERRUPTED]`) |
|---|---|
| "I am a voice agent that can provide you with weather forecasts **and horoscopes.**" | "I am a voice agent that can provide you with weather forecasts and" |
| "I'm sorry, I cannot fulfill this request. I do not have access to information about the micro. **Is there anything else?**" | "I'm sorry, I cannot fulfill this request. I do not have access to information about the micro." |
| "The weather in Dubai is 37 degrees **Celsius.**" | "The weather in Dubai is 37 degrees" |
| "...your horoscope. Please provide me with **your zodiac sign.**" | "...your horoscope. Please provide me with" |

The bold portion in each row is text the user did not hear — and it
correctly does *not* appear in chat context. The next LLM turn can no
longer phantom-attribute the user's reply to an unspoken question.

### Other test scenarios to cover

- **Tier 1 path** (TTS without word timestamps, e.g. Sarvam): swap
  `CartesiaTTS()` → `SarvamAITTS()` in
  [cascade_basic.py](../examples/cascade_basic.py). Expect ±a few-word
  approximation, snapped to a word boundary, but no trailing unspoken
  text. ✅ behavior, ⏳ pending live confirmation.
- **No-interrupt regression**: let the agent finish a turn fully. The
  recorded assistant message must equal the full reply (no premature
  truncation). `spoken_transcript` is reset at the start of each
  `synthesize()` so it can't leak between turns.
- **Text-only / no-TTS pipeline regression**: with `speech_generation`
  set to `None`, the fallback to `_partial_response` runs unchanged.

## Limitations and out-of-scope

- **Backchannel handling**. "yeah yeah" still meets `interrupt_min_words=2`
  and triggers the interrupt path. That's a semantic detection problem,
  not a data-fidelity one. Mitigations at example level: raise
  `interrupt_min_words` to 3-4, raise `interrupt_min_duration`, or filter
  acknowledgement-shaped utterances inside the `stt_hook` while the
  agent is `SPEAKING` (regex along the lines of
  `^(yeah|right|ok|exactly|हाँ|हां)([\s,.]+\1)*[\s.,]*$`).

- **WebRTC peer-side jitter buffer**. `_samples_played` measures samples
  popped from our local audio track into WebRTC, which still has roughly
  100–300 ms of network/jitter buffer downstream. Some text might be
  reflected as "played" that wasn't actually heard. Acceptable —
  dramatically better than today's behavior (which records the entire
  LLM reply).

- **Format converters honoring `interrupted=True`**. We could further
  hint to the LLM by suffixing interrupted messages with something like
  "[…interrupted by user]" before sending. Considered nice-to-have; the
  truncation alone resolves the reported sessions.
