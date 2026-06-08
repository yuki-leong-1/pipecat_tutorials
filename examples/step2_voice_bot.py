"""
Step 2 — Full Local Voice Agent
================================
Requires: Deepgram + OpenAI + Cartesia API keys (all three).
Once running, speak directly into your microphone to talk with the agent.
Press Ctrl+C to stop.

What you'll learn:
    1. The full STT → LLM → TTS pipeline
    2. VAD (Voice Activity Detection) — detecting when you've finished speaking
    3. LLMContext & Aggregators — how to manage conversation history
    4. Bidirectional audio with LocalAudioTransport
    5. LLMRunFrame — proactively triggering the LLM

Install dependencies:
    pip install "pipecat-ai[local,deepgram,openai,cartesia,silero]" python-dotenv loguru

Configuration:
    Copy .env.example to .env and fill in the three API keys
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# VAD: detects when the user starts and stops speaking
from pipecat.audio.vad.silero import SileroVADAnalyzer

# Frames: "containers" for data
from pipecat.frames.frames import LLMRunFrame

# Pipeline: a chain of processors
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask

# Context: stores conversation history
from pipecat.processors.aggregators.llm_context import LLMContext

# Aggregators: accumulate transcriptions/responses into context
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

# AlwaysUserMuteStrategy: built-in Pipecat strategy that mutes user input while the bot is speaking
# Official docs: https://docs.pipecat.ai/api-reference/server/utilities/turn-management/user-mute-strategies
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

# AI Services
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService

# Local audio transport
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


#   ┌──────────────────────────────────┬──────────────────────────────────────────────┬──────────────────────────────────────────┐
#   │            Scenario              │                   Approach                   │                  Effect                  │
#   ├──────────────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────┤
#   │ LocalAudio + speakers (current)  │ BotSpeakingUserMuteStrategy                  │ No echo, but can't interrupt while bot   │
#   │                                  │                                              │ is speaking                              │
#   ├──────────────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────┤
#   │ LocalAudio + headphones          │ Remove mute strategy, allow_interruptions=True│ No echo + interruptions supported        │
#   ├──────────────────────────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────┤
#   │ Web transport (step5)            │ Daily / WebRTC, browser built-in AEC         │ No echo + interruptions supported        │
#   └──────────────────────────────────┴──────────────────────────────────────────────┴──────────────────────────────────────────┘


async def main():
    # ═══════════════════════════════════════════════════════════════════════
    # 1. TRANSPORT
    # Uses the computer's microphone (input) and speakers (output)
    # ═══════════════════════════════════════════════════════════════════════
    # To list device indices:
    #   uv run python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
    # Set input_device_index to your physical microphone's index (None = use system default)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # device 1 = Microphone Array on SoundWire (physical microphone)
            # Avoid device 0 (Sound Mapper) — it may map to a loopback device
            # Avoid device 12/16 (Input SoundWire Speaker) — that is a loopback for system audio
            input_device_index=1,
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 2. SERVICES (three AI services)
    # ═══════════════════════════════════════════════════════════════════════

    # STT: converts your speech to text
    # Input: AudioRawFrame  →  Output: TranscriptionFrame (transcribed text)
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    # TTS: converts the LLM's text reply into speech
    # Input: TextFrame  →  Output: AudioRawFrame (audio)
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM") 
    )

    # LLM: processes the conversation and generates a reply
    # Input: LLMContextFrame  →  Output: TextFrame (reply text)
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",  # cheap and fast, good for learning
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Keep your responses short and conversational (1-3 sentences). "
                "Do not use bullet points, markdown, or emojis."
            ),
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 3. CONTEXT & AGGREGATORS
    # ═══════════════════════════════════════════════════════════════════════

    # LLMContext: stores conversation history (a messages list in OpenAI API format)
    context = LLMContext()

    # LLMContextAggregatorPair returns two processors:
    #
    # user_aggregator:
    #   - placed after STT
    #   - listens for TranscriptionFrames (transcriptions)
    #   - uses SileroVADAnalyzer to detect when you've finished speaking
    #   - once you're done, appends the full sentence to context and emits an LLMContextFrame to trigger the LLM
    #
    # assistant_aggregator:
    #   - placed after transport.output()
    #   - collects all TextFrames produced by the LLM; once the turn ends, appends them to context
    #   - this way the LLM knows what it said in previous turns
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            # Echo prevention: mutes user input while the bot is speaking; resumes 0.5 s after the bot stops
            # To support true barge-in → switch to Daily/WebRTC transport (browser built-in AEC)
            # Mutes the microphone while the bot speaks to prevent echo from triggering VAD and interrupting itself
            # Side effect: the user also cannot barge-in while the bot is speaking
            # True barge-in support → use Web transport (Daily/WebRTC, browser built-in AEC)
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PIPELINE (core data flow)
    # ═══════════════════════════════════════════════════════════════════════
    #
    # Full data flow:
    #
    #  [microphone] ──AudioRawFrame──► transport.input()
    #                                          │
    #                                   AudioRawFrame
    #                                          │
    #                                         stt  ◄── Deepgram real-time transcription
    #                                          │
    #                                TranscriptionFrame
    #                                          │
    #                                 user_aggregator  ◄── waits for you to finish + accumulates context
    #                                          │
    #                                LLMContextFrame (full conversation history)
    #                                          │
    #                                         llm  ◄── OpenAI processes
    #                                          │
    #                                    TextFrame (reply text, streamed)
    #                                          │
    #                                         tts  ◄── Cartesia synthesis
    #                                          │
    #                                  AudioRawFrame (audio)
    #                                          │
    #                                transport.output()
    #                                          │
    #                                  [speaker playback] ──► assistant_aggregator
    #                                                               │
    #                                                    recorded into context for next LLM turn
    pipeline = Pipeline([
        transport.input(),       # receive audio from microphone
        stt,                     # speech → text
        user_aggregator,         # accumulate what the user said
        llm,                     # generate a reply
        tts,                     # text → speech
        transport.output(),      # play to speakers
        assistant_aggregator,    # record what the agent said
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # 5. PIPELINE TASK
    # ═══════════════════════════════════════════════════════════════════════
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 6. LET THE AGENT SPEAK FIRST ON STARTUP
    # ═══════════════════════════════════════════════════════════════════════
    # Add a system message to context instructing the LLM to introduce itself first
    context.add_message({
        "role": "developer",
        "content": "Please greet the user and briefly introduce yourself. Keep it under 2 sentences."
    })
    # LLMRunFrame: immediately triggers the LLM to process context (without waiting for the user to speak first)
    await task.queue_frames([LLMRunFrame()])

    # ═══════════════════════════════════════════════════════════════════════
    # 7. RUN
    # ═══════════════════════════════════════════════════════════════════════
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("\n🎤 Voice agent is running! Speak into your microphone.")
    print("   Press Ctrl+C to stop.\n")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
