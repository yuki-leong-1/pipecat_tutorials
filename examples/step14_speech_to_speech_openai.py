"""
Step 14 — Speech-to-Speech: OpenAI Realtime (LocalAudio version)
=================================================================
Traditional pipeline: mic → Deepgram STT → OpenAI LLM → ElevenLabs TTS → speaker
                           ↑                                               ↑
                   separate STT service                          separate TTS service
                   E2E latency ~800ms

Speech-to-Speech: mic → OpenAI Realtime → speaker
                                ↑
                     one service handles STT + LLM + TTS simultaneously
                     E2E latency ~300ms

What you will learn:
    1. OpenAIRealtimeLLMService — one service replaces three (STT + LLM + TTS)
    2. SessionProperties — configure the Realtime session (voice, VAD, transcription, etc.)
    3. SemanticTurnDetection — smarter semantic turn detection than Silero VAD
    4. InputAudioNoiseReduction — built-in noise reduction (far_field = speaker scenario)
    5. Pipeline structure change: no longer need Deepgram / ElevenLabs
    6. universal LLMContext + LLMContextAggregatorPair (same as the traditional pipeline)

Pipeline comparison:
    Traditional: transport.input() → stt → user_agg → llm → tts → transport.output() → asst_agg
    S2S:         transport.input() → user_agg → [OpenAIRealtime] → transport.output() → asst_agg
                                                   (STT+LLM+TTS all handled internally)

Your Twilio version (production-grade, with WebSocket server):
    C:\\Users\\Yuki.Leong\\github\\twilio

Installation:
    uv add "pipecat-ai[local,openai,silero]"
    (no deepgram or elevenlabs needed)

Required API key: OPENAI_API_KEY (requires Realtime API access: gpt-4o-realtime-preview)
Note: headphones are recommended, or enable InputAudioNoiseReduction to reduce echo
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

# ── OpenAI Realtime imports ───────────────────────────────────────────────
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    AudioOutput,
    InputAudioNoiseReduction,
    InputAudioTranscription,
    SemanticTurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=1,
        )
    )

    # ── SessionProperties: configure the Realtime session ────────────────
    # All behaviour is configured here: voice selection, VAD strategy, noise reduction, transcription model, etc.
    session_properties = SessionProperties(
        instructions=(
            "You are a helpful voice assistant. "
            "Keep responses short and conversational. "
            "No markdown or bullet points."
        ),
        output_modalities=["audio"],  # output audio only (no TTS needed)
        audio=AudioConfiguration(
            input=AudioInput(
                # built-in transcription: converts what the user says to text (optional, useful for debugging)
                transcription=InputAudioTranscription(
                    model="gpt-4o-transcribe"
                ),
                # noise reduction: far_field is suited for speaker scenarios (mic is far from the speaker)
                noise_reduction=InputAudioNoiseReduction(type="far_field"),
                # semantic turn detection: smarter than Silero VAD, understands semantic boundaries
                # eagerness='low' = wait for the user to finish a complete sentence before responding (reduces false interruptions)
                turn_detection=SemanticTurnDetection(
                    eagerness="low",
                    interrupt_response=True,  # allow the user to interrupt the bot
                ),
            ),
            output=AudioOutput(
                voice="alloy",  # shimmer / echo / onyx / nova / fable / alloy
            ),
        ),
    )

    # ── OpenAI Realtime LLM Service ───────────────────────────────────────
    # Note: this single service replaces the STT + LLM + TTS trio from the traditional pipeline
    llm = OpenAIRealtimeLLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAIRealtimeLLMService.Settings(
            model="gpt-4o-realtime-preview",
            session_properties=session_properties,
        ),
    )

    # ── Context and Aggregators ───────────────────────────────────────────
    # Identical to the traditional pipeline! One of the advantages of the universal LLMContext
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    # ── Pipeline ──────────────────────────────────────────────────────────
    # Note: no STT or TTS! OpenAI Realtime handles all audio internally
    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),    # handle user context (VAD params no longer needed)
        llm,                          # ← one service does three things
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    context.add_message({
        "role": "user",
        "content": "Please greet the user and introduce yourself briefly.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Speech-to-Speech: OpenAI Realtime (LocalAudio)")
    print(" One model handles STT + LLM + TTS, latency ~300ms")
    print(" Headphones recommended (echo issues)")
    print(" Ctrl+C to exit")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
