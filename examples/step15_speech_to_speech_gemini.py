"""
Step 15 — Speech-to-Speech: Gemini Live (LocalAudio version)
=============================================================
Same Speech-to-Speech concept as step14, but using Google Gemini Live.

OpenAI Realtime vs Gemini Live comparison:
    OpenAI Realtime:  Stable, mature, more voice options, strong semantic turn detection
    Gemini Live:      Integrates Google Search tools, supports video (multimodal),
                      supports VAD tuning via Google VAD parameters

What you will learn:
    1. GeminiLiveLLMService — Gemini's S2S implementation
    2. LiveVADParams / GeminiVADParams — Gemini's VAD configuration
    3. Parameter differences between the two S2S services
    4. Why S2S pipeline structures are essentially the same (the benefit of a universal LLMContext)

Installation:
    uv add "pipecat-ai[local,google,silero]"

Required API key: GOOGLE_API_KEY (requires Gemini API access)
Apply here: https://aistudio.google.com/apikey (free)
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

# ── Gemini Live imports ───────────────────────────────────────────────────
from pipecat.services.google.gemini_multimodal_live.gemini import (
    GeminiLiveLLMService,
    GeminiLiveParams,
    InputParams,
)

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

    # ── GeminiLiveLLMService ──────────────────────────────────────────────
    llm = GeminiLiveLLMService(
        api_key=os.environ["GOOGLE_API_KEY"],
        system_instruction=(
            "You are a helpful voice assistant. "
            "Keep responses short and conversational. "
            "No markdown or bullet points."
        ),
        params=GeminiLiveParams(
            model="gemini-2.0-flash-live-001",  # Latest Gemini Live model
            voice_name="Puck",                   # Puck/Charon/Kore/Fenrir/Aoede
            input=InputParams(
                # Gemini's VAD parameters (similar to OpenAI's turn_detection)
                # Accessible with just a GOOGLE_API_KEY — no extra configuration needed
            ),
        ),
    )

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    # ── Pipeline structure is identical to step14 ────────────────────────
    pipeline = Pipeline([
        transport.input(),
        context_aggregator.user(),
        llm,                           # GeminiLive handles STT + LLM + TTS
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
    print(" Speech-to-Speech: Gemini Live (LocalAudio)")
    print(" One model handles STT + LLM + TTS")
    print(" Headphones recommended (echo issues)")
    print(" Ctrl+C to quit")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
