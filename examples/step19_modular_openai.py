"""
Step 19 — Modular Pipeline: All OpenAI (STT + LLM + TTS)
==============================================================
Like step2, this is a "three-stage" voice agent, but all three services
are replaced with OpenAI — you only need **one** OPENAI_API_KEY to run
the entire pipeline.

Two ways to build a voice agent with OpenAI:

    Modular (this example, step19)      Speech-to-Speech (step14)
    ────────────────────────────────    ──────────────────────────────
    mic → OpenAI STT                    mic → OpenAI Realtime → speaker
        → OpenAI LLM                              ↑
        → OpenAI TTS                   One model handles STT+LLM+TTS
        → speaker                      Lower latency (~300ms), but pricier
    Three independent models,          Black box, hard to inject custom logic
    each swappable/debuggable          Requires Realtime API access
    Slightly higher latency (~800ms),
    cheaper

Why it's called "modular":
    STT / LLM / TTS are three independent processors, each replaceable.
    Want to swap STT? Replace OpenAISTTService with DeepgramSTTService (step2 mixes them).
    Want to swap TTS? Replace with ElevenLabsTTSService / CartesiaTTSService.
    This example shows the "all-OpenAI" combination — single vendor, single key, single bill.

What you'll learn:
    1. OpenAISTTService —— OpenAI speech-to-text (gpt-4o-transcribe)
    2. OpenAITTSService —— OpenAI text-to-speech (gpt-4o-mini-tts)
    3. All three services share a single OPENAI_API_KEY
    4. OpenAI STT operates in "segmented" mode (transcribes sentence by sentence):
       VAD on the aggregator detects speech start/end and broadcasts those events
       upstream to the STT processor

Install dependencies:
    uv add "pipecat-ai[local,openai,silero]" python-dotenv loguru
    (no deepgram / elevenlabs / cartesia needed)

Required API key: only OPENAI_API_KEY

Configuration:
    Copy .env.example to .env and fill in OPENAI_API_KEY
    Press Ctrl+C to exit
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

# VAD: detects when the user starts and stops speaking (OpenAI segmented STT uses it to cut sentences)
from pipecat.audio.vad.silero import SileroVADAnalyzer

# Frames: containers for data
from pipecat.frames.frames import LLMRunFrame

# Pipeline: the chain of processors
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

# AlwaysUserMuteStrategy: mutes user input while the bot is speaking, preventing speaker echo from interrupting itself
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

# ── Three AI services, all from OpenAI ─────────────────────────────────────────
from pipecat.services.openai.stt import OpenAISTTService   # speech → text
from pipecat.services.openai.llm import OpenAILLMService   # text → response
from pipecat.services.openai.tts import OpenAITTSService   # text → speech

# Language enum (tells STT the input language)
from pipecat.transcriptions.language import Language

# Local audio Transport (computer mic + speakers)
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    # ═══════════════════════════════════════════════════════════════════════
    # 1. TRANSPORT —— local microphone (input) + speakers (output)
    # ═══════════════════════════════════════════════════════════════════════
    # To list device indices:
    #   uv run python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
    # Fill in the physical mic index for input_device_index (None = system default)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            input_device_index=1,  # physical mic — avoids loopback devices (device 0 / Sound Mapper)
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 2. SERVICES —— all three are OpenAI, sharing one OPENAI_API_KEY
    # ═══════════════════════════════════════════════════════════════════════
    api_key = os.environ["OPENAI_API_KEY"]

    # STT: converts speech to text
    # Input: AudioRawFrame  →  Output: TranscriptionFrame
    # OpenAI STT runs in REST "segmented" mode: once VAD detects you've finished
    # a sentence, the audio segment is sent as a whole to gpt-4o-transcribe
    # (not word-by-word streaming).
    stt = OpenAISTTService(
        api_key=api_key,
        settings=OpenAISTTService.Settings(
            model="gpt-4o-transcribe",  # alternatives: "whisper-1" / "gpt-4o-mini-transcribe"
            language=Language.EN,
        ),
    )

    # LLM: processes the conversation and generates a response
    # Input: LLMContextFrame  →  Output: TextFrame (streaming)
    llm = OpenAILLMService(
        api_key=api_key,
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",  # cheap and fast, good for learning
            system_instruction=(
                "You are a helpful assistant in a voice conversation. "
                "Keep your responses short and conversational (1-3 sentences). "
                "Do not use bullet points, markdown, or emojis."
            ),
        ),
    )

    # TTS: converts the LLM's text response into speech
    # Input: TextFrame  →  Output: TTSAudioRawFrame (24kHz PCM)
    # Available voices: alloy / ash / ballad / cedar / coral / echo / fable /
    #                   marin / nova / onyx / sage / shimmer / verse
    tts = OpenAITTSService(
        api_key=api_key,
        settings=OpenAITTSService.Settings(
            model="gpt-4o-mini-tts",
            voice="alloy",
            # instructions: gpt-4o-mini-tts supports "acting directions" to control tone/emotion
            instructions="Speak in a warm, friendly and natural tone.",
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 3. CONTEXT & AGGREGATORS
    # ═══════════════════════════════════════════════════════════════════════
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # VAD lives on the aggregator: when it detects "speech started/stopped",
            # it broadcasts VADUserStarted/StoppedSpeakingFrame **upstream**.
            # The upstream OpenAI segmented STT receives these frames to know when
            # to send the buffered audio for transcription.
            # So this VAD simultaneously drives STT sentence segmentation and LLM triggering.
            vad_analyzer=SileroVADAnalyzer(),
            # Echo cancellation: mute the mic while the bot is speaking
            # (essential when using speakers; remove when using headphones to allow interruption)
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PIPELINE —— same structure as step2, just all three services swapped to OpenAI
    # ═══════════════════════════════════════════════════════════════════════
    #  [mic] → transport.input() → stt → user_aggregator → llm → tts
    #        → transport.output() → [speakers] → assistant_aggregator
    pipeline = Pipeline([
        transport.input(),       # receive audio from mic
        stt,                     # OpenAI: speech → text
        user_aggregator,         # accumulate what the user said (VAD lives here)
        llm,                     # OpenAI: generate response
        tts,                     # OpenAI: text → speech
        transport.output(),      # play to speakers
        assistant_aggregator,    # record what the agent said
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # 5. PIPELINE TASK
    # ═══════════════════════════════════════════════════════════════════════
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # 6. Let the agent speak first on startup
    # ═══════════════════════════════════════════════════════════════════════
    context.add_message({
        "role": "developer",
        "content": "Please greet the user and briefly introduce yourself. Keep it under 2 sentences.",
    })
    await task.queue_frames([LLMRunFrame()])

    # ═══════════════════════════════════════════════════════════════════════
    # 7. RUN
    # ═══════════════════════════════════════════════════════════════════════
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("\n🎤 Modular OpenAI voice agent is running! Speak into your microphone.")
    print("   STT + LLM + TTS are all provided by OpenAI — only one API key needed.")
    print("   Press Ctrl+C to stop.\n")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
