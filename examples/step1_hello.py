"""
Step 1 — Simplest Pipecat Example
================================
Only an ElevenLabs API key is required.
After running, your computer's speaker will say one sentence, then the program exits.

Goal: understand the three most fundamental concepts
    1. Pipeline  - a chain of processors
    2. Frame     - the "container" for data flowing through the chain (TTSSpeakFrame, EndFrame)
    3. Transport - how audio gets out (LocalAudioTransport = local speakers)

Install dependencies:
    pip install "pipecat-ai[local,elevenlabs]" python-dotenv loguru

Configuration:
    Copy .env.example to .env and fill in ELEVENLABS_API_KEY
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()

# Route loguru log output to stderr at INFO level (not too noisy)
logger.remove(0)
logger.add(sys.stderr, level="INFO")


async def main():
    # ── 1. Transport ──────────────────────────────────────────────────────
    # Transport handles how audio comes in (input) and goes out (output)
    # LocalAudioTransport = uses the computer's microphone and speakers
    # Only audio_out is enabled here because we only need to speak, not listen
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_out_enabled=True)
    )

    # ── 2. TTS Service ────────────────────────────────────────────────────
    # TTS = Text-to-Speech, responsible for converting text into audio
    # It receives TTSSpeakFrame (text) and outputs AudioRawFrame (audio)
    # tts = CartesiaTTSService(
    #   api_key=os.environ["CARTESIA_API_KEY"],
    #   settings=CartesiaTTSService.Settings(
    #   voice="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady voice
    # ),
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        # voice_id can be found in the Voice Library on the ElevenLabs website
        # This is the built-in "Rachel" voice (available on free accounts)
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM")
    )

    # ── 3. Pipeline ───────────────────────────────────────────────────────
    # Pipeline chains processors together in order
    # Data flow: tts → transport.output()
    #
    # When TTSSpeakFrame("Hello!") enters the pipeline:
    #   tts processes it → generates AudioRawFrame
    #   transport.output() receives it → plays it back
    pipeline = Pipeline([tts, transport.output()])

    # ── 4. PipelineTask ───────────────────────────────────────────────────
    # PipelineTask wraps the pipeline into a runnable async task
    task = PipelineTask(pipeline)

    # ── 5. Send Frames ────────────────────────────────────────────────────
    async def say_something():
        # Wait 1 second for the pipeline to fully initialize
        await asyncio.sleep(1)

        # queue_frames puts frames into the pipeline's processing queue
        # TTSSpeakFrame: instructs TTS to speak this text
        # EndFrame: tells the pipeline "task complete, you can shut down now"
        await task.queue_frames([
            TTSSpeakFrame("Hello! I am your Pipecat voice agent. This is Step 1."),
            EndFrame(),
        ])

    # ── 6. PipelineRunner ─────────────────────────────────────────────────
    # PipelineRunner runs the task and manages the asyncio event loop
    # On Windows, handle_sigint must be False (otherwise it raises an error)
    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    # asyncio.gather runs runner and say_something concurrently
    # runner.run(task) keeps running until it receives EndFrame, then exits
    await asyncio.gather(runner.run(task), say_something())


if __name__ == "__main__":
    asyncio.run(main())
