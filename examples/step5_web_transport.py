"""
Step 5 — Web Transport (browser-accessible)
============================================
Upgrade from "local microphone" to "browser connection".
After running, open http://localhost:7860/client in your browser to start a conversation.

What you'll learn:
    1. Pipecat Runner system — handles transport selection and server startup uniformly
    2. transport_params dict — supports multiple transports, switchable from the command line
    3. Event handling — on_client_connected / on_client_disconnected
    4. The difference between the two web transports:
       - webrtc (SmallWebRTC) — no extra key needed, P2P direct connection
       - daily              — requires DAILY_API_KEY, better support for multi-party calls

How to run:
    # Option 1: WebRTC (no Daily key required)
    uv run python examples/step5_web_transport.py --transport webrtc

    # Option 2: Daily (requires DAILY_API_KEY)
    uv run python examples/step5_web_transport.py --transport daily

    Then open your browser → http://localhost:7860/client

Required API keys: DEEPGRAM + OPENAI + ELEVENLABS
Optional API key:  DAILY_API_KEY (only needed when using --transport daily)

Sign up for a free Daily account: https://dashboard.daily.co/u/signup
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

# Runner utility: responsible for parsing the --transport argument and creating the corresponding transport
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

# FastAPI WebSocket (used with --transport twilio, or for direct WebSocket connections)
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# Daily is imported only when actually selected (daily-python does not support Windows; deferred import avoids errors)
def _daily_params():
    from pipecat.transports.daily.transport import DailyParams
    return DailyParams(audio_in_enabled=True, audio_out_enabled=True)

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════
# transport_params: a dict where key = transport name, value = its Params
#
# This is the recommended Pipecat pattern for supporting multiple transports
# from a single codebase. The active transport is chosen via --transport.
# ═══════════════════════════════════════════════════════════════════════════
transport_params = {
    # SmallWebRTC: lightweight P2P WebRTC built into pipecat, no extra service needed
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    # Daily WebRTC: requires DAILY_API_KEY, more fully featured (recording, multi-party calls, etc.)
    # Windows does not support daily-python; run on Linux/macOS or WSL2
    "daily": _daily_params,
    # Twilio / WebSocket: for telephone dial-in
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    """Bot logic — the transport has already been decided by the runner."""
    logger.info("Bot starting...")

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM"),
    )
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            system_instruction=(
                "You are a friendly voice assistant accessible via web browser. "
                "Keep responses short and conversational."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,  # Web users typically wear headphones, so interruptions can be enabled
            enable_metrics=True,
        ),
        # idle_timeout_secs: how long to wait before auto-ending due to inactivity (read from runner_args)
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # ── Event handling ──────────────────────────────────────────────────────
    # on_client_connected: fires when a browser/client connects
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected: {client}")
        # Once the client connects, have the agent speak first to greet them
        context.add_message({
            "role": "developer",
            "content": "A user just connected via web browser. Greet them warmly and ask how you can help.",
        })
        await task.queue_frames([LLMRunFrame()])

    # on_client_disconnected: fires when a client leaves
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected: {client}")
        # Cancel the pipeline and free resources
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


# ── bot() is the entry point for Pipecat Cloud ─────────────────────────────
# Pipecat Cloud calls this function on deployment;
# locally, main() also calls it
async def bot(runner_args: RunnerArguments):
    """Pipecat Cloud-compatible entry point."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


# ── Local execution ─────────────────────────────────────────────────────────
# pipecat.runner.run.main() will:
#   1. Parse command-line arguments (--transport, --port, etc.)
#   2. Start a FastAPI server (default port 7860)
#   3. Serve the built-in browser client at http://localhost:7860/client
#   4. Wait for a client connection, then call bot(runner_args)
if __name__ == "__main__":
    from pipecat.runner.run import main
    main()

#   ┌──────────────────────────────────┬──────────────────────────────────────────────────────┐
#   │             Option               │                       Description                    │
#   ├──────────────────────────────────┼──────────────────────────────────────────────────────┤
#   │ WSL2 (recommended)               │ Linux environment inside Windows; installs            │
#   │                                  │ pipecat-ai[daily] without issues                     │
#   ├──────────────────────────────────┼──────────────────────────────────────────────────────┤
#   │ Docker                           │ Run a Linux container                                │
#   ├──────────────────────────────────┼──────────────────────────────────────────────────────┤
#   │ Deploy to Pipecat Cloud / Linux  │ Production environments are naturally Linux           │
#   │ VPS                              │                                                      │
#   └──────────────────────────────────┴──────────────────────────────────────────────────────┘
#   uv run python .\examples\step5_web_transport.py --transport webrtc
