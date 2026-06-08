"""
Step 8 — MCPClient (connecting MCP tools inside a bot)
=======================================================
Lets the bot's LLM call tools provided by an MCP server directly,
without having to write FunctionSchema by hand.

What you will learn:
    1. MCPClient                — connect to an MCP server
    2. mcp.register_tools(llm) — auto-discover tools + register them with the LLM
                                  (replaces the manual wiring from step 3 in one line)
    3. Three connection modes:
       - StdioServerParameters  → local subprocess (used in this example)
       - SseServerParameters    → remote SSE
       - StreamableHttpParameters → remote HTTP (e.g. GitHub Copilot MCP)
    4. tools_filter             — expose only a subset of tools to the LLM
    5. tools_output_filters     — truncate/filter tool return values (prevents context overflow)

Compared with step 3:
    step3 = manually write FunctionSchema + register_function + handler functions
    step8 = MCPClient auto-discovers tools + registers them in one line; LLM calls the MCP server directly

This example uses mcp-server-time (the recommended beginner MCP server from Pipecat):
    - Provides get_current_time / list_timezones tools
    - Runs via uvx — no manual installation, no Node.js required
    - Try asking the bot: "What time is it?" or "What time is it in Tokyo?"

Installation:
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero,mcp]"
    (mcp-server-time is handled automatically by uvx)
"""

import asyncio
import os
import shutil
import sys

from dotenv import load_dotenv
from loguru import logger

from mcp import StdioServerParameters

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
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService

# ── MCPClient: the core class for connecting to an MCP server ────────────────
from pipecat.services.mcp_service import MCPClient

from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

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
                "You are a helpful voice assistant with access to time tools. "
                "Keep responses short and conversational. "
                "When asked about time or timezones, use your tools."
            ),
        ),
    )

    # ── MCPClient: use async with to manage the connection lifecycle ─────────
    # StdioServerParameters = run the MCP server as a local subprocess
    # uvx is uv's tool runner, similar to npx — automatically installs and runs Python packages
    async with MCPClient(
        server_params=StdioServerParameters(
            command=shutil.which("uvx"),          # locate the uvx executable
            args=["mcp-server-time"],              # which server to run
        ),
        # tools_filter: expose only these two tools to the LLM, ignore the rest
        tools_filter=["get_current_time", "convert_time"],
        # tools_output_filters: truncate overly long return values
        tools_output_filters={
            "get_current_time": lambda r: str(r)[:200],
        },
    ) as mcp:
        # register_tools does three things:
        # 1. Connects to the MCP server and lists all available tools
        # 2. Converts each tool's schema into Pipecat's FunctionSchema format
        # 3. Calls llm.register_function() to bind a handler for each tool
        # The return value is a ToolsSchema; pass it to LLMContext so the LLM knows which tools exist
        tools = await mcp.register_tools(llm)

        # LLMContext requires tools to be passed in so the LLM can call them in its responses
        context = LLMContext(tools=tools)
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
                user_mute_strategies=[AlwaysUserMuteStrategy()],
            ),
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

        task = PipelineTask(pipeline, params=PipelineParams())

        context.add_message({
            "role": "developer",
            "content": "Greet the user and let them know they can ask about the current time.",
        })
        await task.queue_frames([LLMRunFrame()])

        runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

        print("=" * 55)
        print(" MCP Integration Demo")
        print(" MCP server: mcp-server-time (via uvx)")
        print(" Try asking:")
        print("   'What time is it?'")
        print("   'What time is it in Tokyo?'")
        print("=" * 55)

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())

# What is the "MCP server" and where does it live?

# In step 8, the MCP server is mcp-server-time — a standalone Python program.
# It is not a persistent service; instead, step 8 spawns it as a temporary
# subprocess over stdio at runtime, and it exits when the bot exits.

# Physical location (on disk)

# uvx installs it as a self-contained tool (with its own venv). Three key paths:

# ┌──────────────────────────────┬───────────────────────────────────────────────────────────────┐
# │           Contents           │                             Path                              │
# ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
# │ Tool install dir (own venv)  │ C:\Users\Yuki.Leong\AppData\Roaming\uv\tools\mcp-server-time\ │
# ├──────────────────────────────┼───────────────────────────────────┤
# │ Executable entry point       │ ...\mcp-server-time\Scripts\mcp-server-time.exe               │
# ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
# │ Python source (server logic) │ ...\mcp-s\mcp_server_time\        │
# ├──────────────────────────────┼───────────────────────────────────────────────────────────────┤
# │ uvx download cache           │ C:\Users\Yuki.Leong\AppData\Local\uv\cache                    │
# └──────────────────────────────┴───────────────────────────────────┘

# Version: mcp-server-time v2026.6.4.

# How it gets started

# These lines in step 8 specify where the MCP server is and how to run it:

# server_params=StdioServerParameters(
#     command=shutil.which("uvx"),     # use uvx to launch it
#     args=["mcp-server-time"],         # which server to start
# )

# Execution flow:

# step8 (your bot)
#    └─ uvx mcp-server-time          ← spawned
#         └─ mcp-server-time.exe      ← the actual MCP server process
#              ↕ stdio (JSON-RPC over stdin/stdout)
#         bot discovers tools via this pipe (get_curr ...

# The bot and the server communicate over standard I/O (stdio) — no network port is used.
# That's why you won't see a listening port in Task Manager or netstat;
# it's just a temporary child process.

# Viewing / running the server directly

# # Start the server directly (waits for stdin input; press Ctrl+C to exit)
# uvx mcp-server-time

# # View the main source entry point
# code "C:\Users\Yuki.Leong\AppData\Roaming\uv\tools\mcp-server-time\Lib\site-packages\mcp_server_time\__main__.py"

# # Confirm where it is installed
