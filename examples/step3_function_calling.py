"""
Step 3 — Function Calling (Tool Calling)
=========================================
Building on step2, this adds the ability for the agent to call external tools.
After running, you can ask the agent "What's the weather in Tokyo?" or "Any restaurant suggestions in Seattle?"

What you'll learn:
    1. FunctionSchema — how to define the parameter schema for a tool
    2. ToolsSchema     — how to bundle multiple tools together for the LLM
    3. llm.register_function() — register the actual handler function for a tool
    4. FunctionCallParams — the arguments and result callback passed when a function is invoked
    5. Event on_function_calls_started — a hook that fires when a tool is called

Install dependencies:
    pip install "pipecat-ai[local,deepgram,openai,cartesia,silero]" python-dotenv loguru
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# ═══════════════════════════════════════════════════════════════════════════
# Tool functions (simulating API calls)
# In a real scenario these would call a weather API, database, etc.
# ═══════════════════════════════════════════════════════════════════════════

async def get_current_weather(params: FunctionCallParams):
    location = params.arguments.get("location", "Unknown")
    format_ = params.arguments.get("format", "fahrenheit")
    logger.info(f"Tool called: get_current_weather({location}, {format_})")
    # Simulated API result
    await params.result_callback({
        "location": location,
        "conditions": "sunny",
        "temperature": "75" if format_ == "fahrenheit" else "24",
        "unit": format_,
    })


async def get_restaurant_recommendation(params: FunctionCallParams):
    location = params.arguments.get("location", "Unknown")
    logger.info(f"Tool called: get_restaurant_recommendation({location})")
    # Simulated recommendation
    await params.result_callback({
        "name": "The Golden Spoon",
        "cuisine": "Italian",
        "rating": 4.8,
        "location": location,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Tool schemas (tell the LLM each tool's name and parameters)
# ═══════════════════════════════════════════════════════════════════════════

weather_tool = FunctionSchema(
    name="get_current_weather",
    description="Get the current weather in a city",
    properties={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. Tokyo, Japan",
        },
        "format": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit. Infer from the user's location.",
        },
    },
    required=["location", "format"],
)

restaurant_tool = FunctionSchema(
    name="get_restaurant_recommendation",
    description="Get a restaurant recommendation for a given city",
    properties={
        "location": {
            "type": "string",
            "description": "The city, e.g. Seattle, WA",
        },
    },
    required=["location"],
)

# Bundle all tools together
tools = ToolsSchema(standard_tools=[weather_tool, restaurant_tool])


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])

    tts = ElevenLabsTTSService(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM") 
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            system_instruction=(
                "You are a helpful assistant. You can check the weather and "
                "recommend restaurants. Keep responses short and conversational."
            ),
        ),
    )

    # ── Register tool functions ───────────────────────────────────────────
    # Bind the LLM's function calls to their actual Python implementations
    llm.register_function("get_current_weather", get_current_weather)
    llm.register_function("get_restaurant_recommendation", get_restaurant_recommendation)

    # ── Event: when a tool is called, speak a line so the user knows we're working on it ──
    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        await tts.queue_frame(TTSSpeakFrame("Let me check on that for you."))

    # Passing tools into the Context lets the LLM know which tools are available
    context = LLMContext(tools=tools)
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
        params=PipelineParams(enable_metrics=True),
    )

    context.add_message({
        "role": "developer",
        "content": "Greet the user and let them know you can check weather and recommend restaurants.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("\n🎤 Voice agent with tools is running!")
    print("   Try asking: 'What's the weather in Tokyo?' or 'Restaurant suggestions in Seattle?'")
    print("   Press Ctrl+C to stop.\n")

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
