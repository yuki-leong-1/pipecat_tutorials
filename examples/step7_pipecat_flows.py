"""
Step 7 — Pipecat Flows (Structured Conversational State Machine)
================================================================
Ideal for scenarios with well-defined flows: food ordering, appointments, surveys, customer-service routing.
Each Node gives the LLM exactly one task + one set of tools, preventing hallucinations caused by overly large prompts.

What you will learn:
    1. NodeConfig  — define a conversation node (persona / task / functions)
    2. FlowManager — manage transitions between nodes and global state
    3. FlowsFunctionSchema — function definitions for a node, with automatic handler binding
    4. Edge Function — returns (result, next_node) to trigger a node transition
    5. Node Function — returns (result, None)   to stay on the current node
    6. flow_manager.state — share data across nodes
    7. post_actions — actions executed automatically after a node completes (e.g. end_conversation)

Compared with Function Calling in step3:
    step3 = free-form conversation + tool calls (the LLM decides where to go)
    step7 = structured flow + explicit paths (you decide where to go; the LLM only handles the current node)

Installation:
    uv add pipecat-ai-flows
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]"

This example: a three-node coffee-order bot
    greeting   → ask for the user's name
    take_order → ask what they would like
    confirm    → confirm the order and end the conversation
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
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
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

# ── Core imports for Pipecat Flows ────────────────────────────────────────
from pipecat_flows import (
    FlowArgs,       # parameter dict passed when a function is called
    FlowManager,    # manages node transitions and state
    FlowsFunctionSchema,  # defines the functions available to a node
    NodeConfig,     # defines a conversation node
)

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ═══════════════════════════════════════════════════════════════════════════
# Node definitions
# Each node = a dict or NodeConfig object containing:
#   role_message  : system instructions for the LLM (persona; only needed on the first node, inherited after)
#   task_messages : what this node should do (developer role)
#   functions     : which functions this node may call
#   post_actions  : executed automatically after the node completes (e.g. end_conversation)
# ═══════════════════════════════════════════════════════════════════════════

def create_greeting_node() -> NodeConfig:
    """First node: ask for the user's name."""

    # FlowsFunctionSchema = function definition for this node + handler binding
    # When the LLM calls record_name, Pipecat Flows automatically invokes handle_record_name
    record_name_func = FlowsFunctionSchema(
        name="record_name",
        description="Record the customer's name after they provide it.",
        properties={
            "name": {"type": "string", "description": "The customer's name"},
        },
        required=["name"],
        handler=handle_record_name,  # Edge function: returns (result, next_node)
    )

    return NodeConfig(
        name="greeting",
        role_message=(
            "You are a friendly barista at a coffee shop. "
            "Be warm and brief. Responses will be spoken aloud — no markdown."
        ),
        task_messages=[{
            "role": "developer",
            "content": "Greet the customer warmly and ask for their name.",
        }],
        functions=[record_name_func],
    )


async def handle_record_name(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, NodeConfig]:
    """Edge function: record the name and transition to the next node."""
    name = args["name"]
    # flow_manager.state is a dictionary shared across nodes
    flow_manager.state["customer_name"] = name
    return f"Name recorded: {name}", create_take_order_node()


def create_take_order_node() -> NodeConfig:
    """Second node: take the customer's order."""

    record_order_func = FlowsFunctionSchema(
        name="record_order",
        description="Record what the customer wants to order.",
        properties={
            "item": {"type": "string", "description": "The coffee/drink item ordered"},
            "size": {
                "type": "string",
                "enum": ["small", "medium", "large"],
                "description": "The size of the drink",
            },
        },
        required=["item", "size"],
        handler=handle_record_order,
    )

    return NodeConfig(
        name="take_order",
        task_messages=[{
            "role": "developer",
            "content": (
                "Ask the customer what they'd like to order. "
                "We have coffee, tea, and hot chocolate in small, medium, and large."
            ),
        }],
        functions=[record_order_func],
    )


async def handle_record_order(
    args: FlowArgs, flow_manager: FlowManager
) -> tuple[str, NodeConfig]:
    """Edge function: record the order and transition to the confirmation node."""
    flow_manager.state["order_item"] = args["item"]
    flow_manager.state["order_size"] = args["size"]
    return f"Order recorded: {args['size']} {args['item']}", create_confirm_node()


def create_confirm_node() -> NodeConfig:
    """Third node: confirm the order and end the conversation."""

    # Read the name and order from state (deferred via lambda because state is only populated at runtime)
    return NodeConfig(
        name="confirm",
        task_messages=[{
            "role": "developer",
            "content": (
                "Confirm the order details using the customer's name and what they ordered "
                "(available in the conversation history). Tell them the order will be ready shortly. "
                "Be warm and brief."
            ),
        }],
        # post_actions: executed automatically after the node completes; end_conversation shuts down the pipeline
        post_actions=[{"type": "end_conversation"}],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main program
# ═══════════════════════════════════════════════════════════════════════════
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
        settings=OpenAILLMService.Settings(model="gpt-4o-mini"),
    )

    context = LLMContext()
    # Keep the pair object intact — do NOT unpack it; FlowManager needs to call pair.user() / pair.assistant()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),       # user aggregator
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),  # assistant aggregator
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    # FlowManager: connects task / llm / aggregator / transport
    # Pass the pair object (not the unpacked tuple); it calls .user() / .assistant() internally
    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )

    async def start_flow():
        await asyncio.sleep(1)
        # initialize() sets the first node and triggers the LLM to speak
        await flow_manager.initialize(create_greeting_node())

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 50)
    print(" Pipecat Flows Demo — Coffee Shop Bot")
    print(" Flow: greeting → take_order → confirm → end")
    print("=" * 50)

    await asyncio.gather(runner.run(task), start_flow())


if __name__ == "__main__":
    asyncio.run(main())
