"""
Step 9 — Multi-Agent Architecture (Pipecat Subagents)
======================================================
When a single pipeline isn't enough, use multiple agents working together.
Each agent has its own LLM + pipeline and communicates via the AgentBus.

What you'll learn:
    1. AgentRunner        — manages the lifecycle of all agents
    2. AgentBus           — message bus for inter-agent communication
    3. BusBridgeProcessor — the "router" in the main pipeline that dispatches frames to the active agent
    4. LLMAgent           — base class for an agent that has its own LLM pipeline
    5. @tool decorator    — registers tools inside an LLMAgent (cleaner than register_function)
    6. handoff_to()       — transfers control to another agent (seamless handoff)
    7. @agent_ready       — runs a callback once the specified agent has finished starting up

Architecture diagram:
    AgentRunner
      └── MainAgent (owns transport + BusBridgeProcessor)
            ├── GreeterAgent (greets the user + routes to SupportAgent)
            └── SupportAgent (answers questions + can end the conversation)

    MainAgent's Pipeline:
      mic → STT → user_agg → [BusBridgeProcessor] → TTS → speaker → assistant_agg
                                     ↑↓  (via Bus)
                               GreeterAgent / SupportAgent (each with its own LLM)

Compared with step2 (single agent):
    step2 = everything chained together in one pipeline
    step9 = main agent routes audio; multiple LLM agents wait in parallel; the active agent handles the conversation

Installation:
    uv add pipecat-ai-subagents
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]"
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

# ── Core imports for Subagents ────────────────────────────────────────────
from pipecat_subagents.agents import (
    BaseAgent,
    LLMAgent,               # agent base class that includes an LLM pipeline
    LLMAgentActivationArgs, # arguments passed when activating an agent
    agent_ready,            # decorator that runs after a specified agent has finished starting up
    tool,                   # registers tools inside an LLMAgent (replaces register_function)
)
from pipecat_subagents.bus import AgentBus, BusBridgeProcessor  # message bus and router
from pipecat_subagents.runner import AgentRunner                 # manages all agents
from pipecat_subagents.types import AgentReadyData               # data received by @agent_ready callbacks

load_dotenv()
logger.remove(0)
# logger.add(sys.stderr, level="WARNING")
logger.add(sys.stderr, level="INFO")

# ═══════════════════════════════════════════════════════════════════════════
# LLM Agent base class: both LLM agents inherit from this
# Shared tools: transfer_to_agent (switch agents) and end_conversation (end the conversation)
# ═══════════════════════════════════════════════════════════════════════════
class BaseVoiceAgent(LLMAgent):

    def __init__(self, name: str, *, bus: AgentBus, system_instruction: str):
        # bridged=() means this agent receives frames from the bus (it does not own a transport directly)
        super().__init__(name, bus=bus, bridged=())
        self._system_instruction = system_instruction

    def build_llm(self) -> LLMService:
        """Required by LLMAgent: return the LLM instance this agent uses."""
        return OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMSettings(
                model="gpt-4o-mini",
                system_instruction=self._system_instruction,
            ),
        )

    # ── @tool: a cleaner way to register tools than register_function ─────
    # cancel_on_interruption=False: even if the user interrupts, wait for the tool to finish (ensures the handoff completes)
    @tool(cancel_on_interruption=False)
    async def transfer_to_agent(
        self, params: FunctionCallParams, agent: str, reason: str
    ):
        """Transfer the conversation to another agent.

        Args:
            agent (str): Target agent name ('greeter' or 'support').
            reason (str): Why the user is being transferred.
        """
        logger.info(f"[{self.name}] handoff to '{agent}': {reason}")
        await self.handoff_to(
            agent,
            activation_args=LLMAgentActivationArgs(
                messages=[{"role": "user", "content": reason}],
            ),
            result_callback=params.result_callback,
        )

    @tool
    async def end_conversation(self, params: FunctionCallParams, reason: str):
        """End the conversation when the user says goodbye.

        Args:
            reason (str): Why the conversation is ending.
        """
        logger.info(f"[{self.name}] ending conversation: {reason}")
        await params.llm.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": reason}],
                run_llm=True,
            )
        )
        await self.end(reason=reason, result_callback=params.result_callback)


# ═══════════════════════════════════════════════════════════════════════════
# GreeterAgent: welcomes the user, then transfers to SupportAgent once the need is clear
# ═══════════════════════════════════════════════════════════════════════════
class GreeterAgent(BaseVoiceAgent):

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(
            name,
            bus=bus,
            system_instruction=(
                "You are a friendly greeter. Welcome the user briefly and ask how you can help. "
                "If they have a product question or need support, transfer them to the support agent. "
                "Keep responses very short."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SupportAgent: handles specific questions; can transfer back to greeter or end the conversation
# ═══════════════════════════════════════════════════════════════════════════
class SupportAgent(BaseVoiceAgent):

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(
            name,
            bus=bus,
            system_instruction=(
                "You are a helpful support agent for a fictional coffee shop app. "
                "Answer questions about orders, menu, and hours. "
                "If the user just wants to chat, transfer them back to the greeter. "
                "End the conversation when the user says goodbye. "
                "Keep responses short."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# MainAgent: owns the transport; uses BusBridgeProcessor in place of an LLM
# Responsible for audio I/O and routing frames to the active agent via the bus
# ═══════════════════════════════════════════════════════════════════════════
class MainAgent(BaseAgent):

    def __init__(self, name: str, *, bus: AgentBus):
        super().__init__(name, bus=bus, active=True)  # active=True means this agent starts up immediately
        self._transport = LocalAudioTransport(
            LocalAudioTransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                input_device_index=1,
            )
        )

    async def build_pipeline(self) -> Pipeline:
        """Optional override from BaseAgent: defines this agent's pipeline."""
        stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
        tts = ElevenLabsTTSService(
            api_key=os.environ["ELEVENLABS_API_KEY"],
            settings=ElevenLabsTTSService.Settings(voice="21m00Tcm4TlvDq8ikWAM"),
        )

        context = LLMContext()
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(),
                user_mute_strategies=[AlwaysUserMuteStrategy()],
            ),
        )

        # BusBridgeProcessor: sits in the position normally occupied by an LLM.
        # It forwards frames from STT onto the bus; the active agent processes them
        # and sends its reply back through the bus.
        bridge = BusBridgeProcessor(bus=self.bus, agent_name=self.name)

        pipeline = Pipeline([
            self._transport.input(),
            stt,
            user_aggregator,
            bridge,             # ← key: routes frames to the active LLM agent
            tts,
            self._transport.output(),
            assistant_aggregator,
        ])

        self._task = PipelineTask(pipeline, params=PipelineParams())
        return pipeline

    # name= is a keyword argument (agent_ready signature is *, name: str)
    # the handler is called as handler(data), so it must accept data: AgentReadyData
    @agent_ready(name="greeter")
    async def on_greeter_ready(self, data: AgentReadyData):
        """Once GreeterAgent is ready, activate it so it speaks first."""
        logger.info("Greeter agent ready, activating...")
        # Key: activation_args must include messages so the greeter's LLM actually runs and greets proactively.
        # In LLMAgent.on_activated, the LLM only runs when `if activation.messages:` is true.
        # Without messages the greeter is activated but the LLM never fires → the bot stays
        # silent indefinitely, waiting for the user to speak first.
        await self.activate_agent(
            "greeter",
            args=LLMAgentActivationArgs(
                messages=[{
                    "role": "user",
                    "content": "Greet the user warmly and ask how you can help.",
                }],
                run_llm=True,
            ),
        )


async def main():
    # AgentRunner: manages the lifecycle of all agents.
    # Uses AsyncQueueBus by default (in-process; no Redis required).
    runner = AgentRunner(handle_sigint=False if sys.platform == "win32" else True)

    # Create all agents (they share runner.bus)
    main_agent = MainAgent("main", bus=runner.bus)
    greeter = GreeterAgent("greeter", bus=runner.bus)
    support = SupportAgent("support", bus=runner.bus)

    # Add main to the runner. Agents added before run() are held by the runner
    # and started together when run() is called — this part is fine.
    await runner.add_agent(main_agent)

    # ⚠️ Critical fix: child agents must NOT be added before run().
    # main_agent.add_agent(child) sends a BusAddAgentMessage onto the bus,
    # but the runner only subscribes to and starts the bus inside run().
    # Messages sent before run() are silently dropped by AgentBus.on_message_received
    # (the subscriber list is empty, so the for-loop body never executes).
    # As a result the runner never learns about greeter/support → their pipelines
    # never start → they never become ready → on_greeter_ready never fires →
    # the bot is completely silent with no log output whatsoever.
    #
    # Fix: wait until main's pipeline is running (on_ready fires, meaning the bus
    # is live and the runner is subscribed), then call add_agent — the message
    # will actually reach the runner.
    @main_agent.event_handler("on_ready")
    async def _add_children(agent):
        logger.info("Main agent ready, adding child agents...")
        await agent.add_agent(greeter)
        await agent.add_agent(support)

    print("=" * 55)
    print(" Multi-Agent Demo")
    print(" Agents: MainAgent → GreeterAgent ↔ SupportAgent")
    print(" Greeter will welcome you, then transfer to Support")
    print(" Say 'goodbye' to end the conversation")
    print("=" * 55)

    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
