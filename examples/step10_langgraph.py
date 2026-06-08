"""
Step 10 — LangGraph Integration (Compatible, Not Incompatible)
==============================================================
LangGraph and Pipecat are fully compatible.
Integration approach: write a custom FrameProcessor as a bridge to replace
the LLM service in the pipeline.

What you will learn:
    1. Why Pipecat + LangGraph are compatible
    2. LangGraphProcessor — how to write a bridging FrameProcessor
    3. Key frame protocol: LLMContextFrame → LLMFullResponseStartFrame → LLMTextFrame × N → LLMFullResponseEndFrame
    4. Converting between Pipecat messages (OpenAI dicts) and LangGraph messages (LangChain Message objects)
    5. How a LangGraph graph manages its own conversation state

Why use LangGraph instead of Pipecat's built-in LLM service?
    - You already have a LangGraph workflow (with complex conditional edges, tools, memory)
    - You want to add a real-time voice interface without rewriting the entire agent logic
    - You need LangGraph features like checkpointing and human-in-the-loop

Limitations:
    - Conversation history is managed by LangGraph MessagesState, not Pipecat LLMContext
    - If you need Pipecat's context summarization or function calling, additional adaptation is required
    - Streaming interruption is not supported (LangGraph invocations are atomic)

Install dependencies:
    uv add langgraph langchain-openai langchain-core
    (No new Pipecat extras required)

Required API keys: DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from typing import Annotated

from dotenv import load_dotenv
from loguru import logger

# ── LangGraph imports ──────────────────────────────────────────────────────
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ── Pipecat imports ────────────────────────────────────────────────────────
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ═══════════════════════════════════════════════════════════════════════════
# 1. LangGraph Graph Definition
#
# This is the simplest possible chatbot graph:
#   user input → chatbot node (calls LLM) → end
#
# In a real scenario this could be any complex LangGraph workflow:
#   - Conditional edges
#   - Multiple nodes (retrieval, tools, routing)
#   - Human-in-the-loop review
#   - Persistent memory (checkpointing)
# ═══════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    # add_messages is LangGraph's built-in reducer that automatically appends new messages to the list
    messages: Annotated[list, add_messages]


def build_langgraph() -> "CompiledStateGraph":
    """Build the LangGraph graph"""
    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.7,
    )

    def chatbot_node(state: GraphState):
        """Call the LLM and return the response"""
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(GraphState)
    builder.add_node("chatbot", chatbot_node)
    builder.set_entry_point("chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════
# 2. LangGraphProcessor — Core Bridge Class
#
# This class follows the official Pipecat framework integration pattern
# (the same pattern used by the built-in LangchainProcessor):
#
# Input:  LLMContextFrame (Pipecat conversation history)
# Output: LLMFullResponseStartFrame → LLMTextFrame × N → LLMFullResponseEndFrame
#
# Key frame requirements (confirmed by official Discord staff):
#   - Must use LLMTextFrame, not TextFrame
#     LLMTextFrame has includes_inter_frame_spaces=True, which the TTS aggregator needs
#   - Must push LLMFullResponseEndFrame, otherwise TTS will not flush the last sentence
#   - LLMFullResponseStartFrame lets the transport know the bot has started speaking
#     (affects VAD and interruption logic)
# ═══════════════════════════════════════════════════════════════════════════

class LangGraphProcessor(FrameProcessor):
    """Bridge processor that embeds a LangGraph graph into a Pipecat pipeline"""

    def __init__(self, graph):
        super().__init__()
        self._graph = graph
        # LangGraph manages its own conversation history (not Pipecat's LLMContext)
        self._lg_messages: list = [
            SystemMessage(content=(
                "You are a helpful voice assistant. "
                "Keep responses short and conversational. "
                "No markdown, no bullet points."
            ))
        ]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            # Messages inside LLMContextFrame are in Pipecat format (OpenAI dict list).
            # We extract only the latest user message and hand it to LangGraph.
            pipecat_messages = frame.context.get_messages()
            last_message = pipecat_messages[-1] if pipecat_messages else None

            if not last_message or not isinstance(last_message, dict):
                await self.push_frame(frame, direction)
                return

            role = last_message.get("role", "")
            content = last_message.get("content", "")

            # Only process user messages
            if role not in ("user", "human") or not content.strip():
                await self.push_frame(frame, direction)
                return

            logger.info(f"LangGraphProcessor: user said: {content!r}")

            # Add the user message to LangGraph's message history
            self._lg_messages.append(HumanMessage(content=content.strip()))

            # ── Push frame sequence: tell Pipecat the bot is starting its reply ──
            await self.push_frame(LLMFullResponseStartFrame())

            try:
                # Invoke the LangGraph graph to get the full response.
                # Note: using ainvoke (non-streaming) here because LangGraph streaming
                # requires additional configuration.
                result = await self._graph.ainvoke({"messages": self._lg_messages})

                # Extract the AI reply from the result
                response_messages = result.get("messages", [])
                ai_message = None
                for msg in reversed(response_messages):
                    if isinstance(msg, AIMessage):
                        ai_message = msg
                        break

                if ai_message and ai_message.content:
                    response_text = ai_message.content
                    logger.info(f"LangGraphProcessor: LangGraph replied: {response_text!r}")

                    # Add the AI reply to LangGraph's history for use in the next turn
                    self._lg_messages.append(ai_message)

                    # Push the response in small chunks (simulates streaming).
                    # In a real scenario you can use graph.astream() for true token streaming.
                    words = response_text.split()
                    chunk_size = 5  # push 5 words at a time
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        # Add a trailing space so the TTS aggregator joins sentences correctly
                        await self.push_frame(LLMTextFrame(chunk + " "))

            except Exception as e:
                logger.error(f"LangGraphProcessor error: {e}")
            finally:
                # ── Must push EndFrame, otherwise TTS will not flush the last sentence ──
                await self.push_frame(LLMFullResponseEndFrame())

        else:
            # Pass all other frames through unchanged
            await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Main Program
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    # Build the LangGraph graph
    graph = build_langgraph()
    logger.info("LangGraph graph compiled successfully")

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

    # ── LangGraphProcessor replaces OpenAILLMService ──────────────────────
    langgraph_processor = LangGraphProcessor(graph)

    # Pipecat's context here is used only to pass user messages to LangGraphProcessor.
    # Conversation history is managed entirely by LangGraph's MessagesState.
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # Pipeline: LangGraphProcessor replaces OpenAILLMService
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        langgraph_processor,   # ← LangGraph runs here
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    # On startup, have the agent speak first (add a message directly to LangGraph history, then trigger)
    langgraph_processor._lg_messages.append(
        HumanMessage(content="Please greet the user briefly.")
    )
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" LangGraph + Pipecat Voice Agent")
    print(" LangGraph graph: simple chatbot (swap in any graph)")
    print(" Try: 'What is LangGraph?' or 'Tell me a joke'")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
