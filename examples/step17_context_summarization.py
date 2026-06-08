"""
Step 17 — Context Summarization (Long-Conversation Memory Compression)
=======================================================================
The longer a conversation grows, the larger the LLM context becomes — more
tokens means higher cost, slower responses, and eventually a context-window
overflow error.  Pipecat has built-in automatic compression: once a threshold
is exceeded, the LLM condenses old messages into a summary while keeping the
most recent turns intact.

How it works:
    Messages  1-100 → Summary ("User chatted about the weather, asked for
                                restaurant recommendations, likes Italian food…")
    Messages 95-100 → Preserved (the most recent turns remain in full)
    Messages  101+  → Conversation continues normally; the LLM sees:
                       summary + recent messages

What you will learn:
    1. enable_auto_context_summarization — one parameter to turn on auto-compression
    2. LLMAutoContextSummarizationConfig — configure trigger conditions (token count, message count)
    3. LLMContextSummaryConfig — configure the target compressed size and number of messages to retain
    4. on_summary_applied — event hook fired when compression occurs
    5. Manual trigger: LLMSummarizeContextFrame (compress on demand, without waiting for auto-trigger)

Installation: (same as step 2)
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]"

Required API keys: DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

# ── Context Summarization imports ──────────────────────────────────────────
from pipecat.utils.context.llm_context_summarization import (
    LLMAutoContextSummarizationConfig,
    LLMContextSummaryConfig,
)
# The data object received by the on_summary_applied event callback (contains counts only, not the summary text)
from pipecat.processors.aggregators.llm_context_summarizer import SummaryAppliedEvent

from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
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
                "You are a helpful assistant for long conversations. "
                "When you notice context is getting summarized, acknowledge it naturally. "
                "Keep responses short."
            ),
        ),
    )

    context = LLMContext()

    # ── Context Summarization configuration ───────────────────────────────
    # Trigger conditions (either one is enough to trigger compression):
    #   max_context_tokens = 1000   → triggers when the estimated context exceeds 1000 tokens (~4000 chars)
    #   max_unsummarized_messages=5 → triggers when more than 5 new messages accumulate (low value for easy testing)
    #
    # Recommended production values:
    #   max_context_tokens = 8000 (default), max_unsummarized_messages = 20 (default)
    summarization_config = LLMAutoContextSummarizationConfig(
        max_context_tokens=1000,         # Set low so compression triggers easily during testing
        max_unsummarized_messages=5,     # Compress every 5 messages (for testing)
        summary_config=LLMContextSummaryConfig(
            target_context_tokens=500,   # Target token count after compression
            min_messages_after_summary=2, # Keep the 2 most recent messages uncompressed
        ),
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
        assistant_params=LLMAssistantAggregatorParams(
            # ── Core: enable automatic compression ─────────────────────────
            enable_auto_context_summarization=True,
            auto_context_summarization_config=summarization_config,
        ),
    )

    # ── Listen for compression events ────────────────────────────────────
    # The actual callback signature is (aggregator, summarizer, event):
    #   - aggregator : the LLMAssistantAggregator that fired the event
    #   - summarizer : the internal summarizer object
    #   - event      : SummaryAppliedEvent — carries message-count statistics only, not the summary text
    @assistant_aggregator.event_handler("on_summary_applied")
    async def on_summary_applied(aggregator, summarizer, event: SummaryAppliedEvent):
        print(f"\n[Context Summarized]")
        print(f"  Before     : {event.original_message_count} messages")
        print(f"  After      : {event.new_message_count} messages")
        print(f"  Compressed : {event.summarized_message_count} messages → summary")
        print(f"  Preserved  : {event.preserved_message_count} recent messages kept")

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

    # You can also trigger compression manually (without waiting for auto-trigger):
    # from pipecat.frames.frames import LLMSummarizeContextFrame
    # await task.queue_frames([LLMSummarizeContextFrame()])

    context.add_message({
        "role": "developer",
        "content": (
            "Start a long conversation. Ask the user about their day, "
            "their favorite things, and have a natural chat. "
            "The context will be summarized after a few exchanges."
        ),
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Context Summarization Demo")
    print(" Context is auto-compressed every 5 messages (or when exceeding 1000 tokens)")
    print(" [Context Summarized] is printed when compression occurs")
    print(" Chat for several rounds to trigger compression")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())

#(pipecat) PS C:\Users\Yuki.Leong\github\pipecat> python .\examples\step17_context_summarization.py
# 2026-06-07 23:36:31.766 | INFO     | pipecat:<module>:14 - ᓚᘏᗢ Pipecat 1.2.1 (Python 3.12.13 (main, Apr 14 2026, 14:31:26) [MSC v.1944 64 bit (AMD64)]) ᓚᘏᗢ
# [transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
# =======================================================
#  Context Summarization Demo
#  Context is auto-compressed every 5 messages (or when exceeding 1000 tokens)
#  [Context Summarized] is printed when compression occurs
#  Chat for several rounds to trigger compression
# =======================================================

# [Context Summarized]
#   Before     : 8 messages
#   After      : 4 messages
#   Compressed : 5 messages → summary
#   Preserved  : 3 recent messages kept

# [Context Summarized]
#   Before     : 7 messages
#   After      : 4 messages
#   Compressed : 4 messages → summary
#   Preserved  : 3 recent messages kept

# [Context Summarized]
#   Before     : 8 messages
#   After      : 4 messages
#   Compressed : 5 messages → summary
#   Preserved  : 3 recent messages kept
