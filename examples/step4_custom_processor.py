"""
Step 4 — Custom FrameProcessor
================================
In Pipecat, STT / LLM / TTS / VAD are all FrameProcessors.
This step teaches you how to write your own FrameProcessor and insert it
anywhere in a pipeline.

You will learn:
    1. The basic structure and lifecycle of a FrameProcessor
    2. process_frame(frame, direction) -- every frame passes through here
    3. push_frame(frame) -- passes the frame to the next processor
    4. "Swallowing" a frame -- not calling push_frame filters it out
    5. Three practical examples:
       - TranscriptionPrinter   prints what the user says to the terminal
       - ConversationLogger     saves the conversation to a JSON file
       - ConversationResetter   saying "reset" clears the conversation history

How to run:
    uv run python examples/step4_custom_processor.py

Required API keys (same as step2): DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
# ── Core imports: these two are needed to write a custom processor ──
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ═══════════════════════════════════════════════════════════════════════════
# Processor 1：TranscriptionPrinter
#
# Insertion point: stt → [TranscriptionPrinter] → user_aggregator
# Purpose: prints each STT transcription result to the terminal
#
# Key concepts:
#   - Subclass FrameProcessor
#   - Override process_frame and check the frame type
#   - Always call push_frame at the end, or the frame won't continue downstream
# ═══════════════════════════════════════════════════════════════════════════
class TranscriptionPrinter(FrameProcessor):

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # 1. Call super() first so the base class can handle system-level frames (e.g. StartFrame, EndFrame)
        await super().process_frame(frame, direction)

        # 2. Check the frame type; TranscriptionFrame = the final transcription result from STT
        if isinstance(frame, TranscriptionFrame):
            print(f"\n👤 YOU: {frame.text}")

        # 3. Pass the frame to the next processor.
        #    If push_frame is not called, the frame disappears here (i.e. it is filtered out).
        await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# Processor 2：ConversationLogger
#
# Insertion point: llm → [ConversationLogger] → tts
# Purpose: accumulates LLM replies into complete sentences, prints them,
#          and saves them to a JSON file
#
# Key concepts:
#   - LLM output is "streaming": one sentence = many TextFrame or LLMTextFrame chunks
#   - Use a buffer to reassemble fragments into a complete sentence
#   - FrameDirection.DOWNSTREAM = frame flowing from upstream to downstream (normal direction)
# ═══════════════════════════════════════════════════════════════════════════
class ConversationLogger(FrameProcessor):

    def __init__(self, log_file="conversation_log.json"):
        super().__init__()
        self._log_file = log_file
        self._log = []
        self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # Text frames output by the LLM: TextFrame or LLMTextFrame
        # direction == DOWNSTREAM ensures we only process frames coming from upstream (avoids duplication)
        if isinstance(frame, (TextFrame, LLMTextFrame)) and direction == FrameDirection.DOWNSTREAM:
            self._buffer += frame.text

            # When a sentence-ending punctuation mark is encountered, treat the sentence as complete
            if self._buffer.strip() and frame.text.endswith((".", "!", "?", "\n")):
                sentence = self._buffer.strip()
                print(f"🤖 BOT: {sentence}")
                self._log.append({
                    "timestamp": datetime.now().isoformat(),
                    "role": "assistant",
                    "text": sentence,
                })
                self._buffer = ""
                self._save_log()

        await self.push_frame(frame, direction)

    def _save_log(self):
        with open(self._log_file, "w", encoding="utf-8") as f:
            json.dump(self._log, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Processor 3：ConversationResetter
#
# Insertion point: stt → TranscriptionPrinter → [ConversationResetter] → user_aggregator
# Purpose: clears the conversation history and restarts when the user says "reset"
#
# Key concepts:
#   - A FrameProcessor can directly manipulate external objects (context)
#   - Not calling push_frame swallows the frame (prevents "reset" from reaching the LLM)
#   - Use context.set_messages([]) to clear, then add_message to restore the system prompt,
#     and send LLMRunFrame to trigger a new opening greeting
# ═══════════════════════════════════════════════════════════════════════════
class ConversationResetter(FrameProcessor):

    def __init__(self, context: LLMContext, task_ref: list):
        super().__init__()
        self._context = context
        # task_ref is a list [task] used to indirectly reference the task (avoids circular references)
        self._task_ref = task_ref

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            if "reset" in frame.text.lower():
                # Clear the conversation history, keeping only the system instruction
                self._context.set_messages([])
                self._context.add_message({
                    "role": "developer",
                    "content": "The conversation was just reset. Greet the user again briefly.",
                })
                print("\n[System] 🔄 Conversation reset!")

                # Trigger the LLM to run immediately (deliver a new opening greeting)
                task = self._task_ref[0]
                if task:
                    await task.queue_frames([LLMRunFrame()])

                # Swallow this TranscriptionFrame so "reset" never enters the LLM context
                return

        await self.push_frame(frame, direction)


# ═══════════════════════════════════════════════════════════════════════════
# Main program
# ═══════════════════════════════════════════════════════════════════════════
async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True)
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
                "You are a helpful assistant. Keep responses short. "
                "If the user asks you to reset, a separate system will handle it."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # task_ref wraps the task in a list so ConversationResetter can reference it after initialization
    task_ref = [None]

    # Instantiate the custom processors
    transcription_printer = TranscriptionPrinter()
    resetter = ConversationResetter(context, task_ref)
    conversation_logger = ConversationLogger("conversation_log.json")

    # Pipeline data flow:
    #   mic → stt → TranscriptionPrinter → ConversationResetter → user_agg
    #       → llm → ConversationLogger → tts → speaker → assistant_agg
    pipeline = Pipeline([
        transport.input(),
        stt,
        transcription_printer,   # ← Processor 1: print transcription
        resetter,                # ← Processor 3: detect "reset" command
        user_aggregator,
        llm,
        conversation_logger,     # ← Processor 2: log bot replies
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=False),
    )
    task_ref[0] = task  # allow ConversationResetter to access task

    context.add_message({
        "role": "developer",
        "content": "Greet the user. Let them know they can say 'reset' to restart the conversation.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Custom Processor Demo")
    print(" 👤 YOU: (your speech appears here)")
    print(" 🤖 BOT: (bot responses appear here)")
    print(" Say 'reset' to clear conversation history")
    print(" Conversation saved to: conversation_log.json")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
