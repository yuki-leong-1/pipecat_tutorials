"""
Step 6 — Dynamic Context Injection (Persona Switching + User Memory)
=====================================================================
A more "life-like" agent: it knows who you are, can switch personality
mid-conversation, and can inject external information on the fly.

What you'll learn:
    1. User Profile injection   ── load user info into context at startup
    2. Persona switching        ── say "switch to formal" to make the agent formal
    3. Dynamic context injection ── insert information at runtime (e.g. search
                                    results, database queries)
    4. LLMMessagesAppendFrame   ── append messages to context without interrupting
                                    the conversation
    5. LLMMessagesUpdateFrame   ── fully replace context (nuclear-level reset)
    6. Use a FrameProcessor to watch for keywords and trigger context operations

How to run:
    uv run python examples/step6_context_injection.py

Things you can say:
    - "switch to formal"  → agent switches to a formal tone
    - "switch to casual"  → agent switches back to a relaxed tone
    - "who am I"          → agent answers using the injected profile
    - "what time is it"   → demonstrates real-time data injection

Required API keys: DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ── Simulated user profile (in a real project, read this from a database) ────
USER_PROFILE = {
    "name": "Alex",
    "language": "English",
    "preferences": "prefers concise answers, interested in technology",
    "subscription": "Pro user since 2024",
}

# ── Persona definitions ───────────────────────────────────────────────────
PERSONAS = {
    "casual": (
        "You are a friendly, casual assistant. Use relaxed language, "
        "contractions, and a warm tone. Keep responses short."
    ),
    "formal": (
        "You are a professional, formal assistant. Use proper grammar, "
        "avoid contractions, and maintain a respectful tone. Keep responses concise."
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# CommandDetector: watches for keywords and triggers context operations
#
# This is a real-world application of the FrameProcessor pattern learned in
# step4: intercept TranscriptionFrames, check for keywords, and inject the
# corresponding context changes.
# ═══════════════════════════════════════════════════════════════════════════
class CommandDetector(FrameProcessor):

    def __init__(self, context: LLMContext, task_ref: list, tts_ref: list):
        super().__init__()
        self._context = context
        self._task_ref = task_ref
        self._tts_ref = tts_ref
        self._current_persona = "casual"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.lower().strip()
            task = self._task_ref[0]

            # ── Command 1: switch persona ─────────────────────────────────
            if "switch to formal" in text or "be more formal" in text:
                await self._switch_persona("formal", task)
                return  # swallow this frame — do not pass it to the LLM

            if "switch to casual" in text or "be more casual" in text:
                await self._switch_persona("casual", task)
                return

            # ── Command 2: inject real-time data (demo: current time) ────
            if "what time" in text or "what's the time" in text:
                now = datetime.now().strftime("%I:%M %p on %A, %B %d")
                # LLMMessagesAppendFrame: append information to context without
                # interrupting the conversation. The difference from calling
                # context.add_message() directly is that AppendFrame is processed
                # through the pipeline, ensuring correct ordering.
                inject_frame = LLMMessagesAppendFrame(messages=[{
                    "role": "developer",
                    "content": f"[Real-time data injected] Current time: {now}. "
                               f"Answer the user's time question using this.",
                }])
                await self.push_frame(inject_frame, direction)
                # Do NOT return here — let the original TranscriptionFrame continue
                # downstream as well. The LLM will see the injected developer
                # message first, then the user's question.

        await self.push_frame(frame, direction)

    async def _switch_persona(self, persona_name: str, task):
        """Switch persona: update the system instruction and trigger an LLM acknowledgement."""
        if persona_name == self._current_persona:
            return

        self._current_persona = persona_name
        new_instruction = PERSONAS[persona_name]

        # LLMMessagesUpdateFrame: fully replace the entire context.
        # Rebuild context with the new system instruction while retaining the user profile.
        new_messages = [
            {
                "role": "developer",
                "content": (
                    f"{new_instruction}\n\n"
                    f"User profile: {USER_PROFILE}\n\n"
                    f"You just switched to {persona_name} mode. "
                    f"Briefly acknowledge this switch."
                ),
            }
        ]

        update_frame = LLMMessagesUpdateFrame(messages=new_messages)
        await task.queue_frames([update_frame, LLMRunFrame()])

        print(f"\n[System] 🎭 Switched to {persona_name} persona")


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
        settings=OpenAILLMService.Settings(model="gpt-4o-mini"),
    )

    # ── Context initialization: inject the user profile ───────────────────
    # This simulates a common real-world agent pattern: after the user logs in,
    # read their profile from a database and inject it into the context's system message.
    context = LLMContext()
    context.add_message({
        "role": "developer",
        "content": (
            f"{PERSONAS['casual']}\n\n"
            f"You know the following about this user:\n"
            f"- Name: {USER_PROFILE['name']}\n"
            f"- Language: {USER_PROFILE['language']}\n"
            f"- Preferences: {USER_PROFILE['preferences']}\n"
            f"- Account: {USER_PROFILE['subscription']}\n\n"
            f"Use this information to personalize your responses. "
            f"You can say the user's name occasionally."
        ),
    })

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    task_ref = [None]
    tts_ref = [tts]
    command_detector = CommandDetector(context, task_ref, tts_ref)

    pipeline = Pipeline([
        transport.input(),
        stt,
        command_detector,        # ← intercepts commands before user_aggregator
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=False),
    )
    task_ref[0] = task

    # ── Startup: have the agent greet the user by name from their profile ─
    context.add_message({
        "role": "developer",
        "content": "Greet the user by name. Keep it to one sentence.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Context Injection Demo")
    print(" Try saying:")
    print("   'who am I'          → agent uses your profile")
    print("   'switch to formal'  → change persona")
    print("   'switch to casual'  → change back")
    print("   'what time is it'   → real-time data injection")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
