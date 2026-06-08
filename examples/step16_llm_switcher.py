"""
Step 16 — LLMSwitcher (Runtime LLM Switching + Failover)
====================================================
Seamlessly switch LLMs while the pipeline is running, without a restart.

Use cases:
    - Cost optimization: default to a cheap model (gpt-4o-mini), switch to a pricier one (gpt-4o) for complex questions
    - Failover: if one LLM API errors out, automatically switch to a backup
    - Multilingual: switch to a model that handles a particular language better
    - A/B testing: compare the behavior of different models

What you'll learn:
    1. LLMSwitcher — replaces a single LLM service and manages multiple LLMs inside the pipeline
    2. ServiceSwitcherStrategyManual — trigger a switch manually
    3. ServiceSwitcherStrategyFailover — automatic failover (an LLM errors → switch to the next one)
    4. ManuallySwitchServiceFrame — send this frame to trigger a manual switch
    5. Key requirement: all LLMs participating in the switch must share the same LLMContext

In this example:
    - gpt-4o-mini is the default (fast and cheap)
    - Say "switch to smart" to switch to gpt-4o (more precise but expensive)
    - Say "switch to fast" to switch back to gpt-4o-mini
    Note: if you have an Anthropic or Google key, replacing the second LLM with one of those makes for a more meaningful demo

Installation: (same as step2)
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]"

Required API keys: DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    ManuallySwitchServiceFrame,  # ← sending this frame triggers a manual LLM switch
    TranscriptionFrame,
    TTSSpeakFrame,
)

# ── LLMSwitcher imports ───────────────────────────────────────────────────
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.service_switcher import (
    ServiceSwitcherStrategyFailover,  # automatic failover
    ServiceSwitcherStrategyManual,    # manually triggered switch
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
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ── Switch Command Detector ───────────────────────────────────────────────
class SwitchCommandDetector(FrameProcessor):
    """Listens for user commands and triggers an LLM switch."""

    def __init__(self, llm_switcher: LLMSwitcher, llm_mini, llm_full, tts):
        super().__init__()
        self._switcher = llm_switcher
        self._llm_mini = llm_mini
        self._llm_full = llm_full
        self._tts = tts
        self._current = "mini"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.lower()
            # Route to the appropriate LLM based on what the user said
            if ("switch to smart" in text or "smart mode" in text) and self._current != "full":
                self._current = "full"
                # ManuallySwitchServiceFrame: tells LLMSwitcher to switch to the specified LLM
                await self.push_frame(
                    ManuallySwitchServiceFrame(service=self._llm_full)
                )
                await self._tts.queue_frame(
                    TTSSpeakFrame("Switched to smart mode. Using GPT-4o now.")
                )
                return

            elif ("switch to fast" in text or "fast mode" in text) and self._current != "mini":
                self._current = "mini"
                await self.push_frame(
                    ManuallySwitchServiceFrame(service=self._llm_mini)
                )
                await self._tts.queue_frame(
                    TTSSpeakFrame("Switched to fast mode. Using GPT-4o-mini now.")
                )
                return

        await self.push_frame(frame, direction)


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

    # ── Two LLM instances ─────────────────────────────────────────────────
    # Key: must use the universal LLMContext (not OpenAILLMContext)
    # so that both LLMs share the same conversation history
    system_instruction = (
        "You are a helpful assistant. "
        "Keep responses short and conversational. "
        "Mention which model you are when asked."
    )

    llm_mini = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            system_instruction=system_instruction + " You are the fast/mini model.",
        ),
    )
    llm_full = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o",
            system_instruction=system_instruction + " You are the smart/full model.",
        ),
    )

    # ── LLMSwitcher: manages multiple LLMs ───────────────────────────────
    # ServiceSwitcherStrategyManual: waits for a ManuallySwitchServiceFrame to trigger a switch
    # ServiceSwitcherStrategyFailover: automatically switches to the next LLM when a non-fatal error occurs
    llm_switcher = LLMSwitcher(
        llms=[llm_mini, llm_full],           # first one is active by default
        strategy_type=ServiceSwitcherStrategyManual,
    )
    # For automatic failover, use this instead:
    # llm_switcher = LLMSwitcher(llms=[llm_mini, llm_full], strategy_type=ServiceSwitcherStrategyFailover)

    # Register tools via the switcher (registers them on all LLMs simultaneously)
    # llm_switcher.register_function("my_tool", my_tool_handler)

    context = LLMContext()  # universal LLMContext (not OpenAILLMContext)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    switch_detector = SwitchCommandDetector(llm_switcher, llm_mini, llm_full, tts)

    pipeline = Pipeline([
        transport.input(),
        stt,
        switch_detector,         # ← listens for "switch to smart/fast"
        user_aggregator,
        llm_switcher,            # ← replaces a single llm; internally tracks which LLM is active
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams())

    context.add_message({
        "role": "developer",
        "content": "Greet the user. Tell them you're currently gpt-4o-mini (fast mode). They can say 'switch to smart' to use gpt-4o.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" LLMSwitcher Demo")
    print(" Default: gpt-4o-mini (fast, cheap)")
    print(" Say 'switch to smart' → gpt-4o (smart, expensive)")
    print(" Say 'switch to fast'  → back to gpt-4o-mini")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())


#(pipecat) PS C:\Users\Yuki.Leong\github\pipecat> python .\examples\step16_llm_switcher.py           
# 2026-06-07 23:28:13.553 | INFO     | pipecat:<module>:14 - ᓚᘏᗢ Pipecat 1.2.1 (Python 3.12.13 (main, Apr 14 2026, 14:31:26) [MSC v.1944 64 bit (AMD64)]) ᓚᘏᗢ
# [transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
# =======================================================
#  LLMSwitcher Demo
#  Default: gpt-4o-mini (fast, cheap)
#  Say 'switch to smart' → gpt-4o (smart, expensive)
#  Say 'switch to fast'  → back to gpt-4o-mini
# =======================================================