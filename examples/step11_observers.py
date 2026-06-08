"""
Step 11 — Observer System (Non-Intrusive Monitoring)
=====================================================
Observers let you monitor the flow of all frames through a pipeline without
inserting processors or affecting the data stream.

Core difference between Observer and FrameProcessor:
    FrameProcessor = IN the pipeline; frames must pass through it and can be modified/filtered
    BaseObserver   = OUTSIDE the pipeline; passively watches all frames without affecting data

What you will learn:
    1. Built-in observers: LLMLogObserver, TranscriptionLogObserver, MetricsLogObserver
    2. The three hooks of BaseObserver: on_push_frame, on_process_frame, on_pipeline_started
    3. FramePushed dataclass: source, destination, frame, direction, timestamp
    4. What data lives inside MetricsFrame: TTFB, processing time, token usage, TTS chars
    5. Writing a custom Observer: collect session stats in real time and print a summary at the end
    6. Observing BotStartedSpeakingFrame / BotStoppedSpeakingFrame to measure bot speaking duration

Pipeline structure is identical to step2; only the PipelineTask gains observers:
    transport.input() → stt → user_aggregator → llm → tts → transport.output() → assistant_aggregator

Key: enable_metrics=True and enable_usage_metrics=True must both be enabled,
     otherwise MetricsFrame will not be emitted and MetricsLogObserver receives no data.

Install dependencies: (same as step2)
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]" python-dotenv loguru

Required API keys: DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMRunFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)

# ── Observer imports ──────────────────────────────────────────────────────
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.observers.loggers.metrics_log_observer import MetricsLogObserver
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()

# Set loguru to DEBUG so the built-in observer output is visible
logger.remove(0)
logger.add(sys.stderr, level="DEBUG", format="<level>{level}</level> | {message}")


# ═══════════════════════════════════════════════════════════════════════════
# Custom Observer: Session Statistics Collector
#
# This observer demonstrates:
# - How to filter for specific frame types inside on_push_frame
# - How to track stateful data (bot speaking start time, cumulative metrics)
# - How to perform initialization inside on_pipeline_started
# - How to print a summary report after the session ends
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SessionStats:
    """Statistics for one complete conversation session."""
    start_time: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    turns: int = 0                      # Number of user speaking turns
    transcriptions: list[str] = field(default_factory=list)  # All transcribed text

    total_prompt_tokens: int = 0        # Total LLM prompt tokens
    total_completion_tokens: int = 0    # Total LLM completion tokens
    total_tts_chars: int = 0            # Total characters consumed by TTS

    ttfb_values: list[float] = field(default_factory=list)  # All TTFB values (seconds)
    processing_times: dict = field(default_factory=lambda: defaultdict(list))

    bot_speaking_durations: list[float] = field(default_factory=list)
    _bot_speaking_start: float | None = None  # Timestamp when the bot started speaking

    def record_bot_started(self, timestamp_ns: int):
        self._bot_speaking_start = timestamp_ns / 1e9

    def record_bot_stopped(self, timestamp_ns: int):
        if self._bot_speaking_start is not None:
            duration = timestamp_ns / 1e9 - self._bot_speaking_start
            self.bot_speaking_durations.append(duration)
            self._bot_speaking_start = None

    def summary(self) -> str:
        elapsed = asyncio.get_event_loop().time() - self.start_time
        avg_ttfb = sum(self.ttfb_values) / len(self.ttfb_values) if self.ttfb_values else 0
        avg_bot_speaking = (
            sum(self.bot_speaking_durations) / len(self.bot_speaking_durations)
            if self.bot_speaking_durations else 0
        )

        lines = [
            "",
            "═" * 55,
            " Session Summary",
            "═" * 55,
            f" Duration        : {elapsed:.1f}s",
            f" Conversation turns: {self.turns}",
            "",
            " STT Transcriptions:",
        ]
        for i, t in enumerate(self.transcriptions, 1):
            lines.append(f"   {i}. {t!r}")

        lines += [
            "",
            " LLM Usage:",
            f"   Prompt tokens   : {self.total_prompt_tokens}",
            f"   Completion tokens: {self.total_completion_tokens}",
            f"   Total tokens    : {self.total_prompt_tokens + self.total_completion_tokens}",
            "",
            " TTS Usage:",
            f"   Total characters: {self.total_tts_chars}",
            "",
            " Latency:",
            f"   Avg TTFB        : {avg_ttfb * 1000:.0f}ms  (across {len(self.ttfb_values)} measurements)",
            f"   Avg bot speaking: {avg_bot_speaking:.1f}s per turn",
            "═" * 55,
        ]
        return "\n".join(lines)


class SessionStatsObserver(BaseObserver):
    """
    Collects statistics for the entire conversation session.

    on_push_frame is called for every frame transfer inside the pipeline.
    data.source      = the processor that pushed this frame
    data.destination = the processor that receives this frame
    data.frame       = the frame itself
    data.direction   = DOWNSTREAM or UPSTREAM
    data.timestamp   = nanosecond timestamp (pipeline internal clock)
    """

    def __init__(self):
        super().__init__()
        self._stats = SessionStats()

    async def on_pipeline_started(self):
        """Called after the pipeline has fully started (StartFrame has passed all processors)."""
        print("\n[Observer] Pipeline started — session tracking begins\n")

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        ts = data.timestamp

        # ── STT transcription ────────────────────────────────────────────
        if isinstance(frame, TranscriptionFrame):
            self._stats.turns += 1
            self._stats.transcriptions.append(frame.text)

        # ── Bot speaking duration ─────────────────────────────────────────
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._stats.record_bot_started(ts)

        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._stats.record_bot_stopped(ts)

        # ── MetricsFrame: performance metrics ────────────────────────────
        # MetricsFrame is only emitted when enable_metrics=True
        # Use frame.__class__.__name__ to avoid a noisy import
        elif frame.__class__.__name__ == "MetricsFrame":
            for d in frame.data:
                if isinstance(d, TTFBMetricsData):
                    # TTFB = Time To First Byte, in seconds
                    self._stats.ttfb_values.append(d.value)

                elif isinstance(d, LLMUsageMetricsData):
                    self._stats.total_prompt_tokens += d.value.prompt_tokens
                    self._stats.total_completion_tokens += d.value.completion_tokens

                elif isinstance(d, TTSUsageMetricsData):
                    self._stats.total_tts_chars += d.value

                elif isinstance(d, ProcessingMetricsData):
                    service_name = str(data.source).split("#")[0]
                    self._stats.processing_times[service_name].append(d.value)

    def print_summary(self):
        print(self._stats.summary())


# ═══════════════════════════════════════════════════════════════════════════
# Main
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
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            system_instruction=(
                "You are a helpful assistant. Keep responses short and conversational."
            ),
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_mute_strategies=[AlwaysUserMuteStrategy()],
        ),
    )

    # Pipeline is identical to step2 — no changes whatsoever
    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    # ── Instantiate the custom observer ──────────────────────────────────
    stats_observer = SessionStatsObserver()

    # ── All observers are registered here; none are inserted into the pipeline ──
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,          # Required → MetricsFrame will be emitted
            enable_usage_metrics=True,    # Required → LLMUsageMetricsData / TTSUsageMetricsData
        ),
        observers=[
            # ── Built-in observers ─────────────────────────────────────
            TranscriptionLogObserver(),   # 💬 Print each STT transcription result
            LLMLogObserver(),             # 🧠 Print LLM-generated tokens (streaming)
            MetricsLogObserver(           # 📊 Print performance metrics
                include_metrics={
                    TTFBMetricsData,       # Show TTFB only
                    LLMUsageMetricsData,   # and token usage
                }
            ),
            # ── Custom observer ────────────────────────────────────────
            stats_observer,               # 📈 Collect session statistics
        ],
    )

    context.add_message({
        "role": "developer",
        "content": "Greet the user briefly. Let them know they can talk to you.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Observer Demo")
    print(" You will see in the terminal:")
    print("   💬 STT transcriptions (TranscriptionLogObserver)")
    print("   🧠 LLM-generated tokens (LLMLogObserver)")
    print("   📊 TTFB and token usage (MetricsLogObserver)")
    print(" Press Ctrl+C to stop — a Session Summary will be printed")
    print("=" * 55)

    try:
        await runner.run(task)
    finally:
        # Print the summary after Ctrl+C or pipeline completion
        stats_observer.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
