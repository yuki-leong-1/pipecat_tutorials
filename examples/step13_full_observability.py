"""
Step 13 — Full Observability (all metrics beyond just performance)
==================================================================
Pipecat provides 6 layers of observability; this example demonstrates all of them.

Layer overview:
    Layer 1  Performance metrics      MetricsFrame → TTFBMetricsData / ProcessingMetricsData (step12)
    Layer 2  E2E latency breakdown    UserBotLatencyObserver (built-in, more complete than step12)
    Layer 3  Conversation turn events TurnTrackingObserver (turn start/end/interrupted/duration)
    Layer 4  Pipeline startup timing  StartupTimingObserver (initialization time per processor)
    Layer 5  OpenTelemetry tracing    enable_tracing=True (requires Jaeger/Langfuse; covered conceptually here)
    Layer 6  Custom conversation analytics  ConversationAnalyticsObserver (hand-written; tracks interruptions, speaking durations, etc.)

Key insight:
    UserBotLatencyObserver already has most of the functionality hand-written in step12 built in,
    and it also provides on_latency_breakdown with a detailed, chronologically ordered event timeline.
    For production use, it is recommended to use the built-in observer rather than writing your own.

Installation: (same as step2 — no new extras required)
    uv add "pipecat-ai[local,deepgram,openai,elevenlabs,silero]"

Required API keys: DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterruptionFrame,
    LLMRunFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)

# ── Built-in observers ─────────────────────────────────────────────────────
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.observers.startup_timing_observer import StartupTimingObserver
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver

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
logger.remove(0)
logger.add(sys.stderr, level="WARNING")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 6: Custom Conversation Analytics Observer
#
# Tracks "user behavior" and "conversation quality" rather than technical performance:
# - Number of interruptions (InterruptionFrame)
# - Distribution of user speaking durations (UserStartedSpeakingFrame → UserStoppedSpeakingFrame)
# - Distribution of bot speaking durations (BotStartedSpeakingFrame → BotStoppedSpeakingFrame)
# - Whether each turn was interrupted
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationAnalytics:
    total_turns: int = 0
    interrupted_turns: int = 0
    e2e_latencies: list[float] = field(default_factory=list)
    user_speaking_durations: list[float] = field(default_factory=list)
    bot_speaking_durations: list[float] = field(default_factory=list)

    # Internal state
    _user_speaking_start: float | None = None
    _bot_speaking_start: float | None = None
    _user_stopped_ts: float | None = None
    _bot_started_ts: float | None = None
    _interrupted_this_turn: bool = False

    def avg(self, lst) -> str:
        return f"{sum(lst)/len(lst):.2f}s" if lst else "—"

    def summary(self) -> str:
        pct = (
            f"{self.interrupted_turns / self.total_turns * 100:.0f}%"
            if self.total_turns else "—"
        )
        avg_e2e = (
            f"{sum(self.e2e_latencies)/len(self.e2e_latencies)*1000:.0f}ms"
            if self.e2e_latencies else "—"
        )
        return (
            f"\n{'═'*55}\n"
            f" Conversation Quality Report\n"
            f"{'─'*55}\n"
            f" Total turns         : {self.total_turns}\n"
            f" Interrupted turns   : {self.interrupted_turns} ({pct})\n"
            f" Avg E2E latency     : {avg_e2e}\n"
            f" Avg user speaking   : {self.avg(self.user_speaking_durations)}\n"
            f" Avg bot speaking    : {self.avg(self.bot_speaking_durations)}\n"
            f"{'═'*55}"
        )


class ConversationAnalyticsObserver(BaseObserver):
    """
    Tracks conversation behavior metrics without concern for technical performance details.

    Can be used to answer:
    - Do users tend to interrupt the bot? (interruption rate)
    - How long does the user speak on average? (user speaking duration)
    - How long does the bot speak on average? (bot speaking duration)
    - What is the E2E response latency? (user stop → bot start)
    """

    def __init__(self):
        super().__init__()
        self._data = ConversationAnalytics()

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        ts_secs = data.timestamp / 1e9

        # User started speaking
        if isinstance(frame, UserStartedSpeakingFrame):
            self._data._user_speaking_start = ts_secs
            self._data._user_stopped_ts = None  # reset

        # User stopped speaking
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._data._user_stopped_ts = ts_secs
            if self._data._user_speaking_start:
                duration = ts_secs - self._data._user_speaking_start
                self._data.user_speaking_durations.append(duration)
                self._data._user_speaking_start = None

        # Bot started speaking
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._data._bot_speaking_start = ts_secs
            self._data._interrupted_this_turn = False
            # Calculate E2E latency
            if self._data._user_stopped_ts:
                e2e = ts_secs - self._data._user_stopped_ts
                self._data.e2e_latencies.append(e2e)

        # Bot stopped speaking = this turn has ended
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if self._data._bot_speaking_start:
                duration = ts_secs - self._data._bot_speaking_start
                self._data.bot_speaking_durations.append(duration)
                self._data._bot_speaking_start = None
            self._data.total_turns += 1
            if self._data._interrupted_this_turn:
                self._data.interrupted_turns += 1
            self._data._interrupted_this_turn = False

        # Interruption event (user speech interrupts the bot)
        elif isinstance(frame, InterruptionFrame):
            if self._data._bot_speaking_start is not None:
                self._data._interrupted_this_turn = True

    def print_summary(self):
        print(self._data.summary())


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
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",
            system_instruction="You are a helpful assistant. Keep responses short.",
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

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    # ── Layer 4: StartupTimingObserver ────────────────────────────────────────
    startup_observer = StartupTimingObserver()

    @startup_observer.event_handler("on_startup_timing_report")
    async def on_startup(observer, report):
        print(f"\n[Startup] Pipeline ready in {report.total_duration_secs:.2f}s")
        for t in report.processor_timings:
            print(f"  {t.processor_name:<35} {t.duration_secs:.3f}s")

    # ── Layer 2: UserBotLatencyObserver ───────────────────────────────────────
    # Built-in observer, more complete than the hand-written version in step12:
    # - on_latency_measured : simple E2E latency in seconds
    # - on_latency_breakdown: detailed event timeline in chronological order
    # - on_first_bot_speech_latency: client connection → bot first utterance
    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_speech(observer, latency):
        print(f"\n[First Speech] {latency:.3f}s after pipeline start")

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency(observer, latency):
        print(f"\n[E2E Latency] {latency * 1000:.0f}ms")

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_breakdown(observer, breakdown):
        # breakdown.chronological_events() returns a chronologically ordered list of event strings
        # Pipecat organizes these for you: user turn → STT TTFB → LLM TTFB → TTS TTFB → text agg
        print("[Latency Breakdown]")
        for event in breakdown.chronological_events():
            print(f"  {event}")
        if breakdown.user_turn_secs:
            print(f"  user speaking duration: {breakdown.user_turn_secs:.2f}s")
        if breakdown.function_calls:
            for fc in breakdown.function_calls:
                print(f"  function call: {fc}")

    # ── Layer 3: TurnTrackingObserver ─────────────────────────────────────────
    # Tracks turn rhythm independently of performance metrics
    # turn_end_timeout_secs: how long to wait after the bot stops speaking before the turn is considered ended
    turn_observer = TurnTrackingObserver(turn_end_timeout_secs=2.0)

    @turn_observer.event_handler("on_turn_started")
    async def on_turn_started(observer, turn_count):
        print(f"\n[Turn {turn_count}] Started")

    @turn_observer.event_handler("on_turn_ended")
    async def on_turn_ended(observer, turn_count, duration, was_interrupted):
        status = "INTERRUPTED ⚡" if was_interrupted else "completed"
        print(f"[Turn {turn_count}] Ended — {duration:.1f}s, {status}")

    # ── Layer 6: Custom conversation analytics ───────────────────────────────────────────────
    analytics_observer = ConversationAnalyticsObserver()

    # ── Attach all observers ─────────────────────────────────────────────────
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,       # Required for UserBotLatencyObserver breakdown
            enable_usage_metrics=True,
        ),
        observers=[
            startup_observer,          # Layer 4: startup timing
            latency_observer,          # Layer 2: E2E breakdown
            turn_observer,             # Layer 3: turn events
            analytics_observer,        # Layer 6: conversation quality analytics
        ],
    )

    context.add_message({
        "role": "developer",
        "content": "Greet the user briefly.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Full Observability Demo")
    print(" Layer 2 E2E breakdown — Layer 3 turn events — Layer 4 startup timing")
    print(" Layer 6 conversation quality analytics (printed after Ctrl+C)")
    print()
    print(" Layer 5 OpenTelemetry (not demonstrated here):")
    print("   uv add 'pipecat-ai[tracing]'")
    print("   PipelineTask(..., enable_tracing=True, enable_turn_tracking=True)")
    print("   → integrates with Jaeger / Langfuse / Opik")
    print("=" * 55)

    try:
        await runner.run(task)
    finally:
        analytics_observer.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
