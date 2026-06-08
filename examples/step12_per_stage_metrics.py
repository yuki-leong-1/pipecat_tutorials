"""
Step 12 — Duration and Processing Time Per Pipeline Stage
==========================================================
Focused on using an Observer to measure performance metrics for each pipeline stage.

You will learn:
    1. MetricsData.processor — each metric is tagged with the service it came from (Deepgram/OpenAI/ElevenLabs)
    2. The definitions of three core latency metrics:
       - TTFB (Time To First Byte): time from request sent to first output received
       - Processing Time: total time for an entire service from start to completion
       - Text Aggregation: time from the first LLM token to the first complete sentence (the "wait for first sentence" latency for TTS)
    3. Using BotStartedSpeakingFrame timestamps to calculate end-to-end (E2E) latency
    4. After each conversation turn, print a complete stage breakdown table for that turn

Metric meanings for each pipeline stage:

  [STT - Deepgram]
    TTFB          = audio in → first transcribed word out (network + model first-word time)
    ProcessingTime = total time to transcribe the entire audio segment

  [LLM - OpenAI]
    TTFB          = context sent → first token out (network + model first-token time)
    ProcessingTime = total time for the entire LLM response to be generated (usually = TTFB + generation time)

  [TTS - ElevenLabs]
    TTFB          = first sentence text in → first audio chunk out (synthesis first-syllable time)
    ProcessingTime = internal processing time of the synthesis engine (usually very short due to streaming)
    TextAggregation = first LLM token out → enough text accumulated for the first complete sentence (sentence accumulation wait time)

  E2E Latency    = user stops speaking → bot starts playing audio (latency as actually perceived by the user)
                   ≈ STT tail + LLM TTFB + TTS TTFB + TextAggregation

Required API keys: DEEPGRAM + OPENAI + ELEVENLABS
"""

import asyncio
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMRunFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMUsageMetricsData,
    ProcessingMetricsData,
    TextAggregationMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
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

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")  # Only show output we print ourselves


# ═══════════════════════════════════════════════════════════════════════════
# Data structure: metrics snapshot for a single conversation turn
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TurnMetrics:
    turn_number: int
    transcription: str = ""

    # TTFB for each stage (seconds)
    stt_ttfb: float | None = None
    llm_ttfb: float | None = None
    tts_ttfb: float | None = None

    # Processing time for each stage (seconds)
    stt_processing: float | None = None
    llm_processing: float | None = None
    tts_processing: float | None = None

    # TTS-specific: time from the first LLM token to the first complete sentence
    tts_text_aggregation: float | None = None

    # LLM token usage
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0

    # TTS character count
    tts_chars: int = 0

    # E2E latency (user stops speaking → bot starts playing audio)
    user_stopped_ts: float | None = None    # nanosecond timestamp
    bot_started_ts: float | None = None     # nanosecond timestamp

    @property
    def e2e_latency_ms(self) -> float | None:
        if self.user_stopped_ts and self.bot_started_ts:
            return (self.bot_started_ts - self.user_stopped_ts) / 1e6  # ns → ms
        return None

    def print_table(self):
        def ms(v):
            return f"{v * 1000:6.0f}ms" if v is not None else "     —  "

        e2e = f"{self.e2e_latency_ms:.0f}ms" if self.e2e_latency_ms else "—"

        print(f"\n{'═' * 65}")
        print(f" Turn {self.turn_number}: \"{self.transcription}\"")
        print(f"{'─' * 65}")
        print(f" {'Stage':<18} {'TTFB':>10}  {'Processing':>12}  Notes")
        print(f"{'─' * 65}")

        # STT row
        print(
            f" {'STT (Deepgram)':<18} {ms(self.stt_ttfb):>10}  {ms(self.stt_processing):>12}"
        )

        # LLM row
        token_note = ""
        if self.llm_prompt_tokens:
            token_note = f"  p:{self.llm_prompt_tokens} c:{self.llm_completion_tokens}"
        print(
            f" {'LLM (OpenAI)':<18} {ms(self.llm_ttfb):>10}  {ms(self.llm_processing):>12}{token_note}"
        )

        # TTS row
        char_note = f"  {self.tts_chars}chars" if self.tts_chars else ""
        print(
            f" {'TTS (ElevenLabs)':<18} {ms(self.tts_ttfb):>10}  {ms(self.tts_processing):>12}{char_note}"
        )

        # Text aggregation (time TTS waits for the first sentence)
        if self.tts_text_aggregation is not None:
            print(
                f" {'  └ text aggregation':<18} {'':>10}  {ms(self.tts_text_aggregation):>12}"
                f"  (wait for 1st sentence)"
            )

        print(f"{'─' * 65}")
        print(f" E2E Latency (user stop → bot speak): {e2e}")
        print(f"{'═' * 65}")


# ═══════════════════════════════════════════════════════════════════════════
# Per-Stage Metrics Observer
# ═══════════════════════════════════════════════════════════════════════════

class PerStageMetricsObserver(BaseObserver):
    """
    At the end of each conversation turn, prints the stage breakdown table for that turn.

    How it works:
    1. Listens for MetricsFrame → extracts TTFB / Processing metrics for each service
       MetricsData.processor field identifies the source (e.g. "OpenAILLMService#0")
    2. Listens for UserStoppedSpeakingFrame → records the timestamp when the user stopped speaking
    3. Listens for BotStartedSpeakingFrame  → records the timestamp when the bot starts playing audio, calculates E2E
    4. Listens for BotStoppedSpeakingFrame  → one turn ends: prints the table, resets current-turn data
    5. Listens for TranscriptionFrame       → records transcribed text (used as the table heading)
    """

    def __init__(self):
        super().__init__()
        self._turn = 0
        self._current: TurnMetrics | None = None
        self._session_turns: list[TurnMetrics] = []

    def _ensure_current_turn(self):
        if self._current is None:
            self._turn += 1
            self._current = TurnMetrics(turn_number=self._turn)

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        ts = data.timestamp  # nanoseconds

        # ── E2E timestamps ────────────────────────────────────────────────────
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._ensure_current_turn()
            self._current.user_stopped_ts = ts

        elif isinstance(frame, BotStartedSpeakingFrame):
            if self._current and self._current.bot_started_ts is None:
                self._current.bot_started_ts = ts

        elif isinstance(frame, BotStoppedSpeakingFrame):
            # Bot finished speaking = this turn is over, print the table
            if self._current:
                self._session_turns.append(self._current)
                self._current.print_table()
                self._current = None

        # ── Transcription text ──────────────────────────────────────────────────────
        elif isinstance(frame, TranscriptionFrame):
            self._ensure_current_turn()
            self._current.transcription = frame.text

        # ── MetricsFrame: all performance data comes from here ───────────────────────────
        elif isinstance(frame, MetricsFrame):
            self._ensure_current_turn()
            for d in frame.data:
                p = d.processor.lower()  # lowercase for easy contains-check

                # --- TTFB ---
                if isinstance(d, TTFBMetricsData):
                    if "deepgram" in p or "stt" in p:
                        self._current.stt_ttfb = d.value
                    elif "openai" in p or "llm" in p or "gpt" in p:
                        self._current.llm_ttfb = d.value
                    elif "elevenlabs" in p or "tts" in p or "cartesia" in p:
                        self._current.tts_ttfb = d.value

                # --- Processing Time ---
                elif isinstance(d, ProcessingMetricsData):
                    if "deepgram" in p or "stt" in p:
                        self._current.stt_processing = d.value
                    elif "openai" in p or "llm" in p or "gpt" in p:
                        self._current.llm_processing = d.value
                    elif "elevenlabs" in p or "tts" in p or "cartesia" in p:
                        self._current.tts_processing = d.value

                # --- Text Aggregation (TTS-specific) ---
                elif isinstance(d, TextAggregationMetricsData):
                    self._current.tts_text_aggregation = d.value

                # --- Token Usage ---
                elif isinstance(d, LLMUsageMetricsData):
                    self._current.llm_prompt_tokens = d.value.prompt_tokens
                    self._current.llm_completion_tokens = d.value.completion_tokens

                # --- TTS chars ---
                elif isinstance(d, TTSUsageMetricsData):
                    self._current.tts_chars += d.value

    def print_session_summary(self):
        if not self._session_turns:
            return

        total_turns = len(self._session_turns)
        stt_ttfbs = [t.stt_ttfb for t in self._session_turns if t.stt_ttfb]
        llm_ttfbs = [t.llm_ttfb for t in self._session_turns if t.llm_ttfb]
        llm_procs = [t.llm_processing for t in self._session_turns if t.llm_processing]
        tts_ttfbs = [t.tts_ttfb for t in self._session_turns if t.tts_ttfb]
        e2es = [t.e2e_latency_ms for t in self._session_turns if t.e2e_latency_ms]
        total_tokens = sum(
            t.llm_prompt_tokens + t.llm_completion_tokens for t in self._session_turns
        )
        total_chars = sum(t.tts_chars for t in self._session_turns)

        def avg_ms(lst):
            return f"{sum(lst) / len(lst) * 1000:.0f}ms" if lst else "—"

        print(f"\n{'═' * 65}")
        print(f" Session Summary  ({total_turns} turns)")
        print(f"{'─' * 65}")
        print(f" {'Stage':<25} {'Avg TTFB':>10}  {'Avg Processing':>14}")
        print(f"{'─' * 65}")
        print(f" {'STT (Deepgram)':<25} {avg_ms(stt_ttfbs):>10}  {'—':>14}")
        print(f" {'LLM (OpenAI)':<25} {avg_ms(llm_ttfbs):>10}  {avg_ms(llm_procs):>14}")
        print(f" {'TTS (ElevenLabs)':<25} {avg_ms(tts_ttfbs):>10}  {'—':>14}")
        print(f"{'─' * 65}")
        print(f" Avg E2E latency : {avg_ms(e2es)}")
        print(f" Total LLM tokens: {total_tokens}")
        print(f" Total TTS chars : {total_chars}")
        print(f"{'═' * 65}\n")


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

    metrics_observer = PerStageMetricsObserver()

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,          # enable → TTFBMetricsData, ProcessingMetricsData
            enable_usage_metrics=True,    # enable → LLMUsageMetricsData, TTSUsageMetricsData
        ),
        observers=[metrics_observer],
    )

    context.add_message({
        "role": "developer",
        "content": "Greet the user briefly.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 65)
    print(" Per-Stage Metrics Demo")
    print(" A stage breakdown table for each turn will be printed after the turn ends")
    print(" Press Ctrl+C to exit; session averages will be printed on exit")
    print("=" * 65)

    try:
        await runner.run(task)
    finally:
        metrics_observer.print_session_summary()


if __name__ == "__main__":
    asyncio.run(main())
