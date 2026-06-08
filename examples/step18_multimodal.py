"""
Step 18 — Multimodal (Images/Vision into LLM Context)
======================================================
Enable a voice agent to "see" images: add images to the LLM context and ask
questions about them using voice.

You will learn:
    1. LLMContext.create_image_url_message() — add an image URL to the context
    2. LLMContext.create_image_message()     — add a local image (bytes) to the context
    3. LLMMessagesAppendFrame + run_llm=True  — inject an image and immediately trigger the LLM
    4. Use a FrameProcessor to listen for keywords and dynamically inject images into the conversation
    5. Which LLMs support multimodal input (GPT-4o, Claude 3, Gemini)

Use cases:
    - Visual question answering ("What is in this image?")
    - Document comprehension ("Explain this chart for me")
    - Real-time visual assistant (screenshot → ask a question)
    - Medical / industrial image analysis

Installation: (same as step2; gpt-4o-mini supports images)
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
    LLMMessagesAppendFrame,
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
from pipecat.turns.user_mute import AlwaysUserMuteStrategy

load_dotenv()
logger.remove(0)
logger.add(sys.stderr, level="WARNING")

# Sample image URLs — important: must be publicly accessible addresses that OpenAI's servers can download directly!
# create_image_url_message only sends the URL to OpenAI; OpenAI itself fetches the image.
# ⚠️ Do not use upload.wikimedia.org: it returns 403/400 for non-browser requests (including OpenAI's downloader)
#    → OpenAI reports 400 invalid_image_url (this was the error encountered previously).
#    Using Unsplash CDN instead (returns 200 for any client) plus the Pipecat repo's own logo.
SAMPLE_IMAGES = {
    "pipecat": "https://raw.githubusercontent.com/pipecat-ai/pipecat/main/pipecat.png",       # Pipecat official logo
    "chart":   "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=512&q=80",     # green mountain landscape photo
    "diagram": "https://images.unsplash.com/photo-1610832958506-aa56368176cf?w=512&q=80",     # assorted fruits
}


class ImageInjector(FrameProcessor):
    """
    Listens for the user saying "show image" or "load image" and injects an image into the LLM context.

    Core pattern:
        1. LLMContext.create_image_url_message() creates a message containing the image
        2. LLMMessagesAppendFrame(messages=[image_msg], run_llm=True)
           → adds the image to the context and immediately triggers the LLM to generate a description
    """

    def __init__(self, context: LLMContext, tts, task_ref: list):
        super().__init__()
        self._context = context
        self._tts = tts
        self._task_ref = task_ref
        self._image_loaded = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.lower()

            # Load an image when the user says "show image" or "describe image"
            if ("show image" in text or "load image" in text or "describe image" in text):
                await self._load_image("chart", text)
                return

            # Load the Pipecat logo when the user says "show pipecat logo"
            elif ("pipecat" in text or "cat" in text) and ("logo" in text or "image" in text):
                await self._load_image("pipecat", text)
                return

            elif "fruit" in text or "food" in text:
                await self._load_image("diagram", text)
                return

        await self.push_frame(frame, direction)

    async def _load_image(self, image_key: str, user_text: str):
        url = SAMPLE_IMAGES.get(image_key, SAMPLE_IMAGES["chart"])

        # ── Core: add the image URL to the LLM context ──────────────────────
        # create_image_url_message() creates a multimodal message (image + text)
        image_message = LLMContext.create_image_url_message(
            url=url,
            text=f"The user said: '{user_text}'. Describe what you see in this image in 2-3 sentences.",
        )

        task = self._task_ref[0]
        if task:
            # LLMMessagesAppendFrame appends the image message to the context
            # run_llm=True → triggers the LLM to generate a reply immediately after appending
            await task.queue_frames([
                LLMMessagesAppendFrame(
                    messages=[image_message],
                    run_llm=True,
                )
            ])

        self._image_loaded = True
        print(f"\n[Multimodal] Loaded image: {url}")


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

    # Use a model that supports images (gpt-4o-mini also supports vision)
    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            model="gpt-4o-mini",  # vision-capable models: gpt-4o, gpt-4o-mini, gpt-4-vision
            system_instruction=(
                "You are a helpful voice assistant that can see and describe images. "
                "When shown an image, describe what you see naturally in conversational language. "
                "Keep responses short."
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

    task_ref = [None]
    image_injector = ImageInjector(context, tts, task_ref)

    pipeline = Pipeline([
        transport.input(),
        stt,
        image_injector,          # ← listens for user commands and injects images
        user_aggregator,
        llm,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    task = PipelineTask(pipeline, params=PipelineParams())
    task_ref[0] = task

    context.add_message({
        "role": "developer",
        "content": "Greet the user. Tell them they can say 'show image' to load an image for you to describe.",
    })
    await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=False if sys.platform == "win32" else True)

    print("=" * 55)
    print(" Multimodal Demo (Voice + Vision)")
    print(" Say 'show image'      → load a nature photo")
    print(" Say 'pipecat logo'    → load Pipecat logo")
    print(" Say 'fruit image'     → load a fruit photo")
    print(" Then ask questions about the image!")
    print("=" * 55)

    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
