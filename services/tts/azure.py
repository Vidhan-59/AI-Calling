"""Azure Neural TTS - specifically for Gujarati (gu-IN-DhwaniNeural), which
has no first-class Sarvam/ElevenLabs voice at the time this was written.
Uses the official `azure-cognitiveservices-speech` SDK, whose streaming API
is push/pull-stream based rather than an async generator - the bridge
below adapts it, but has not been run against a live Azure Speech resource
in this build."""

import asyncio
from typing import AsyncIterator

from config import settings
from services.tts.base import TTSProvider


class AzureTTSProvider(TTSProvider):
    name = "azure"

    def __init__(self) -> None:
        if not (settings.azure_speech_key and settings.azure_speech_region):
            raise RuntimeError(
                "TTS_PROVIDER=azure requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION."
            )
        try:
            import azure.cognitiveservices.speech as speechsdk  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "`azure-cognitiveservices-speech` is not installed. Run "
                "`pip install azure-cognitiveservices-speech` (only TTS_PROVIDER=azure needs it)."
            ) from e
        self._speechsdk = speechsdk
        self._speech_config = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key, region=settings.azure_speech_region
        )
        self._speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
        )

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        *,
        language: str,
        voice_id: str,
        speed: float = 1.0,
    ) -> AsyncIterator[bytes]:
        self._speech_config.speech_synthesis_voice_name = voice_id
        rate_pct = int((speed - 1.0) * 100)
        async for sentence in text_stream:
            if not sentence.strip():
                continue
            ssml = (
                f'<speak version="1.0" xml:lang="{language}">'
                f'<voice name="{voice_id}"><prosody rate="{rate_pct:+d}%">{sentence}</prosody></voice>'
                "</speak>"
            )
            synthesizer = self._speechsdk.SpeechSynthesizer(
                speech_config=self._speech_config, audio_config=None
            )
            result = await asyncio.to_thread(synthesizer.speak_ssml_async(ssml).get)
            if result.reason == self._speechsdk.ResultReason.SynthesizingAudioCompleted:
                yield result.audio_data
