"""Speech-to-text factory shared by both live pipelines. One switch picks the
engine: USE_SARVAM_STT=false -> Sarvam saaras:v3 cloud STT, false -> local
faster-whisper on this machine's GPU (services/voice_pipeline/local_stt.py)."""

from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.transcriptions.language import Language

from config import settings

_STT_LANGUAGE = {"hi": Language.HI_IN, "gu": Language.GU_IN, "en": Language.EN_IN}
STT_SAMPLE_RATE = 16000


class _BackgroundConnectSarvamSTT(SarvamSTTService):
    """Connects to Sarvam without blocking pipeline start. The stock start()
    awaits the websocket connect (~0.6s measured) before StartFrame moves on,
    which delayed the opening line by that much - yet the caller is only
    listening during the opening. Audio arriving before the connection is up
    is dropped by run_stt (it checks for the socket), which is harmless here."""

    async def start(self, frame):
        await super(SarvamSTTService, self).start(frame)
        self.create_task(self._connect(), "sarvam_connect")


def build_stt(language: str):
    """Sarvam cloud STT when USE_SARVAM_STT=false, else local faster-whisper."""
    if settings.stt_engine == "local":
        from services.voice_pipeline.local_stt import LocalWhisperSTT

        return LocalWhisperSTT(
            source=settings.local_stt_model_for(language),
            compute_type=settings.local_stt_compute_type,
            language=language,
            beam_size=settings.local_stt_beam_size,
            isolation=settings.local_stt_isolation,
            sample_rate=STT_SAMPLE_RATE,
        )
    return _BackgroundConnectSarvamSTT(
        api_key=settings.sarvam_api_key,
        sample_rate=STT_SAMPLE_RATE,
        settings=SarvamSTTService.Settings(
            model=settings.sarvam_stt_model,
            language=_STT_LANGUAGE.get(language, Language.HI_IN),
        ),
    )
