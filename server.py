#!/usr/bin/env python3
"""HTTP server for Irodori-TTS: the Gradio UI at ``/`` and a REST API at ``/api``.

Both share the process-wide cached runtime, so the model is loaded only once.

    uv run --no-sync python server.py --host 0.0.0.0 --port 7860

    curl -X POST http://HOST:7860/api/tts -H "Content-Type: application/json" \\
        -d '{"text": "こんにちは"}' -o out.wav
"""

from __future__ import annotations

import argparse
import io
import secrets
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import gradio as gr
import soundfile as sf
import torch
import uvicorn
from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from gradio_app import (
    _build_runtime_key,
    _default_checkpoint,
    _default_codec_device,
    _default_model_device,
    build_ui,
)
from irodori_tts import length_check
from irodori_tts.gradio_emoji_palette import EMOJI_PALETTE_CSS
from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    SamplingResult,
    get_cached_runtime,
    list_available_runtime_precisions,
)


class TTSParams(BaseModel):
    """Synthesis options. Defaults match the Gradio UI."""

    text: str = Field(min_length=1)
    seed: int | None = None
    num_steps: int | None = Field(default=None, ge=1)
    seconds: float | None = Field(default=None, gt=0)
    duration_scale: float = Field(default=1.0, gt=0)
    t_schedule_mode: Literal["linear", "sway"] = "linear"
    sway_coeff: float = -1.0
    cfg_guidance_mode: Literal["independent", "joint", "alternating"] = "independent"
    cfg_scale_text: float = 3.0
    cfg_scale_speaker: float = 5.0
    cfg_scale: float | None = None
    cfg_min_t: float = 0.5
    cfg_max_t: float = 1.0
    context_kv_cache: bool = True
    truncation_factor: float | None = None
    rescale_k: float | None = None
    rescale_sigma: float | None = None
    speaker_kv_scale: float | None = None
    speaker_kv_min_t: float | None = 0.9
    speaker_kv_max_layers: int | None = None
    speaker_embedding: str | None = Field(
        default=None,
        description="File name of a .speaker.safetensors under the server's --speaker-dir.",
    )
    lora_adapter: str | None = Field(
        default=None, description="Adapter directory name under the server's --lora-dir."
    )
    length_check: bool = False


class TTSRefParams(TTSParams):
    """Multipart form: TTSParams fields plus reference clips (concatenated in order)."""

    ref_audio: list[UploadFile]


@dataclass(frozen=True)
class AssetDirs:
    """Server-configured directories; clients may only name entries inside them."""

    speaker_dir: Path | None
    lora_dir: Path | None


def _resolve_asset(base: Path | None, name: str | None, label: str, *, is_dir: bool) -> str | None:
    if name is None:
        return None
    if base is None:
        raise ValueError(
            f"{label} is disabled on this server (start it with --{label.replace('_', '-')}-dir)."
        )
    root = base.resolve()
    target = (root / name).resolve()
    if target.parent != root:
        raise ValueError(f"{label} must be a plain name directly under the configured directory.")
    if is_dir and not target.is_dir():
        raise ValueError(f"{label} not found: {name}")
    if not is_dir and not (target.is_file() and target.name.endswith(".speaker.safetensors")):
        raise ValueError(f"{label} must be an existing .speaker.safetensors file: {name}")
    return str(target)


def _build_request(params: TTSParams, ref_wavs: list[str], dirs: AssetDirs) -> SamplingRequest:
    speaker_embedding = _resolve_asset(
        dirs.speaker_dir, params.speaker_embedding, "speaker", is_dir=False
    )
    lora_adapter = _resolve_asset(dirs.lora_dir, params.lora_adapter, "lora", is_dir=True)
    if ref_wavs and speaker_embedding is not None:
        raise ValueError("Reference audio and speaker_embedding are mutually exclusive.")
    return SamplingRequest(
        text=params.text,
        ref_wavs=ref_wavs or None,
        ref_embed=speaker_embedding,
        no_ref=not ref_wavs and speaker_embedding is None,
        ref_normalize_db=-16.0,
        ref_ensure_max=True,
        num_candidates=1,
        decode_mode="sequential",
        seconds=params.seconds,
        duration_scale=params.duration_scale,
        num_steps=params.num_steps,
        seed=params.seed,
        cfg_guidance_mode=params.cfg_guidance_mode,
        cfg_scale_text=params.cfg_scale_text,
        cfg_scale_speaker=params.cfg_scale_speaker,
        cfg_scale=params.cfg_scale,
        cfg_min_t=params.cfg_min_t,
        cfg_max_t=params.cfg_max_t,
        truncation_factor=params.truncation_factor,
        rescale_k=params.rescale_k,
        rescale_sigma=params.rescale_sigma,
        context_kv_cache=params.context_kv_cache,
        speaker_kv_scale=params.speaker_kv_scale,
        speaker_kv_min_t=params.speaker_kv_min_t,
        speaker_kv_max_layers=params.speaker_kv_max_layers,
        t_schedule_mode=params.t_schedule_mode,
        sway_coeff=params.sway_coeff,
        trim_tail=True,
        lora_adapter=lora_adapter,
    )


def _encode_wav(audio: torch.Tensor, sample_rate: int) -> bytes:
    audio_cpu = audio.detach().to(device="cpu", dtype=torch.float32)
    audio_np = audio_cpu.squeeze(0).numpy() if audio_cpu.shape[0] == 1 else audio_cpu.T.numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _synthesize(
    runtime_key: RuntimeKey, dirs: AssetDirs, params: TTSParams, ref_wavs: list[str]
) -> Response:
    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        request = _build_request(params, ref_wavs, dirs)
        runtime, reloaded = get_cached_runtime(runtime_key)
        log(f"[api] runtime: {'reloaded' if reloaded else 'reused'} refs={len(ref_wavs)}")
        result: SamplingResult
        if params.length_check:
            result, _info = length_check.synthesize(runtime, request, log_fn=log)
        else:
            result = runtime.synthesize(request, log_fn=log)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return Response(
        content=_encode_wav(result.audios[0], result.sample_rate),
        media_type="audio/wav",
        headers={
            "X-Seed": str(result.used_seed),
            "X-Total-To-Decode-Sec": f"{result.total_to_decode:.3f}",
        },
    )


@contextmanager
def _saved_uploads(files: list[UploadFile]) -> Iterator[list[str]]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="irodori_ref_"))
    try:
        paths: list[str] = []
        for i, upload in enumerate(files):
            suffix = Path(upload.filename or "").suffix or ".wav"
            path = tmp_dir / f"ref_{i:03d}{suffix}"
            with path.open("wb") as out:
                shutil.copyfileobj(upload.file, out)
            paths.append(str(path))
        yield paths
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _api_key_dependency(api_key: str | None):
    def verify(
        x_api_key: Annotated[str | None, Header()] = None,
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if api_key is None:
            return
        bearer = authorization.removeprefix("Bearer ").strip() if authorization else None
        supplied = x_api_key or bearer
        if supplied is None or not secrets.compare_digest(supplied, api_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    return verify


def create_app(
    runtime_key: RuntimeKey, dirs: AssetDirs, *, api_key: str | None, with_ui: bool
) -> FastAPI:
    app = FastAPI(title="Irodori-TTS API")
    auth = [Depends(_api_key_dependency(api_key))]

    @app.get("/api/health", dependencies=auth)
    def health() -> dict[str, str]:
        return {"status": "ok", "checkpoint": runtime_key.checkpoint}

    @app.post("/api/tts", dependencies=auth, response_class=Response)
    def tts(params: TTSParams) -> Response:
        """Synthesize speech without a reference voice (or with a server-side speaker embedding)."""
        return _synthesize(runtime_key, dirs, params, [])

    @app.post("/api/tts/ref", dependencies=auth, response_class=Response)
    def tts_with_reference(form: Annotated[TTSRefParams, Form()]) -> Response:
        """Synthesize speech cloning the voice of the uploaded reference audio."""
        params = TTSParams.model_validate(form.model_dump(exclude={"ref_audio"}))
        with _saved_uploads(form.ref_audio) as ref_wavs:
            return _synthesize(runtime_key, dirs, params, ref_wavs)

    if with_ui:
        demo = build_ui()
        demo.queue(default_concurrency_limit=1)
        app = gr.mount_gradio_app(app, demo, path="/", css=EMOJI_PALETTE_CSS)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Irodori-TTS server (Gradio UI + REST API).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--checkpoint", default=None, help="API model (.pt/.safetensors or HF repo id)."
    )
    parser.add_argument("--model-device", default=None)
    parser.add_argument("--model-precision", default=None)
    parser.add_argument("--codec-device", default=None)
    parser.add_argument("--codec-precision", default=None)
    parser.add_argument(
        "--api-key", default=None, help="Require X-API-Key or Bearer token on /api/*."
    )
    parser.add_argument(
        "--speaker-dir",
        type=Path,
        default=None,
        help="Enables speaker_embedding (names in this dir).",
    )
    parser.add_argument(
        "--lora-dir", type=Path, default=None, help="Enables lora_adapter (names in this dir)."
    )
    parser.add_argument("--no-ui", action="store_true", help="Serve only the REST API.")
    parser.add_argument("--preload", action="store_true", help="Load the model before serving.")
    args = parser.parse_args()

    model_device = args.model_device or _default_model_device()
    codec_device = args.codec_device or _default_codec_device()
    # Same defaults as the UI so that both reuse one cached runtime.
    runtime_key = _build_runtime_key(
        checkpoint=args.checkpoint or _default_checkpoint(),
        model_device=model_device,
        model_precision=args.model_precision or list_available_runtime_precisions(model_device)[0],
        codec_device=codec_device,
        codec_precision=args.codec_precision or list_available_runtime_precisions(codec_device)[0],
    )
    print(f"[api] runtime key: {runtime_key}", flush=True)
    if args.preload:
        get_cached_runtime(runtime_key)

    dirs = AssetDirs(speaker_dir=args.speaker_dir, lora_dir=args.lora_dir)
    app = create_app(runtime_key, dirs, api_key=args.api_key, with_ui=not args.no_ui)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
