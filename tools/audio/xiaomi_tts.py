"""Xiaomi MiMo TTS provider tool.

Generates speech from text using Xiaomi's MiMo V2.5 TTS models.
Includes support for voice cloning and voice design.
OpenAI-compatible API at https://token-plan-sgp.xiaomimimo.com/v1
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import requests

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)


class XiaomiTTS(BaseTool):
    name = "xiaomi_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "xiaomi"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set XIAOMI_API_KEY environment variable:\n"
        "  export XIAOMI_API_KEY=your_key_here\n"
        "Configured via OpenClaw xiaomi provider."
    )
    fallback = "piper_tts"
    fallback_tools = ["piper_tts", "stepfun_tts", "openai_tts"]
    agent_skills = []

    capabilities = [
        "text_to_speech",
        "voice_selection",
        "voice_cloning",
        "voice_design",
    ]
    supports = {
        "voice_cloning": True,
        "voice_design": True,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "voice cloning from reference audio",
        "custom voice design from text description",
        "Chinese and English narration",
        "cost-effective TTS (included in Xiaomi plan)",
    ]
    not_good_for = ["fully offline production"]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to synthesize"},
            "voice": {
                "type": "string",
                "default": "default",
                "description": "Voice name or ID",
            },
            "model": {
                "type": "string",
                "enum": [
                    "mimo-v2-tts",
                    "mimo-v2.5-tts",
                    "mimo-v2.5-tts-voiceclone",
                    "mimo-v2.5-tts-voicedesign",
                ],
                "default": "mimo-v2.5-tts",
                "description": "MiMo TTS model variant",
            },
            "format": {
                "type": "string",
                "enum": ["mp3", "wav", "opus"],
                "default": "mp3",
            },
            "speed": {
                "type": "number",
                "minimum": 0.25,
                "maximum": 4.0,
                "default": 1.0,
            },
            "reference_audio_path": {
                "type": "string",
                "description": "Reference audio for voice cloning (model=voiceclone)",
            },
            "voice_description": {
                "type": "string",
                "description": "Text description of desired voice (model=voicedesign)",
            },
            "output_path": {
                "type": "string",
                "description": "Local path to save audio",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=50, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["text", "voice", "model", "speed"]
    side_effects = ["writes audio file to output_path", "calls Xiaomi API"]
    user_visible_verification = ["Listen to audio for clarity and naturalness"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("XIAOMI_API_KEY")

    def execute(self, inputs: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="XIAOMI_API_KEY is not set.",
            )

        text = inputs.get("text", "")
        if not text:
            return ToolResult(success=False, error="text is required")

        voice = inputs.get("voice", "default")
        model = inputs.get("model", "mimo-v2.5-tts")
        fmt = inputs.get("format", "mp3")
        speed = inputs.get("speed", 1.0)
        ref_audio = inputs.get("reference_audio_path")
        voice_desc = inputs.get("voice_description")
        output_path = inputs.get(
            "output_path", f"xiaomi_tts_{int(time.time())}.{fmt}"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": fmt,
            "speed": speed,
        }

        if model == "mimo-v2.5-tts-voiceclone" and ref_audio:
            ref_path = Path(ref_audio)
            if not ref_path.exists():
                return ToolResult(success=False, error=f"Reference audio not found: {ref_audio}")
            import base64
            ref_b64 = base64.b64encode(ref_path.read_bytes()).decode()
            payload["reference_audio"] = f"data:audio/{ref_path.suffix[1:] or 'wav'};base64,{ref_b64}"

        if model == "mimo-v2.5-tts-voicedesign" and voice_desc:
            payload["voice_description"] = voice_desc

        try:
            resp = requests.post(
                "https://token-plan-sgp.xiaomimimo.com/v1/audio/speech",
                headers=headers,
                json=payload,
                timeout=60,
            )

            if resp.status_code == 401:
                return ToolResult(success=False, error="Xiaomi auth failed. Check XIAOMI_API_KEY.")
            if resp.status_code == 429:
                return ToolResult(success=False, error="Xiaomi rate limit exceeded.")

            resp.raise_for_status()

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(resp.content)

        except requests.exceptions.HTTPError as e:
            return ToolResult(success=False, error=f"Xiaomi API error: {e}")
        except Exception as e:
            return ToolResult(success=False, error=f"Xiaomi TTS failed: {e}")

        return ToolResult(
            success=True,
            data={
                "provider": "xiaomi",
                "model": model,
                "voice": voice,
                "output_path": str(out),
                "format": fmt,
                "text_length": len(text),
            },
        )
