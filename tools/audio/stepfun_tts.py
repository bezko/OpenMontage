"""StepFun StepAudio TTS provider tool.

Generates speech from text using StepFun's StepAudio 2.5 TTS model.
OpenAI-compatible API at https://api.stepfun.ai/step_plan/v1
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


class StepFunTTS(BaseTool):
    name = "stepfun_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "stepfun"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set STEPFUN_API_KEY environment variable:\n"
        "  export STEPFUN_API_KEY=your_key_here\n"
        "Get a key at https://platform.stepfun.com/"
    )
    fallback = "piper_tts"
    fallback_tools = ["piper_tts", "openai_tts", "xiaomi_tts"]
    agent_skills = []

    capabilities = ["text_to_speech", "voice_selection"]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "cost-effective TTS (included in StepFun plan)",
        "Chinese and English narration",
        "API-based production",
    ]
    not_good_for = ["voice clone matching", "fully offline production"]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "description": "Text to synthesize"},
            "voice": {
                "type": "string",
                "default": "default",
                "description": "Voice name",
            },
            "model": {
                "type": "string",
                "default": "stepaudio-2.5-tts",
                "description": "StepAudio TTS model",
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
    side_effects = ["writes audio file to output_path", "calls StepFun API"]
    user_visible_verification = ["Listen to audio for clarity and naturalness"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("STEPFUN_API_KEY")

    def execute(self, inputs: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="STEPFUN_API_KEY is not set.",
            )

        text = inputs.get("text", "")
        if not text:
            return ToolResult(success=False, error="text is required")

        voice = inputs.get("voice", "default")
        model = inputs.get("model", "stepaudio-2.5-tts")
        fmt = inputs.get("format", "mp3")
        speed = inputs.get("speed", 1.0)
        output_path = inputs.get(
            "output_path", f"stepfun_tts_{int(time.time())}.{fmt}"
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": fmt,
            "speed": speed,
        }

        try:
            resp = requests.post(
                "https://api.stepfun.ai/step_plan/v1/audio/speech",
                headers=headers,
                json=payload,
                timeout=60,
            )

            if resp.status_code == 401:
                return ToolResult(success=False, error="StepFun auth failed. Check STEPFUN_API_KEY.")
            if resp.status_code == 429:
                return ToolResult(success=False, error="StepFun rate limit exceeded.")

            resp.raise_for_status()

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(resp.content)

        except requests.exceptions.HTTPError as e:
            return ToolResult(success=False, error=f"StepFun API error: {e}")
        except Exception as e:
            return ToolResult(success=False, error=f"StepFun TTS failed: {e}")

        return ToolResult(
            success=True,
            data={
                "provider": "stepfun",
                "model": model,
                "voice": voice,
                "output_path": str(out),
                "format": fmt,
                "text_length": len(text),
            },
        )
