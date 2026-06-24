"""Qwen image generation via Alibaba Cloud DashScope API."""

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


class QwenImage(BaseTool):
    name = "qwen_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "qwen"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = []
    install_instructions = (
        "Set DASHSCOPE_API_KEY to your Alibaba Cloud DashScope API key.\n"
        "  export DASHSCOPE_API_KEY=your_key_here\n"
        "Get a key at https://dashscope.aliyun.com/"
    )
    agent_skills = ["flux-best-practices"]

    capabilities = ["generate_image", "text_to_image", "image_variation"]
    supports = {
        "complex_instructions": True,
        "text_in_image": True,
        "multiple_outputs": True,
        "style_control": True,
    }
    best_for = [
        "Chinese and multilingual prompts",
        "cost-effective high-quality generation",
        "style-controlled outputs",
    ]
    not_good_for = ["offline generation", "budget-constrained projects at high quality"]
    fallback_tools = ["flux_image", "google_imagen", "openai_image"]
    quality_score = 0.8

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string", "description": "Text description of the desired image"},
            "model": {
                "type": "string",
                "enum": ["wanx2.1-t2i-turbo", "wanx2.1-t2i-plus"],
                "default": "wanx2.1-t2i-turbo",
                "description": "Wanx model variant (turbo=faster, plus=higher quality)",
            },
            "n": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
                "default": 1,
                "description": "Number of images to generate",
            },
            "size": {
                "type": "string",
                "enum": ["1024*1024", "720*1280", "1280*720", "1440*720", "720*1440"],
                "default": "1024*1024",
                "description": "Image resolution",
            },
            "style": {
                "type": "string",
                "enum": ["<auto>", "<photography>", "<portrait>", "<3d cartoon>", "<anime>", "<oil painting>", "<watercolor>", "<sketch>", "<chinese painting>", "<flat illustration>"],
                "default": "<auto>",
                "description": "Visual style preset",
            },
            "output_path": {
                "type": "string",
                "description": "Local path to save the generated image(s)",
            },
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=200, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "model", "size", "style", "n"]
    side_effects = ["writes image file(s) to output_path", "calls DashScope API"]
    user_visible_verification = ["Check generated image matches prompt and style"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("DASHSCOPE_API_KEY")

    def execute(self, inputs: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(
                success=False,
                error="DASHSCOPE_API_KEY is not set. Set it to your DashScope API key.",
            )

        prompt = inputs.get("prompt", "")
        if not prompt:
            return ToolResult(success=False, error="prompt is required")

        model = inputs.get("model", "wanx2.1-t2i-turbo")
        n = inputs.get("n", 1)
        size = inputs.get("size", "1024*1024")
        style = inputs.get("style", "<auto>")
        output_path = inputs.get("output_path")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }

        payload = {
            "model": model,
            "input": {
                "prompt": prompt,
            },
            "parameters": {
                "size": size,
                "n": n,
                "style": style,
            },
        }

        try:
            submit_resp = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if submit_resp.status_code == 401:
                return ToolResult(
                    success=False,
                    error="DashScope authentication failed. Check your DASHSCOPE_API_KEY.",
                )

            if submit_resp.status_code == 429:
                return ToolResult(
                    success=False,
                    error="DashScope rate limit exceeded. Please retry later.",
                )

            submit_resp.raise_for_status()
            task_data = submit_resp.json()
            task_id = task_data.get("output", {}).get("task_id")

            if not task_id:
                return ToolResult(
                    success=False,
                    error=f"DashScope submission returned unexpected response: {task_data}",
                )

            poll_url = f"https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
            max_wait = 300
            poll_interval = 3
            elapsed = 0

            while elapsed < max_wait:
                time.sleep(poll_interval)
                elapsed += poll_interval

                status_resp = requests.get(poll_url, headers=headers, timeout=15)
                status_resp.raise_for_status()
                status_data = status_resp.json()
                task_status = status_data.get("output", {}).get("task_status", "UNKNOWN").upper()

                if task_status == "SUCCEEDED":
                    break
                if task_status in ("FAILED", "CANCELED"):
                    error_msg = status_data.get("output", {}).get("message", "Unknown error")
                    return ToolResult(
                        success=False,
                        error=f"DashScope image generation {task_status.lower()}: {error_msg}",
                    )
            else:
                return ToolResult(
                    success=False,
                    error=f"DashScope image generation timed out after {max_wait}s",
                )

            task_results = status_data.get("output", {}).get("results", [])
            if not task_results:
                return ToolResult(
                    success=False,
                    error=f"No images in DashScope response: {status_data}",
                )

            saved_paths = []
            for idx, result in enumerate(task_results):
                img_url = result.get("url")
                if not img_url:
                    continue

                img_resp = requests.get(img_url, timeout=60)
                img_resp.raise_for_status()

                if output_path:
                    if n > 1:
                        base, ext = os.path.splitext(output_path)
                        path = Path(f"{base}_{idx}{ext}")
                    else:
                        path = Path(output_path)
                else:
                    ext = ".png"
                    path = Path(f"qwen_{model}_{int(time.time())}_{idx}{ext}")

                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(img_resp.content)
                saved_paths.append(str(path))

        except requests.exceptions.HTTPError as e:
            return ToolResult(success=False, error=f"DashScope API error: {e}")
        except Exception as e:
            return ToolResult(success=False, error=f"Qwen image generation failed: {e}")

        return ToolResult(
            success=True,
            data={
                "provider": "qwen-dashscope",
                "model": model,
                "prompt": prompt,
                "output_paths": saved_paths,
                "style": style,
                "size": size,
                "n": len(saved_paths),
            },
        )
