"""StepFun Step Image Edit 2 provider tool.

Edits/generates images using StepFun's step-image-edit-2 model.
OpenAI-compatible API at https://api.stepfun.ai/step_plan/v1
Accepts text + image input, returns edited image.
"""

from __future__ import annotations

import base64
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


class StepFunImage(BaseTool):
    name = "stepfun_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_editing"
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
    agent_skills = ["flux-best-practices"]

    capabilities = ["image_editing", "text_guided_edit"]
    supports = {
        "complex_instructions": True,
        "text_in_image": True,
        "image_to_image": True,
    }
    best_for = [
        "instruction-based image editing",
        "Chinese and English prompts",
        "cost-effective image manipulation (included in plan)",
    ]
    not_good_for = ["text-to-image from scratch", "offline generation"]
    fallback_tools = ["openai_image", "flux_image"]
    quality_score = 0.75

    input_schema = {
        "type": "object",
        "required": ["prompt", "image_path"],
        "properties": {
            "prompt": {"type": "string", "description": "Edit instruction"},
            "image_path": {"type": "string", "description": "Path to source image"},
            "model": {
                "type": "string",
                "default": "step-image-edit-2",
                "description": "StepFun image model",
            },
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1792x1024", "1024x1792"],
                "default": "1024x1024",
            },
            "output_path": {"type": "string", "description": "Path to save result"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=200, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["rate_limit", "timeout"])
    idempotency_key_fields = ["prompt", "image_path", "model", "size"]
    side_effects = ["writes image file to output_path", "calls StepFun API"]
    user_visible_verification = ["Check edited image matches instructions"]

    def _get_api_key(self) -> str | None:
        return os.environ.get("STEPFUN_API_KEY")

    def execute(self, inputs: dict[str, Any], runtime: ToolRuntime) -> ToolResult:
        api_key = self._get_api_key()
        if not api_key:
            return ToolResult(success=False, error="STEPFUN_API_KEY is not set.")

        prompt = inputs.get("prompt", "")
        image_path = inputs.get("image_path", "")
        if not prompt or not image_path:
            return ToolResult(success=False, error="prompt and image_path are required")

        model = inputs.get("model", "step-image-edit-2")
        size = inputs.get("size", "1024x1024")
        output_path = inputs.get("output_path", f"stepfun_edit_{int(time.time())}.png")

        img_path = Path(image_path)
        if not img_path.exists():
            return ToolResult(success=False, error=f"Image not found: {image_path}")

        img_b64 = base64.b64encode(img_path.read_bytes()).decode()

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "prompt": prompt,
            "image": f"data:image/{img_path.suffix[1:] or 'png'};base64,{img_b64}",
            "size": size,
            "n": 1,
        }

        try:
            resp = requests.post(
                "https://api.stepfun.ai/step_plan/v1/images/edits",
                headers=headers,
                json=payload,
                timeout=120,
            )

            if resp.status_code == 401:
                return ToolResult(success=False, error="StepFun auth failed.")
            if resp.status_code == 429:
                return ToolResult(success=False, error="StepFun rate limit exceeded.")

            resp.raise_for_status()
            data = resp.json()

            img_url = None
            if "data" in data and data["data"]:
                img_url = data["data"][0].get("url") or data["data"][0].get("b64_json")

            if not img_url:
                return ToolResult(success=False, error=f"No image in response: {data}")

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            if img_url.startswith("http"):
                img_resp = requests.get(img_url, timeout=60)
                img_resp.raise_for_status()
                out.write_bytes(img_resp.content)
            else:
                out.write_bytes(base64.b64decode(img_url))

        except requests.exceptions.HTTPError as e:
            return ToolResult(success=False, error=f"StepFun API error: {e}")
        except Exception as e:
            return ToolResult(success=False, error=f"StepFun image edit failed: {e}")

        return ToolResult(
            success=True,
            data={
                "provider": "stepfun",
                "model": model,
                "prompt": prompt,
                "output_path": str(out),
                "size": size,
            },
        )
