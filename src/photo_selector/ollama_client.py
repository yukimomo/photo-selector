from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass
class OllamaClient:
	base_url: str
	timeout_seconds: int = 30
	max_retries: int = 2
	retry_backoff_seconds: float = 0.8

	def chat(self, model: str, image_b64: str, prompt: str) -> Dict[str, Any]:
		url = f"{self.base_url.rstrip('/')}/api/chat"
		payload = {
			"model": model,
			"stream": False,
			"messages": [
				{
					"role": "system",
					"content": "You are a photo selection assistant. Return only JSON.",
				},
				{
					"role": "user",
					"content": prompt,
					"images": [image_b64],
				},
			],
		}

		last_error: Exception | None = None
		for attempt in range(self.max_retries + 1):
			try:
				response = requests.post(url, json=payload, timeout=self.timeout_seconds)
				if response.status_code != 200:
					raise RuntimeError(
						f"Ollama HTTP {response.status_code}: {response.text[:200]}"
					)
				data = response.json()
				content = data.get("message", {}).get("content", "")
				if not content:
					raise RuntimeError("Empty Ollama response content")
				return _parse_json_from_text(content)
			except Exception as exc:  # noqa: BLE001
				last_error = exc
				if attempt < self.max_retries:
					time.sleep(self.retry_backoff_seconds * (attempt + 1))
					continue
				raise

		if last_error is not None:
			raise last_error
		raise RuntimeError("Unknown Ollama error")


def _parse_json_from_text(text: str) -> Dict[str, Any]:
	stripped = text.strip()
	if stripped.startswith("```"):
		stripped = stripped.strip("`")
		stripped = stripped.replace("json", "", 1).strip()

	if stripped.startswith("{") and stripped.endswith("}"):
		return json.loads(stripped)

	start = stripped.find("{")
	end = stripped.rfind("}")
	if start >= 0 and end > start:
		return json.loads(stripped[start : end + 1])

	raise ValueError("No JSON object found in Ollama response")
