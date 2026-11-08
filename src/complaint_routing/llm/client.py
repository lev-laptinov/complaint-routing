"""One call to the OpenAI-compatible LLM server, with its tokens and latency recorded.

The server address and the model name come from config/llm.yaml, never from code: the same
client talks to the local vLLM server or to an API by changing that file.
"""

import os
import time
from dataclasses import dataclass, field

import numpy as np
from openai import OpenAI


@dataclass
class Call:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class LLM:
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout_s: float
    api_key: str = "unused"
    calls: list[Call] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict, model: str | None = None) -> "LLM":
        c = cfg["llm"]
        return cls(
            base_url=c["base_url"],
            model=model or c["model"],
            temperature=c["temperature"],
            max_tokens=c["max_tokens"],
            timeout_s=c["timeout_s"],
            api_key=os.environ.get(c["api_key_env"], "unused"),
        )

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout_s)

    def complete(self, prompt: str, max_tokens: int | None = None) -> Call:
        start = time.perf_counter()
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        call = Call(
            text=(r.choices[0].message.content or "").strip(),
            prompt_tokens=r.usage.prompt_tokens,
            completion_tokens=r.usage.completion_tokens,
            latency_s=time.perf_counter() - start,
        )
        self.calls.append(call)
        return call


def usage_metrics(calls: list[Call], prefix: str) -> dict[str, float]:
    """Tokens, compute time and latency percentiles over the calls of one run (design 6.3, 6.7)."""
    lat = np.array([c.latency_s for c in calls])
    tin = sum(c.prompt_tokens for c in calls)
    tout = sum(c.completion_tokens for c in calls)
    return {
        f"{prefix}_calls": len(calls),
        f"{prefix}_tokens_in": tin,
        f"{prefix}_tokens_out": tout,
        f"{prefix}_tokens_in_per_call": tin / len(calls),
        f"{prefix}_tokens_out_per_call": tout / len(calls),
        f"{prefix}_compute_s": float(lat.sum()),
        f"{prefix}_latency_first_s": float(lat[0]),  # the first call after the server is up: cold
        f"{prefix}_latency_p50_s": float(np.percentile(lat, 50)),
        f"{prefix}_latency_p95_s": float(np.percentile(lat, 95)),
        f"{prefix}_latency_max_s": float(lat.max()),
    }
