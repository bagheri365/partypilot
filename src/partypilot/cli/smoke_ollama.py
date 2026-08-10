"""Small live Ollama smoke test for PartyPilot's provider integration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from partypilot.adapters import OllamaAdapter, UrllibHttpTransport
from partypilot.cli.eval_baseline import _ollama_config
from partypilot.ports.llm_provider import GenerationRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny live Ollama smoke test.")
    parser.add_argument("--base-url", default=None, help="Override PARTYPILOT_OLLAMA_BASE_URL.")
    parser.add_argument("--model", default=None, help="Override PARTYPILOT_OLLAMA_MODEL.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Override PARTYPILOT_OLLAMA_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override PARTYPILOT_OLLAMA_MAX_RETRIES.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = _ollama_config(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        adapter = OllamaAdapter(config, UrllibHttpTransport())
        response = adapter.generate(
            GenerationRequest(prompt="Reply with only: OK"),
            timeout_seconds=config.timeout_seconds,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: live Ollama smoke test failed. Details: {exc}", file=sys.stderr)
        return 1

    if not response.text.strip():
        print("ERROR: Ollama returned an empty response.", file=sys.stderr)
        return 1

    print("Ollama smoke test passed.")
    print(f"Model: {config.model}")
    print(f"Response: {response.text.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
