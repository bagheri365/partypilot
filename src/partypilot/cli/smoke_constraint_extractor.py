"""Small live Ollama smoke test for PartyPilot's LLM constraint extraction contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from partypilot.adapters import (
    HttpResponse,
    HttpTransport,
    LLMConstraintExtractor,
    OllamaAdapter,
    OllamaConfig,
    UrllibHttpTransport,
)
from partypilot.cli.eval_baseline import _ollama_config
from partypilot.domain.evidence_corpus import (
    EvidenceDocumentMetadata,
    EvidenceDocumentStatus,
    EvidenceDocumentType,
)
from partypilot.domain.party_request import PartyRequest
from partypilot.ports.constraint_extractor import (
    ConstraintExtractionContext,
    ConstraintExtractionInput,
    ConstraintExtractionResult,
)
from partypilot.ports.llm_provider import LLMProvider


def _smoke_request() -> PartyRequest:
    return PartyRequest(
        location="Boston",
        event_date=date(2026, 9, 1),
        guest_count=24,
        child_age=8,
        total_budget=Decimal("1200.00"),
    )


@dataclass(frozen=True, slots=True)
class SmokeConstraintCase:
    label: str
    evidence_text: str
    document_id: str
    resource_id: str
    document_type: EvidenceDocumentType
    version: str
    effective_date: date
    chunk_id: str

    def to_input(self) -> ConstraintExtractionInput:
        return ConstraintExtractionInput(
            evidence_text=self.evidence_text,
            evidence_metadata=EvidenceDocumentMetadata(
                document_id=self.document_id,
                resource_id=self.resource_id,
                document_type=self.document_type,
                version=self.version,
                effective_date=self.effective_date,
                status=EvidenceDocumentStatus.CURRENT,
            ),
            chunk_id=self.chunk_id,
            planning_context=ConstraintExtractionContext(
                request=_smoke_request(),
                resource_id=self.resource_id,
            ),
        )


@dataclass(frozen=True, slots=True)
class _TracingHttpTransport(HttpTransport):
    delegate: HttpTransport
    last_url: str | None = None
    last_payload_text: str | None = None
    last_payload_json: object | None = None
    last_response_status: int | None = None
    last_response_body_text: str | None = None

    def post_json(self, url: str, payload: bytes, *, timeout_seconds: float) -> HttpResponse:
        payload_text = payload.decode("utf-8", errors="replace")
        object.__setattr__(self, "last_url", url)
        object.__setattr__(self, "last_payload_text", payload_text)
        try:
            object.__setattr__(self, "last_payload_json", json.loads(payload_text))
        except json.JSONDecodeError:
            object.__setattr__(self, "last_payload_json", None)
        response = self.delegate.post_json(url, payload, timeout_seconds=timeout_seconds)
        object.__setattr__(self, "last_response_status", response.status_code)
        object.__setattr__(
            self, "last_response_body_text", response.body.decode("utf-8", errors="replace")
        )
        return response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a tiny live Ollama smoke test for constraint extraction."
    )
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
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Only run fixtures with this label; may be repeated.",
    )
    return parser


def build_representative_cases() -> tuple[SmokeConstraintCase, ...]:
    return (
        SmokeConstraintCase(
            label="allergen/shared-kitchen",
            evidence_text=(
                "Family Table peanut and tree nut allergen policy: foods containing peanuts "
                "and tree nuts are prepared in a shared kitchen. The caterer cannot guarantee "
                "an allergen-free meal."
            ),
            document_id="doc-family-allergen-current",
            resource_id="caterer-family-table",
            document_type=EvidenceDocumentType.ALLERGEN_POLICY,
            version="5.0",
            effective_date=date(2026, 2, 10),
            chunk_id="doc-family-allergen-current#chunk-1",
        ),
        SmokeConstraintCase(
            label="gluten-free/shared-kitchen",
            evidence_text=(
                "Family Table offers gluten-free menu selections, but food is prepared in a "
                "shared kitchen and is not certified free from gluten cross-contact."
            ),
            document_id="doc-family-gluten-current",
            resource_id="caterer-family-table",
            document_type=EvidenceDocumentType.ALLERGEN_POLICY,
            version="2.0",
            effective_date=date(2026, 2, 10),
            chunk_id="doc-family-gluten-current#chunk-1",
        ),
        SmokeConstraintCase(
            label="vegan/advance-notice",
            evidence_text=(
                "Family Table menu includes vegan entree and dessert options when requested "
                "at least seven days in advance."
            ),
            document_id="doc-family-vegan-current",
            resource_id="caterer-family-table",
            document_type=EvidenceDocumentType.VENUE_POLICY,
            version="1.0",
            effective_date=date(2026, 1, 1),
            chunk_id="doc-family-vegan-current#chunk-1",
        ),
        SmokeConstraintCase(
            label="accessibility/guidance",
            evidence_text=(
                "Brooklyn Loft provides step-free wheelchair access and an accessible "
                "restroom. Contact staff in advance for seating layout assistance."
            ),
            document_id="doc-loft-accessibility-current",
            resource_id="venue-brooklyn-loft",
            document_type=EvidenceDocumentType.ACCESSIBILITY_GUIDANCE,
            version="2.1",
            effective_date=date(2026, 3, 1),
            chunk_id="doc-loft-accessibility-current#chunk-1",
        ),
        SmokeConstraintCase(
            label="supervision/ratio",
            evidence_text=(
                "Craft Party requires one supervising adult for every five children under age 12."
            ),
            document_id="doc-craft-supervision-current",
            resource_id="activity-craft-party",
            document_type=EvidenceDocumentType.SUPERVISION_REQUIREMENTS,
            version="1.0",
            effective_date=date(2026, 2, 15),
            chunk_id="doc-craft-supervision-current#chunk-1",
        ),
    )


def _build_live_constraint_extractor_and_config(
    *,
    model: str | None,
    base_url: str | None,
    timeout_seconds: float | None,
    max_retries: int | None,
) -> tuple[LLMConstraintExtractor, OllamaConfig, _TracingHttpTransport]:
    config = _ollama_config(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    transport = _TracingHttpTransport(UrllibHttpTransport())
    extractor = LLMConstraintExtractor(OllamaAdapter(config, transport))
    return extractor, config, transport


def build_live_constraint_extractor(
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> LLMConstraintExtractor:
    if provider is not None:
        return LLMConstraintExtractor(provider)

    extractor, _, _ = _build_live_constraint_extractor_and_config(
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    return extractor


def _summarize_result(case: SmokeConstraintCase, result: ConstraintExtractionResult) -> str:
    constraints = result.constraints
    keys = ", ".join(item.constraint.key for item in constraints) if constraints else "n/a"
    operators = (
        ", ".join(item.constraint.operator.value for item in constraints) if constraints else "n/a"
    )
    values = ", ".join(str(item.constraint.value) for item in constraints) if constraints else "n/a"
    source_document_id = (
        ", ".join(sorted({item.provenance.source_document_id or "n/a" for item in constraints}))
        if constraints
        else case.document_id
    )
    success = "yes" if constraints else "no"
    return (
        f"- {case.label} | document/type: {case.document_id} / {case.document_type.value} | "
        f"constraints: {len(constraints)} | keys: {keys} | operators: {operators} | "
        f"values: {values} | source document ID: {source_document_id} | "
        f"structured extraction succeeded: {success}"
    )


def smoke_constraint_extractor(
    extractor: LLMConstraintExtractor,
    cases: Sequence[SmokeConstraintCase],
) -> tuple[str, ...]:
    lines: list[str] = []
    for case in cases:
        result = extractor.extract(case.to_input())
        if not result.constraints:
            raise RuntimeError(
                f"{case.label} produced no typed constraints despite explicit policy evidence"
            )
        lines.append(_summarize_result(case, result))
    return tuple(lines)


def _print_failure_diagnostics(
    *,
    case: SmokeConstraintCase,
    model_name: str,
    transport: _TracingHttpTransport | None,
    validated_result: ConstraintExtractionResult | None,
    failure_stage: str,
    failure_message: str,
) -> None:
    print("Smoke fixture diagnostics:", file=sys.stderr)
    print(f"  fixture label: {case.label}", file=sys.stderr)
    print(f"  model name: {model_name}", file=sys.stderr)
    print("  exact evidence text:", file=sys.stderr)
    print(f"    {case.evidence_text}", file=sys.stderr)
    print(f"  failure stage: {failure_stage}", file=sys.stderr)
    print(f"  failure message: {failure_message}", file=sys.stderr)
    if transport is not None and transport.last_payload_json is not None:
        print("  extractor request:", file=sys.stderr)
        payload = transport.last_payload_json
        if isinstance(payload, dict):
            print(f"    payload keys: {sorted(payload.keys())}", file=sys.stderr)
            print(f"    format sent: {payload.get('format')!r}", file=sys.stderr)
            print(f"    model sent: {payload.get('model')!r}", file=sys.stderr)
            print(f"    system prompt: {payload.get('system')}", file=sys.stderr)
            print(
                "    json schema sent to ollama: no (adapter only toggles format=json)",
                file=sys.stderr,
            )
        print(f"    raw payload: {transport.last_payload_text}", file=sys.stderr)
        print(f"    url: {transport.last_url}", file=sys.stderr)
        print(f"    response status: {transport.last_response_status}", file=sys.stderr)
        print(f"    raw response body: {transport.last_response_body_text}", file=sys.stderr)
        response_text = None
        response_json = None
        if transport.last_response_body_text is not None:
            try:
                response_json = json.loads(transport.last_response_body_text)
            except json.JSONDecodeError:
                response_json = None
        if isinstance(response_json, dict):
            response_text = response_json.get("response")
            print("  raw provider response:", file=sys.stderr)
            print(f"    text: {response_text!r}", file=sys.stderr)
            structured_output = None
            if isinstance(response_text, str):
                try:
                    structured_output = json.loads(response_text)
                except json.JSONDecodeError:
                    structured_output = None
            print(f"    structured_output: {structured_output!r}", file=sys.stderr)
            if isinstance(structured_output, dict):
                has_constraints = "constraints" in structured_output
                print(
                    f"    structured_output has constraints key: {has_constraints}",
                    file=sys.stderr,
                )
    if validated_result is not None:
        print("  validated extraction result:", file=sys.stderr)
        print(f"    constraint count: {len(validated_result.constraints)}", file=sys.stderr)
        keys = [item.constraint.key for item in validated_result.constraints]
        print(f"    constraint keys: {keys}", file=sys.stderr)
    else:
        print("  validated extraction result: none", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    label_filters = tuple(args.label or ())

    try:
        built = _build_live_constraint_extractor_and_config(
            model=args.model,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
        if len(built) == 2:
            extractor, config = built
            transport = None
        else:
            extractor, config, transport = built
        cases = build_representative_cases()
        if label_filters:
            cases = tuple(case for case in cases if case.label in label_filters)
        lines: list[str] = []
        for case in cases:
            request = case.to_input()
            validated_result: ConstraintExtractionResult | None = None
            try:
                validated_result = extractor.extract(request)
                if not validated_result.constraints:
                    raise RuntimeError(
                        f"{case.label} produced no typed constraints despite explicit policy "
                        "evidence"
                    )
                lines.append(_summarize_result(case, validated_result))
            except Exception as exc:
                failure_stage = type(exc).__name__
                failure_message = str(exc)
                _print_failure_diagnostics(
                    case=case,
                    model_name=config.model,
                    transport=transport,
                    validated_result=validated_result,
                    failure_stage=failure_stage,
                    failure_message=failure_message,
                )
                print(
                    f"ERROR: live constraint extraction smoke test failed. Details: {exc}",
                    file=sys.stderr,
                )
                return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Constraint extractor smoke test passed.")
    print(f"Model: {config.model}")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
