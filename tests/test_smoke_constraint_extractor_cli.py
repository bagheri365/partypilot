from __future__ import annotations

from typing import cast

import pytest

from partypilot.adapters import LLMConstraintExtractor, OllamaConfig
from partypilot.cli import smoke_constraint_extractor as smoke_cli
from partypilot.ports.llm_provider import (
    FailingFakeLLMProvider,
    FakeLLMProvider,
    GenerationResponse,
    LLMProvider,
)


def _response(
    *,
    identifier: str,
    key: str,
    operator: str,
    value: str | int | bool,
    constraint_type: str = "HARD",
    description: str,
    derivation_explanation: str,
) -> GenerationResponse:
    return GenerationResponse(
        text="",
        structured_output={
            "constraints": [
                {
                    "identifier": identifier,
                    "key": key,
                    "operator": operator,
                    "value": value,
                    "constraint_type": constraint_type,
                    "description": description,
                    "confidence": 0.95,
                    "derivation_explanation": derivation_explanation,
                }
            ]
        },
    )


def _fake_config() -> OllamaConfig:
    return OllamaConfig(
        base_url="http://localhost:11434",
        model="fake-model",
        timeout_seconds=12.0,
        max_retries=2,
    )


def test_smoke_constraint_extractor_reports_all_representative_cases(
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = FakeLLMProvider(
        [
            _response(
                identifier="policy-family-allergen",
                key="allergen_policy",
                operator="eq",
                value="foods containing peanuts and tree nuts are prepared in a shared kitchen",
                description="Family Table allergen policy.",
                derivation_explanation="Directly stated allergen policy.",
            ),
            _response(
                identifier="policy-family-gluten-cross-contact",
                key="cross_contact_risk",
                operator="eq",
                value="present",
                description=(
                    "Food is prepared in a shared kitchen and is not certified free from "
                    "gluten cross-contact."
                ),
                derivation_explanation="Directly stated cross-contact risk.",
            ),
            _response(
                identifier="policy-family-vegan-notice",
                key="vegan_notice_days",
                operator="gte",
                value=7,
                description=(
                    "Vegan entree and dessert options require at least seven days of "
                    "advance notice."
                ),
                derivation_explanation="Directly stated advance-notice requirement.",
            ),
            _response(
                identifier="policy-loft-accessibility",
                key="accessible_restroom",
                operator="eq",
                value=True,
                description="Brooklyn Loft provides an accessible restroom.",
                derivation_explanation="Directly stated accessibility guidance.",
            ),
            _response(
                identifier="policy-adult-child-ratio",
                key="adult_child_ratio",
                operator="eq",
                value="1/5",
                description="One adult is required for every five children.",
                derivation_explanation="Directly stated supervision ratio.",
            ),
        ]
    )
    extractor = smoke_cli.build_live_constraint_extractor(provider=provider)

    lines = smoke_cli.smoke_constraint_extractor(extractor, smoke_cli.build_representative_cases())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(lines) == 5
    assert "allergen/shared-kitchen" in lines[0]
    assert "doc-family-allergen-current" in lines[0]
    assert "allergen_policy" in lines[0]
    assert "structured extraction succeeded: yes" in lines[0]
    assert "cross_contact_risk" in lines[1]
    assert "vegan_notice_days" in lines[2]
    assert "accessible_restroom" in lines[3]
    assert "adult_child_ratio" in lines[4]
    assert "source document ID: doc-craft-supervision-current" in lines[4]
    assert len(provider.requests) == 5
    assert provider.requests[0][0].structured_output is not None
    assert "Return only the typed JSON envelope requested by the schema" in (
        provider.requests[0][0].system_prompt or ""
    )


def test_main_reports_success_with_fake_live_extractor(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = FakeLLMProvider(
        [
            _response(
                identifier="policy-family-allergen",
                key="allergen_policy",
                operator="eq",
                value="foods containing peanuts and tree nuts are prepared in a shared kitchen",
                description="Family Table allergen policy.",
                derivation_explanation="Directly stated allergen policy.",
            ),
            _response(
                identifier="policy-family-gluten-cross-contact",
                key="cross_contact_risk",
                operator="eq",
                value="present",
                description=(
                    "Food is prepared in a shared kitchen and is not certified free from "
                    "gluten cross-contact."
                ),
                derivation_explanation="Directly stated cross-contact risk.",
            ),
            _response(
                identifier="policy-family-vegan-notice",
                key="vegan_notice_days",
                operator="gte",
                value=7,
                description=(
                    "Vegan entree and dessert options require at least seven days of "
                    "advance notice."
                ),
                derivation_explanation="Directly stated advance-notice requirement.",
            ),
            _response(
                identifier="policy-loft-accessibility",
                key="accessible_restroom",
                operator="eq",
                value=True,
                description="Brooklyn Loft provides an accessible restroom.",
                derivation_explanation="Directly stated accessibility guidance.",
            ),
            _response(
                identifier="policy-adult-child-ratio",
                key="adult_child_ratio",
                operator="eq",
                value="1/5",
                description="One adult is required for every five children.",
                derivation_explanation="Directly stated supervision ratio.",
            ),
        ]
    )
    extractor = LLMConstraintExtractor(provider)
    monkeypatch.setattr(
        smoke_cli,
        "_build_live_constraint_extractor_and_config",
        lambda **kwargs: (extractor, _fake_config(), None),
    )

    exit_code = smoke_cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Constraint extractor smoke test passed." in captured.out
    assert "Model: fake-model" in captured.out
    assert "doc-loft-accessibility-current" in captured.out
    assert "doc-craft-supervision-current" in captured.out


@pytest.mark.parametrize(
    "provider",
    [
        FailingFakeLLMProvider(RuntimeError("boom")),
        FakeLLMProvider([GenerationResponse(text="no structured output")]),
        FakeLLMProvider([GenerationResponse(text="", structured_output={"constraints": []})]),
        FakeLLMProvider(
            [
                GenerationResponse(
                    text="",
                    structured_output={
                        "constraints": [
                            {
                                "identifier": "policy-adult-child-ratio",
                                "key": "adult_child_ratio",
                                "operator": "eq",
                                "value": "1/5",
                                "constraint_type": "HARD",
                                "description": "One adult is required for every five children.",
                                "confidence": 0.95,
                                "derivation_explanation": "Directly stated supervision ratio.",
                                "outside_food_rules": "only from licensed caterers",
                            }
                        ]
                    },
                )
            ]
        ),
    ],
)
def test_main_returns_non_zero_for_provider_and_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    provider: object,
) -> None:
    extractor = LLMConstraintExtractor(cast(LLMProvider, provider))
    monkeypatch.setattr(
        smoke_cli,
        "_build_live_constraint_extractor_and_config",
        lambda **kwargs: (extractor, _fake_config(), None),
    )

    exit_code = smoke_cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "ERROR:" in captured.err
    assert "smoke test failed" in captured.err
