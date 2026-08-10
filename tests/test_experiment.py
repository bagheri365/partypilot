from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from partypilot.domain.experiment import ExperimentConfig, ExperimentResultMetadata


def test_experiment_config_captures_traceability_fields() -> None:
    timestamp = datetime(2026, 8, 9, 18, 45, tzinfo=UTC)
    config = ExperimentConfig(
        experiment_id="exp-001",
        code_commit_sha="abc1234",
        dataset_version="v0.1",
        architecture_variant="deterministic_baseline",
        model_provider="ollama",
        model_name="example-model",
        model_version="1",
        prompt_version="p1",
        retrieval_configuration={"top_k": 5, "rerank": False},
        random_seed=42,
        timestamp=timestamp,
    )

    assert config.experiment_id == "exp-001"
    assert config.code_commit_sha == "abc1234"
    assert config.dataset_version == "v0.1"
    assert config.retrieval_configuration == {"top_k": 5, "rerank": False}
    assert config.random_seed == 42
    assert config.timestamp == timestamp


def test_optional_provider_fields_support_non_llm_experiment() -> None:
    config = ExperimentConfig(
        experiment_id="exp-deterministic",
        dataset_version="v0.1",
        architecture_variant="deterministic_baseline",
    )

    assert config.model_provider is None
    assert config.prompt_version is None
    assert config.retrieval_configuration is None
    assert config.timestamp.tzinfo is not None


def test_timestamp_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        ExperimentConfig(
            experiment_id="exp-001",
            dataset_version="v0.1",
            architecture_variant="baseline",
            timestamp=datetime(2026, 8, 9, 18, 45),
        )


def test_blank_required_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_id="",
            dataset_version="v0.1",
            architecture_variant="baseline",
        )


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "experiment_id": "exp-001",
                "dataset_version": "v0.1",
                "architecture_variant": "baseline",
                "unknown": "value",
            }
        )


def test_experiment_results_embed_full_config_for_traceability() -> None:
    config = ExperimentConfig(
        experiment_id="exp-001",
        dataset_version="v0.1",
        architecture_variant="deterministic_baseline",
        random_seed=7,
    )
    metadata = ExperimentResultMetadata(config=config)

    assert metadata.config == config
    assert metadata.config.experiment_id == "exp-001"
