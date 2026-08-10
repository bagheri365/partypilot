from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "partypilot"

FORBIDDEN_DOMAIN_PREFIXES = (
    "fastapi",
    "langgraph",
    "qdrant_client",
    "ollama",
    "openai",
    "partypilot.adapters",
    "partypilot.composition",
)

FORBIDDEN_APPLICATION_PREFIXES = (
    "partypilot.adapters",
    "partypilot.composition",
)


@dataclass(frozen=True, slots=True)
class ImportViolation:
    path: Path
    line: int
    imported_module: str
    reason: str


def _imported_modules(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append((node.lineno, node.module))
    return imports


def _matches_prefix(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def find_forbidden_imports(
    path: Path,
    source: str,
    *,
    prefixes: tuple[str, ...],
    reason: str,
) -> list[ImportViolation]:
    return [
        ImportViolation(path=path, line=line, imported_module=module, reason=reason)
        for line, module in _imported_modules(source)
        if _matches_prefix(module, prefixes)
    ]


def _scan_package_area(
    area: str,
    *,
    prefixes: tuple[str, ...],
    reason: str,
) -> list[ImportViolation]:
    violations: list[ImportViolation] = []
    for path in sorted((PACKAGE_ROOT / area).rglob("*.py")):
        violations.extend(
            find_forbidden_imports(
                path,
                path.read_text(encoding="utf-8"),
                prefixes=prefixes,
                reason=reason,
            )
        )
    return violations


def _format_violations(violations: list[ImportViolation]) -> str:
    return "\n".join(
        f"{violation.path}:{violation.line}: {violation.imported_module} ({violation.reason})"
        for violation in violations
    )


def test_domain_has_no_outward_or_provider_imports() -> None:
    violations = _scan_package_area(
        "domain",
        prefixes=FORBIDDEN_DOMAIN_PREFIXES,
        reason="domain must remain independent of infrastructure and provider SDKs",
    )
    assert not violations, _format_violations(violations)


def test_application_does_not_depend_on_adapters_or_composition() -> None:
    violations = _scan_package_area(
        "application",
        prefixes=FORBIDDEN_APPLICATION_PREFIXES,
        reason="application must depend on ports/contracts rather than concrete infrastructure",
    )
    assert not violations, _format_violations(violations)


def test_detector_finds_forbidden_domain_provider_import() -> None:
    violations = find_forbidden_imports(
        Path("example.py"),
        "from openai import OpenAI\n",
        prefixes=FORBIDDEN_DOMAIN_PREFIXES,
        reason="test",
    )
    assert [violation.imported_module for violation in violations] == ["openai"]


def test_detector_finds_forbidden_domain_adapter_import() -> None:
    violations = find_forbidden_imports(
        Path("example.py"),
        "from partypilot.adapters.database import Repository\n",
        prefixes=FORBIDDEN_DOMAIN_PREFIXES,
        reason="test",
    )
    assert [violation.imported_module for violation in violations] == [
        "partypilot.adapters.database"
    ]


def test_detector_finds_forbidden_application_adapter_import() -> None:
    violations = find_forbidden_imports(
        Path("example.py"),
        "import partypilot.adapters.resources\n",
        prefixes=FORBIDDEN_APPLICATION_PREFIXES,
        reason="test",
    )
    assert [violation.imported_module for violation in violations] == [
        "partypilot.adapters.resources"
    ]


def test_detector_allows_inward_infrastructure_dependency() -> None:
    violations = find_forbidden_imports(
        Path("example.py"),
        "from partypilot.domain import models\nfrom partypilot.ports import resources\n",
        prefixes=FORBIDDEN_APPLICATION_PREFIXES,
        reason="test",
    )
    assert violations == []
