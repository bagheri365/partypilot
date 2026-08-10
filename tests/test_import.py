from __future__ import annotations


def test_package_imports() -> None:
    import partypilot

    assert partypilot.__name__ == "partypilot"
