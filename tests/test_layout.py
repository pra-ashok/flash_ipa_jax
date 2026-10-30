"""Keep the requested module-size limit as the package grows."""

from pathlib import Path


def test_python_modules_stay_within_300_lines():
    root = Path(__file__).resolve().parents[1]
    oversized = []
    for directory in ("src", "examples", "benchmarks", "tests", "docs"):
        for path in (root / directory).rglob("*.py"):
            line_count = len(path.read_text().splitlines())
            if line_count > 300:
                oversized.append(f"{path.relative_to(root)}: {line_count} lines")
    assert not oversized, "Split these modules into subpackages:\n" + "\n".join(
        oversized
    )
