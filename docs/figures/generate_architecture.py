"""Render the editable architecture schematics as matching SVG and PNG files.

Run from the repository root::

    python docs/figures/generate_architecture.py

Install the system ``graphviz`` package to provide ``neato``. No Python plotting
package or project installation is needed. Edit ``architecture.dot`` or
``ipa_attention.dot`` to change labels, connections, and positions. Coordinates
in those files are points; ``neato -n2`` preserves their carefully placed layout.
SVG is used in the documentation; PNG copies are provided for presentations.
"""

from pathlib import Path
import shutil
import subprocess

OUTPUT_DIR = Path(__file__).resolve().parent


def render(name):
    """Render the authoritative DOT source without rewriting the source file."""
    source = OUTPUT_DIR / f"{name}.dot"
    for extension in ("svg", "png"):
        subprocess.run(
            [
                "neato",
                "-n2",
                f"-T{extension}",
                str(source),
                "-o",
                str(source.with_suffix(f".{extension}")),
            ],
            check=True,
        )


def main():
    """Regenerate both figures; fail clearly if Graphviz is not installed."""
    if shutil.which("neato") is None:
        raise SystemExit(
            "Install Graphviz (the 'neato' command), then rerun this script."
        )
    for name in ("architecture", "ipa_attention"):
        render(name)
        print(f"Rendered {name}.svg and {name}.png from {name}.dot in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
