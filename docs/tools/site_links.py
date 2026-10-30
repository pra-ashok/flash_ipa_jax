"""Build one linked documentation site from Markdown kept in the repository.

The README, validation record and examples guide remain authoritative in their
original locations. MkDocs receives in-memory copies, so builds do not create
duplicate tracked Markdown. Guide/image links stay inside the documentation;
links to Python, workflows and directories open the matching GitHub source.
"""

import os
import posixpath
import re
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[2]
VIRTUAL_PAGES = {
    "index.md": "README.md",
    "validation.md": "VALIDATION.md",
    "examples.md": "examples/README.md",
}
MARKDOWN_LINK = re.compile(r"(?P<prefix>!?\[[^\]\n]*\]\()(?P<url>[^\s)]+)")


def on_files(files, config, **kwargs):
    """Include shared repository guides without copying them into ``docs/``."""
    for destination, source in VIRTUAL_PAGES.items():
        files.append(
            File.generated(
                config,
                destination,
                content=(ROOT / source).read_text(encoding="utf-8"),
            )
        )
    return files


def _site_destination(repository_path):
    """Map a source path to its guide or asset location in the built site."""
    for destination, source in VIRTUAL_PAGES.items():
        if repository_path == source:
            return destination
    if repository_path.startswith("docs/"):
        return repository_path.removeprefix("docs/")
    return None


def _rewrite_url(url, source, page_path, config):
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path or url.startswith("/"):
        return url

    # Resolve against the original Markdown, including generated shared guides.
    repository_path = posixpath.normpath(
        posixpath.join(posixpath.dirname(source), unquote(parsed.path))
    )
    disk_path = ROOT / repository_path
    if repository_path.startswith("../") or not disk_path.exists():
        raise PluginError(f"{source}: link target does not exist: {url}")

    destination = _site_destination(repository_path)
    if destination is not None:
        relative = posixpath.relpath(destination, posixpath.dirname(page_path))
        return urlunsplit(("", "", relative, parsed.query, parsed.fragment))

    # Pin deployed source links to the version the reader is looking at.
    repository_url = config["repo_url"].rstrip("/")
    revision = quote(os.environ.get("GITHUB_SHA", "main"), safe="")
    kind = "tree" if disk_path.is_dir() else "blob"
    path = quote(repository_path, safe="/")
    return urlunsplit(
        (
            "https",
            urlsplit(repository_url).netloc,
            f"{urlsplit(repository_url).path}/{kind}/{revision}/{path}",
            parsed.query,
            parsed.fragment,
        )
    )


def on_page_markdown(markdown, page, config, **kwargs):
    """Keep source links useful both on GitHub and on a project Pages site."""
    page_path = page.file.src_uri
    source = VIRTUAL_PAGES.get(page_path, f"docs/{page_path}")

    def rewrite(match):
        url = _rewrite_url(match["url"], source, page_path, config)
        return f"{match['prefix']}{url}"

    # Examples often contain Markdown syntax; leave fenced snippets untouched.
    sections = re.split(
        r"(^```[^\n]*\n.*?^```[^\n]*$)", markdown, flags=re.MULTILINE | re.DOTALL
    )
    for index in range(0, len(sections), 2):
        sections[index] = MARKDOWN_LINK.sub(rewrite, sections[index])
    return "".join(sections)
