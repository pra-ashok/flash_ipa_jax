# Building and publishing the documentation

The documentation uses MkDocs with its built-in Read the Docs theme. A grouped left sidebar contains every guide; the current page expands to show its table of contents. Search works across the guides. The sidebar becomes a menu on small screens, and the appearance control switches between light and dark.

## Preview locally

Run these commands from the repository root. Documentation dependencies are separate from JAX, CUDA, ROCm and TPU packages:

```bash
python3 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

Open the local address printed by MkDocs. It reloads when a guide changes. For a complete build that rejects broken guide and image links, run:

```bash
python -m mkdocs build --strict
```

The generated website goes into `site/`, which is ignored by Git. Checked-in architecture diagrams are included as they are; building the site does not require Graphviz or a running accelerator.

## Publish on GitHub Pages

In the repository's **Settings → Pages → Build and deployment**, select **GitHub Actions** as the source. The Documentation workflow then checks pull requests and publishes changes pushed to `main`. It can also be run from the Actions tab with **Run workflow**.

The workflow reads the site's address from GitHub Pages, so links also work when the site lives under a repository path or uses a custom domain. The local default address in `mkdocs.yml` can be overridden with `SITE_URL`. Read [GitHub's custom-workflow guide](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) for the hosting setup and required repository permissions.

## Edit the content and navigation

Edit guides under `docs/`, and update `nav` in `mkdocs.yml` when adding a page. The root README becomes the homepage; the existing validation record and examples README are also included directly. There are no duplicate guide copies to keep in sync.

Use normal relative Markdown links when writing. During the build, the small `docs/tools/site_links.py` hook keeps guide and image links inside the site and points source-code links at GitHub. On deployed builds, these source links use the same commit as the documentation.

The sidebar spacing, colors and code presentation live in
`docs/stylesheets/extra.css`. The theme override and a small script add the appearance control and keyboard-accessible navigation buttons. The layout, search and mobile drawer come from MkDocs.

References: [MkDocs themes](https://www.mkdocs.org/user-guide/choosing-your-theme/#readthedocs) and [MkDocs configuration](https://www.mkdocs.org/user-guide/configuration/).
