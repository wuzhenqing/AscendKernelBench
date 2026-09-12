# Maintain and publish the documentation

The site uses VitePress with Markdown pages in `docs/`. Documentation builds are
independent of the benchmark's Python dependencies and need no Ascend software.
VitePress is pinned in `package.json`; commit `package-lock.json` when updating it.
The dependency override selects Vite 6.4.3 to avoid the advisories in VitePress
1.6.4's default Vite 5 dependency. Check the production build, development server,
and dependency audit when updating or removing this override.

## Local development

Use Node.js 24, the version recorded in `.nvmrc` and used by GitHub Actions. From
the repository root:

```bash
npm ci
npm run docs:dev
```

Open the URL printed by the server, including `/AscendKernelBench/`. Edit Markdown
files to preview changes. Before committing, build and check the generated site:

```bash
npm run docs:check
npm run docs:preview
```

`docs:check` runs VitePress's production build, which fails on unresolved internal
page links, and a separate check of local HTML links, fragment identifiers, and
referenced assets. It does not contact external websites. Preview the result at
`http://localhost:4173/AscendKernelBench/` unless the server reports another port.

Check the home page, one long reference page, local search, the narrow-screen
navigation, and light/dark appearance when changing the theme. Generated output
in `docs/.vitepress/dist/` and the VitePress cache are ignored by Git.

Python style and unit tests are a separate GitHub Actions workflow
(`.github/workflows/quality.yml`). That job installs `.[dev]`, runs
`pre-commit run --all-files`, and runs `pytest`. It does not compile
Ascend C or require an NPU. See [CONTRIBUTING.md](https://github.com/wuzhenqing/AscendKernelBench/blob/main/CONTRIBUTING.md).

## Add or update a page

1. Write an English Markdown page under `docs/guide/` or `docs/reference/`.
2. Use a single first-level title and descriptive second-level headings.
3. Add the page to `themeConfig.sidebar` in `docs/.vitepress/config.mts`.
4. Link to it from the relevant workflow or guide using a site-relative Markdown
   link, such as `[Configuration](/reference/configuration)`.
5. Run `npm run docs:check` and preview the page.

Keep CLI examples aligned with `scripts/*.py`, defaults with
`src/ascend_kernel_bench/config.py` and `configs/eval_default.yaml`, and scoring
explanations with `src/ascend_kernel_bench/score.py`. Distinguish implemented
behavior from proposals and hardware validation. The existing routes
`/task_authoring` and `/deploy_llm_service` are retained for the original guides.

Do not include credentials, private endpoint addresses, or local run artifacts in
published examples. The site has no external search service; its local search
index is generated from the published pages.

## GitHub Pages deployment

The site is published at
[wuzhenqing.github.io/AscendKernelBench](https://wuzhenqing.github.io/AscendKernelBench/).
The repository Pages source must be **GitHub Actions** in **Settings → Pages →
Build and deployment**. The workflow is
[`.github/workflows/docs.yml`](https://github.com/wuzhenqing/AscendKernelBench/blob/main/.github/workflows/docs.yml).

| Trigger | Behavior |
| --- | --- |
| Pull request targeting `main` | Install dependencies, build, and check internal links; no deployment |
| Push to `main` | Run the same checks, upload the static artifact, and deploy to Pages |
| Manual run on `main` | Rebuild and deploy using the workflow dispatch control |

The build job has read access to repository contents. Only the deployment job
receives `pages: write` and `id-token: write`, and it uses the `github-pages`
environment. Deployment does not require a personal access token in repository
secrets. Main-branch deployments finish before a later deployment starts.

The site uses `base: '/AscendKernelBench/'` because it is hosted below a repository
path. If you fork the repository, rename it, or add a custom domain, update `base`,
the favicon path, sitemap hostname, GitHub edit/social links, the `base` constant
in `scripts/check_docs.mjs`, and the README's documentation URLs. Configure Pages
in the destination repository as well.

The setup follows the
[VitePress deployment guide](https://vitepress.dev/guide/deploy) and
[GitHub's custom workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Diagnose a failed publication

Open the repository's
[Documentation workflow](https://github.com/wuzhenqing/AscendKernelBench/actions/workflows/docs.yml).
If the build fails, fix the reported Markdown or broken link and reproduce it
with `npm run docs:check`. If deployment fails, confirm that Pages uses GitHub
Actions, the `github-pages` environment allows `main`, and repository Actions
policies permit the official actions used in the workflow.

If the home page loads without styles or nested pages return 404, check the
configured base path and preview the production build. Keep the generated `.html`
URLs; the site does not depend on custom server rewrite rules.
