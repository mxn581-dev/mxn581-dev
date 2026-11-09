# Rebuilds the README hero SVGs from real PBMC 3k data.
# Runs when the generator changes, not on a cron: the input data is static,
# so a nightly rebuild would commit byte-identical files and prove nothing.
name: hero

on:
  push:
    branches: [main]
    paths:
      - "scripts/**"
      - "tests/**"
      - ".github/workflows/hero.yml"
  workflow_dispatch:

permissions:
  contents: write

concurrency:
  group: hero
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install
        run: pip install numpy pytest scanpy

      - name: Test generator
        run: pytest -q

      - name: Export PBMC 3k embedding
        run: python scripts/export_pbmc_embedding.py --out data/pbmc.csv --n-cells 800

      - name: Render light and dark SVGs
        run: |
          for theme in light dark; do
            python scripts/make_umap_svg.py \
              --embedding data/pbmc.csv \
              --theme "$theme" \
              --caption "10x PBMC 3k · 800 cells · UMAP + Louvain labels" \
              --out "assets/umap-$theme.svg"
          done

      - name: Commit if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898283+github-actions[bot]@users.noreply.github.com"
          git add assets/umap-light.svg assets/umap-dark.svg
          # The [skip ci] guard stops this commit from retriggering the workflow.
          git diff --cached --quiet || git commit -m "Regenerate hero SVGs [skip ci]"
          git push
