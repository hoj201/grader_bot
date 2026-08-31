#!/bin/bash
# SessionStart hook for Claude Code on the web.
#
# Installs everything the test suite needs beyond `poetry install` so
# `poetry run pytest` is fully green in a fresh remote session:
#   - the LaTeX toolchain (latexmk + the texlive packages
#     tex/gbworksheet.sty needs) that tests/test_worksheet_synth.py compiles
#     tex/demo.tex with, and tesseract-ocr (the native binary pytesseract
#     shells out to) -- mirrors the apt-get list in Dockerfile.
#   - the ArUco marker images the worksheet .tex template references. They're
#     generated locally rather than committed (see tex/aruco_images/ in
#     .gitignore), so a fresh checkout is missing them until this runs.
#
# Only meant for Claude Code on the web -- no-op elsewhere.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

apt-get update
apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-pictures \
    texlive-fonts-recommended \
    latexmk \
    tesseract-ocr

poetry install

poetry run python scripts/generate_aruco.py
