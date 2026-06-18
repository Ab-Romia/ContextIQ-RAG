#!/usr/bin/env bash
# Deploy the current commit to the Hugging Face Space.
#
# The Space reads its configuration from the YAML front matter at the top of README.md,
# but in this repository README.md is the guide and the Space card lives in
# README_HUGGINGFACE.md. This script builds a clean tree from the current commit, swaps
# the Space card into place as README.md, and force-pushes it to the Space, leaving this
# repository untouched.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/deploy_space.sh [owner/space]
#
# Requires a Hugging Face write token. The default Space is Ab-Romia/Context-Aware-AI.

set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face write token}"
SPACE="${1:-Ab-Romia/Context-Aware-AI}"
REMOTE="https://Ab-Romia:${HF_TOKEN}@huggingface.co/spaces/${SPACE}.git"

ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$TMP"
cp "$TMP/README_HUGGINGFACE.md" "$TMP/README.md"

cd "$TMP"
git init -q
git add -A
git -c user.name="Ab-Romia" -c user.email="aabouroumia@gmail.com" commit -qm "Deploy ContextIQ"
git push --force "$REMOTE" HEAD:main

echo "Deployed ${SPACE}. The Space will rebuild; first build downloads nothing extra because models are baked into the image."
