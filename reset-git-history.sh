#!/bin/bash
# reset-git-history.sh
#
# Wipes the entire git history and replaces it with a single initial commit.
# Use this to ensure an autonomous agent cannot retrieve or build upon previous
# experiments. After running, the repo will have one commit with no prior
# experiment history—useful when starting a fresh research run.
#
# Usage:
#   ./reset-git-history.sh
#   # or
#   bash reset-git-history.sh

set -e

echo "Creating orphan branch (no history)..."
git checkout --orphan fresh_main

echo "Staging all files..."
git add -A

echo "Creating fresh initial commit..."
git commit -m "Initial commit"

echo "Replacing main branch..."
git branch -D main 2>/dev/null || true
git branch -m main

echo "Force pushing to origin (overwrites remote history)..."
git push -f origin main

echo "Done. Git history has been reset; remote has a single commit."
