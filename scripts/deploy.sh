#!/usr/bin/env bash
# Deploy rangebench to a remote host via SSH.
# Usage: REMOTE=user@host REMOTE_DIR=/path/to/rangebench ./scripts/deploy.sh
# Defaults to localhost for local testing.
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="${REMOTE:-}"
REMOTE_DIR="${REMOTE_DIR:-${1:-~/rangebench}}"
EXCLUDE=(--exclude=__pycache__ --exclude=results --exclude=.git --exclude=rangebench/__pycache__)
if [[ -z "$REMOTE" ]]; then
  echo "No REMOTE set, assuming local deploy to $REMOTE_DIR"
  mkdir -p "$REMOTE_DIR"
  tar czf - -C "$SRC" "${EXCLUDE[@]}" . | tar xzf - -C "$REMOTE_DIR"
else
  echo "Deploying to $REMOTE:$REMOTE_DIR"
  tar czf - -C "$SRC" "${EXCLUDE[@]}" . | ssh "$REMOTE" "mkdir -p $REMOTE_DIR && tar xzf - -C $REMOTE_DIR"
  ssh "$REMOTE" "cd $REMOTE_DIR && python3 -m compileall -q rangebench && echo compile-ok && ls tasks | head"
fi
echo "done"
