#!/bin/bash
set -e

REPO="$HOME/gpt-notes"
LOGFILE="$REPO/auto_commit.log"

cd "$REPO"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] start auto commit" >> "$LOGFILE"

git add .

if ! git diff --cached --quiet; then
    git commit -m "auto backup: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOGFILE" 2>&1

    if git remote get-url origin >/dev/null 2>&1; then
        git push origin HEAD >> "$LOGFILE" 2>&1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] no remote origin, skip push" >> "$LOGFILE"
    fi
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] no changes" >> "$LOGFILE"
fi
