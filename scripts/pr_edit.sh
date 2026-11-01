#!/usr/bin/env bash
# Edit an open PR's title or body. Refuses a merged or closed PR: on 2026-11-01 a merged PR was retitled with a raw
# `gh api PATCH`, which checks nothing. Uses the REST API (gh pr edit failed on a GraphQL deprecation, 2026-10-31).
# Usage: scripts/pr_edit.sh N [--title TEXT] [--body-file FILE]
set -euo pipefail
[ "$#" -ge 3 ] || { echo "usage: pr_edit.sh N [--title TEXT] [--body-file FILE]" >&2; exit 2; }
n=$1; shift
repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner)
state=$(gh api "repos/$repo/pulls/$n" --jq 'if .merged then "MERGED" else (.state | ascii_upcase) end')
if [ "$state" != "OPEN" ]; then
  echo "REFUSED: PR $n is $state; edit only open PRs (comment on a merged one with gh pr comment)." >&2
  exit 1
fi
args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --title) args+=(-f "title=$2"); shift 2 ;;
    --body-file) args+=(-F "body=@$2"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
gh api -X PATCH "repos/$repo/pulls/$n" "${args[@]}" --jq '.html_url + "  (" + .title + ")"'
