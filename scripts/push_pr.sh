#!/usr/bin/env bash
# Push the current branch only if it has no PR yet or its PR is still open.
# Two commits were lost on 2026-10-27 by pushing to a branch whose PR had just been merged (#14, #28).
# Usage: scripts/push_pr.sh
set -euo pipefail
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" = "main" ]; then
  echo "REFUSED: on main. Create a branch: git checkout -b NAME origin/main" >&2
  exit 1
fi
# Only "no pull requests found" means no PR; any other gh failure (auth, network) refuses rather than skip the check
if ! state=$(gh pr view "$branch" --json state -q .state 2>&1); then
  if [[ "$state" == *"no pull requests found"* ]]; then
    state="NONE"
  else
    echo "REFUSED: could not check the PR for '$branch': $state" >&2
    exit 1
  fi
fi
if [ "$state" = "MERGED" ] || [ "$state" = "CLOSED" ]; then
  echo "REFUSED: the PR for '$branch' is $state; a push here never reaches main." >&2
  echo "Move the new commits: git checkout -b NEW origin/main && git cherry-pick <commits>" >&2
  exit 1
fi
git push -u origin "$branch"
# The PR can be merged while this push is in flight: confirm the head commit is in the PR
if [ "$state" = "OPEN" ]; then
  head=$(git rev-parse HEAD)
  # GitHub's API can lag the push by a few seconds: retry before calling it a mismatch
  for _ in 1 2 3 4 5; do
    pr_head=$(gh pr view "$branch" --json headRefOid -q .headRefOid)
    [ "$head" = "$pr_head" ] && break
    sleep 2
  done
  if [ "$head" != "$pr_head" ]; then
    state_now=$(gh pr view "$branch" --json state -q .state)
    echo "WARNING: PR is $state_now and its head $pr_head != pushed $head; if it merged during the push, the" \
         "commit is not on main" >&2
    exit 1
  fi
fi
echo "pushed $branch (PR: $state)"
