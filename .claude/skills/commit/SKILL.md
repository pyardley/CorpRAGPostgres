---
name: commit
description: Summarise staged/unstaged changes, propose a conventional commit message, confirm with the user, then commit and push.
---

1. Run `git status` and `git diff HEAD` (to see both staged and unstaged changes).
2. Summarise what changed in plain English — file names, what was added/removed/fixed, and why (inferred from context).
3. Choose the correct conventional commit type:
   - **feat** — new feature or capability
   - **fix** — bug fix
   - **docs** — documentation only (README, CLAUDE.md, comments)
   - **chore** — tooling, config, dependencies, CI, scripts
   - **test** — adding or updating tests
   - **refactor** — code restructure with no behaviour change
4. Draft a commit message in this format:
   ```
   <type>(<optional-scope>): <short imperative summary under 72 chars>

   <optional body: one or two sentences on the why, not the what>

   Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
   ```
5. Show the proposed message to the user, then use the AskUserQuestion tool with a single Yes/No-style question (e.g. header "Commit msg", options "Yes, commit" / "No, edit") to get confirmation — do not ask in plain text and wait for a free-form reply.
6. If the user picks "Yes, commit", proceed immediately through staging, committing, and pushing without any further confirmation prompts. If they pick "No, edit" (or use the free-text option to request changes), revise the message and repeat step 5.
7. Stage any unstaged changes (`git add` the relevant files — do not use `git add .` unless the user approves), commit with the confirmed message, and push to the current branch.
8. Report the resulting commit hash and the push status.
