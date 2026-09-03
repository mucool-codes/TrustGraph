# CLAUDE.md

Trust-aware, explainable task offloading for vehicular fog networks: a GNN scores the
trustworthiness of each roadside fog node from observed task outcomes, and a fixed
analytic rule uses that score to pick an offload target and explain the choice. This is
a 40-day solo research project run as a series of one-task sessions, targeting a
conference submission.

## Read first, every session

Before doing anything else, read:

- `PROJECT_SPEC.md` — authoritative. Locked decisions, ablation variants, evaluation
  plan, feature glossary. Supersedes `DEVELOPMENT_GUIDE.md` and
  `TECHNICAL_AND_NOVELTY.md` wherever they conflict.
- `DECISIONS.md` — what was chosen and why.
- `FINDINGS.md` — what was measured and what it means.

Session plan for all 12 sessions: `CLAUDE_CODE_SESSIONS.md`.

---

## STANDING RULES

1. Read PROJECT_SPEC.md and DECISIONS.md before doing anything else. They
   contain locked decisions. Do not silently override them. If a locked
   decision appears wrong, raise a DECISION REQUEST rather than changing it.

2. Scope discipline. Do only this session's task. Do not start the next
   session's work even if it seems trivial. If you finish early, improve
   tests and documentation for work already done.

3. Simplicity rule. Before adding complexity (a fancier layer, an extra
   feature, an abstraction), first confirm the simple version is actually
   failing. Do not add complexity pre-emptively.

4. Stop at the exit condition. Show me a runnable state. Do not continue
   past it.

5. You handle ALL git operations yourself — see GIT WORKFLOW below. I make
   no commits manually and will not run any git commands.

6. Maintain two living documents, and keep them distinct:

   DECISIONS.md — what was CHOSEN and WHY. Every non-trivial design choice:
   the choice, the alternatives considered, the rationale, the date. This is
   the review-defense artifact. If a later session reverses an earlier
   decision, add a new entry that supersedes the old one — never edit or
   delete the original.

   FINDINGS.md — what was MEASURED and what it means. Every empirical result:
   calibration numbers, training diagnostics, sweep outcomes, timings,
   failures. Each entry records the date, commit hash, config/seed used, the
   raw numbers, and a one-line interpretation.

   CRITICAL: write the FINDINGS.md entry BEFORE analysing, tuning, or trying
   to improve a result. A number recorded after three rounds of tuning is not
   the same number. Negative results are recorded with the same weight as
   positive ones and are never overwritten.

7. Determinism. Every run must be reproducible from (config file, seed).
   No unseeded randomness anywhere.

8. When you hit a design question you cannot resolve from PROJECT_SPEC.md,
   emit a DECISION REQUEST in the exact format below and STOP. Do not
   guess and proceed. Do not pick "the reasonable default" silently.

## GIT WORKFLOW

Repo: https://github.com/mucool-codes/TrustGraph — already cloned locally.
The GitHub CLI (gh) is installed and authenticated. You manage branching,
committing, merging, tagging, and pushing entirely yourself.

At session start:
  git checkout main && git pull origin main
  git checkout -b session/s<n>-<short-name>

During the session:
  Commit in small, logical increments with clear messages — several per
  session, not one at the end. Never commit generated artifacts: mobility
  traces, model checkpoints, results tables, figures, __pycache__, venv.
  Add them to .gitignore. Code, configs, tests, and docs are committed.

At the exit condition, ONLY once the work actually runs and tests pass:
  git push -u origin session/s<n>-<short-name>
  git checkout main
  git merge --no-ff session/s<n>-<short-name> -m "S<n>: <one-line summary>"
  git tag s<n>-complete
  git push origin main --tags
  git branch -d session/s<n>-<short-name>
  git push origin --delete session/s<n>-<short-name>

Rules:
  - Merge only after the exit condition is met. A failing session stays on
    its branch, unmerged, and you tell me why.
  - If a merge conflicts, STOP and raise a DECISION REQUEST. Do not resolve
    conflicts by discarding work, and do not rebase main.
  - Never force-push. Never rewrite published history.
  - The s<n>-complete tags are rollback points. Do not delete or move them.
  - Report the merge commit hash and tag at the end of the session.

## DECISION REQUEST FORMAT

===== DECISION REQUEST: S<n> =====
CONTEXT: <3-6 sentences. What you built, where you are, what forced this
         question. Enough that someone who has not seen the code can follow.>
QUESTION 1: <the question>
  OPTIONS: <the options you see, with the trade-off of each in one line>
  YOUR LEAN: <what you would pick and why, in one line>
QUESTION 2: ...
BLOCKING: <yes/no — can you continue on other work while waiting?>
===== END =====

Emit this as plain text I can copy. Keep the whole block under 400 words.
Batch questions — do not emit one request per question if several can wait.
