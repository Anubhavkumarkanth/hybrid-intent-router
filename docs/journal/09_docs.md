# Phase 9: README, interview guide, final report

## What we did
- `README.md`: a one-line hook, the problem, how it works (a code-drawn diagram `results/charts/architecture.png`
  plus a Mermaid version), "what the numbers say", result tables and charts, how to run, design decisions,
  limitations, project layout, dataset credit and references.
- `docs/INTERVIEW_GUIDE.md`: a 30-second pitch, a 2-minute explanation, 20 questions with short answers that use
  this run's numbers, the errors we hit, and two resume bullets (software/ML and data analyst).
- `FINAL_REPORT.md`: what was built, final results, time per phase, every decision and its reason, problems and
  fixes, and what is left for the owner (GitHub push commands).
- `scripts/render_tables.py` now also renders the pitch and the resume bullets from `results/`, so the numbers a
  recruiter reads can never drift from the real results: `scripts/check_numbers.py` fails if they do.

## Why
- Interviewers ask "how do you know?". Every table in the docs is generated from `results/` and checked by
  `make verify`, so the answer is "the test suite checks it".
- Simple English, short sentences, each technical word explained once.

## Result
- `make tables` then `make verify`: ruff clean, all tests pass, `check_numbers: OK` for 15 result sections in
  README (7), FINAL_REPORT (5) and INTERVIEW_GUIDE (3) (`logs/08_verify.log` has the final run).

## Errors and fixes
- The prose said "1,345 ms" while the rendered tables said "1344 ms" for the same value (1,344.5 ms; Python
  rounds halves to the even number). Fixed the prose to match the tables.
- The data-analyst resume bullet repeated "88.9%" twice (calls avoided and cost saved are the same number,
  because cost is proportional to calls). Reworded to show the cost in dollars instead.

## How to explain this in an interview
1. Every number in my README is generated from the results files, and a test fails if the docs and the results disagree.
2. I wrote the limitations down myself: translated data, not real code-mixed speech; a 500-sentence test subset; one laptop.
3. The report lists every decision with its reason, so I can defend each one.
