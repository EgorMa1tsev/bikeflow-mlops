# Contributing

1. Create a focused branch from current `main` (`feat/*`, `fix/*`, `docs/*`, or `chore/*`).
2. Keep commits in Conventional Commits format, for example `feat(api): validate prediction input`.
3. Run `ruff check .`, `ruff format --check .`, `pytest`, and a Docker build.
4. Open a pull request, complete the checklist, and request review from the other participant.
5. Merge only after CI passes and one approval is recorded; use squash merge.

Never commit secrets, datasets, trained model binaries, or generated caches. Changes to the
prediction schema require both participants to approve the model/API contract.
