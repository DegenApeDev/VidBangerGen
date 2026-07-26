# Contributing

Contributions are welcome through issues and pull requests.

## Development setup

```bash
python3 scripts/setup.py setup
npm test
npm run build
```

The regular test suite uses temporary local state and should not submit GPU
work. Live ComfyUI checks are explicit and require an operator-configured
worker.

## Pull requests

- Keep runtime data, `.env`, logs, media, model inventories, and machine-specific
  deployment details out of commits.
- Add or update tests for behavior changes.
- Run the backend tests and frontend production build.
- Describe user-visible changes and any new models, nodes, or capabilities.
- Do not silently enable capability-gated workflows.

Unless stated otherwise, contributions intentionally submitted to this project
are accepted under the Apache License, Version 2.0.
