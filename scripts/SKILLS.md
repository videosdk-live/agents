---
name: scripts
description: Documentation generation and utility scripts for the agents repository
---

# scripts — Utility Scripts

## Purpose

Contains scripts for generating API reference documentation using pdoc3 and serving docs locally.

## Key Files

| File | Description |
|------|-------------|
| `build_docs.py` | Generates HTML API docs for the core agents package + all plugins |
| `serve_docs.py` | Local HTTP server for previewing generated docs |

## build_docs.py Architecture

The doc generation pipeline:

1. **Discovers packages** — scans `videosdk-agents/` and `videosdk-plugins/videosdk-plugins-*/`
2. **Runs pdoc3** — generates HTML docs with proper PYTHONPATH for namespace packages
3. **Post-processes** — removes `version.py` references, flattens nested directory structure
4. **Generates root index** — creates a landing page linking to all doc sections
5. **Special handling** — mocks rnnoise native module for doc generation

### Key Functions

- `build_docs_for_path()` — Core function that runs pdoc3 for a given package path
- `flatten_plugin_docs()` — Moves files from nested `videosdk/plugins/{name}/` to root
- `flatten_agents_docs()` — Flattens `agents/` nesting in output
- `remove_version_files()` — Strips version.py artifacts from generated docs
- `generate_root_index()` — Creates the styled HTML landing page

### Usage

```bash
python scripts/build_docs.py                    # Build all docs
python scripts/build_docs.py --base-url /docs   # Build with custom base URL
python scripts/serve_docs.py                     # Serve locally
```

### Output

Docs are generated to `agent-sdk-reference/` directory in the repo root.

## Gotchas

- Requires `pdoc3` (auto-installed if missing)
- Uses venv Python if available, falls back to system Python
- The rnnoise plugin requires special mocking due to native `.so` dependency
- Plugin docs need PYTHONPATH set to the plugin root for namespace resolution
