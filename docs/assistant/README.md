# RLinf Docs Assistant Runtime

The assistant runtime is plain browser JavaScript because the docs are built by
Sphinx. The canonical editable copies live under:

```text
docs/source-en/_static/
```

The zh docs tree carries the same runtime files so each Sphinx build can package
its own static assets. After changing assistant JavaScript or CSS, run:

```bash
node docs/assistant/sync-static.mjs
node --test docs/tests/assistant-*.test.mjs
```

Keep locale-specific text in Sphinx templates or runtime config. Keep shared
request handling, source rendering, markdown sanitization, query planning, and
Typesense request construction in the synced static files.
