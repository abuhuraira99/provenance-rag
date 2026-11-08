# Security

## Reporting

Please report vulnerabilities privately through GitHub security advisories on this repository rather
than in a public issue. Expect an acknowledgement within a week.

## Design notes

- The container makes **no network calls at run time**; weights ship in the image and
  `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` are set.
- Encrypted files are detected and skipped. Provenance RAG never attempts to decrypt or guess a
  password, and their contents are never indexed.
- No secrets live in the image or the repository. The `openai` backend reads `OPENAI_API_KEY` from
  the environment only (or from an env file passed to `bench --env-file`); the key is never written
  to reports or logs.
- The Studio binds to `127.0.0.1` by default and serves only its own page, its JSON API and report
  files from the configured reports directory (path traversal is rejected). It reads any folder you
  point it at, so do not expose it on a shared network.
- The resident server listens on a local unix socket (`/tmp/provenance_rag.sock`) and accepts only
  newline-delimited JSON requests.
