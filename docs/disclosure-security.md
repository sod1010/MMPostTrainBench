# Credentials and disclosure scope

The source distribution contains configuration examples. Supply service endpoints
and credentials through operator configuration; authentication and benchmark access
controls remain required. Keep real keys outside the checkout and outside agent
workspaces. The example uses `$XDG_CONFIG_HOME/mmposttrainbench` (or
`$HOME/.config/mmposttrainbench`); restrict the directory and key files to the
operator, for example with permissions `0700` and `0600` respectively.

Docker launchers pass credential environment-variable names instead of embedding
their values in command arguments. The HF downloader uses `HF_TOKEN` from its
environment. This reduces command-line exposure; it does not hide credentials from
the process that needs them, privileged host users, Docker administrators, or shell
tracing. Do not publish expanded runtime configuration, shell traces, or container
inspection dumps.

`src/eval/judge.py` excludes common credential files, agent configuration
directories, symlinks and special files. It redacts recognized credentials before
packing stopped-workspace files for the configured judge and before writing verdict
text. This is best-effort minimization, not a guarantee for arbitrary secret formats
or workspaces being modified concurrently. Redacted or omitted content is not proof
of a clean run. Missing credentials still yield an explicit skipped audit.

The legacy Harbor workspace Codex judge is disabled. Verifier output records
`unknown` and `certified=false`, never submitted or stale verdict files. The
verifier does not receive audit API keys. Numeric accuracy alone is not integrity
certification. The Docker loop invokes the operator-side `cheat_gate.sh` static
packer audit outside the workload; skipped or incomplete checks require independent
operator review before reporting. Keep its private inputs, credentials and output
separate from the agent. Research CLI prompts and logs still require disclosure
review.

MMSWE layer extraction uses contained directory descriptors and chroot-style
symlink resolution. Unsupported special members or extraction errors invalidate
the image; all official image variants have not been certified by CPU fixtures.

Public HTTPS uses normal certificate verification. Dataset warmup optionally accepts
an operator-approved CA for a managed proxy; it does not require a private CA and
does not disable TLS verification.

A source snapshot review does not cover Git history, other branches, issues or merge
request attachments, CI variables/artifacts, registry layers, local credentials,
datasets, model exports, or runtime logs. Review those separately if they are to be
disclosed. If a real credential is discovered, revoke or rotate it; deleting a source
line does not invalidate a leaked key.
