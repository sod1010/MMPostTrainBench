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

The separate Codex CLI judge in `src/harbor_adapter/template/tests/test.sh` uses
tool access to the workspace and does not use that packer. The research CLIs also
operate with runtime credentials. Their prompts, tool output and logs require a
separate disclosure review; do not assume that the packer's filtering covers those
paths. A stricter deployment needs a separate judge execution boundary and reviewed
inputs, in addition to source cleanup.

Public HTTPS uses normal certificate verification. Dataset warmup optionally accepts
an operator-approved CA for a managed proxy; it does not require a private CA and
does not disable TLS verification.

A source snapshot review does not cover Git history, other branches, issues or merge
request attachments, CI variables/artifacts, registry layers, local credentials,
datasets, model exports, or runtime logs. Review those separately if they are to be
disclosed. If a real credential is discovered, revoke or rotate it; deleting a source
line does not invalidate a leaked key.
