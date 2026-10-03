# Research use and compliance

This benchmark is intended for evaluation and research. The benchmark protocol does not add restrictions to the MIT license. Users are responsible for ensuring compliance with the terms of service of any third-party services they employ. Agents autonomously execute code and training workflows; running in an isolated environment is recommended.

## Agent rule: no external-model distillation

External-model distillation is disallowed. Agents may not call external model APIs to generate or synthesize training data.

This prohibition covers API-generated questions, answers, labels, rationales, and preference data used to train the submitted model, including calls made through scripts, tools, proxies, or research-agent credentials. Using a third-party model to operate the research agent for planning and coding is not by itself distillation; repurposing its generated outputs as training examples or labels is prohibited. Independent evaluation and integrity judging remain separate from training-data production.

Public availability alone does not establish permission to use a dataset. Operators and agents must respect task-specific data permissions, dataset/model licenses, evaluation-split restrictions, and the applicable third-party service terms.

## Audit treatment

The workspace judge includes `external_model_distillation` as an explicit violation category. The Harbor verifier reports evidence of this violation through its existing `disallowed_model_judgement.txt` verdict. Reviews should trace the external-model call or generated content to training-data construction or training use and retain the supporting code, provenance, and logs. A client-library import or a credential name alone is insufficient evidence.

These rules and post-hoc checks do not provide a complete network-level block. Operators remain responsible for access controls and review. Record which checks were actually completed; missing evidence or a skipped check does not demonstrate compliance. Adding this policy does not retroactively certify historical runs or change historical scores without a new evidence-based audit.

## Autonomous execution

- Run research agents in a dedicated container or virtual machine with only the required workspace, compute, and approved data mounted.
- Keep sealed evaluation assets and audit credentials separate from the research environment. Scope research-agent credentials to the authorized service and use.
- Avoid exposing unrelated host directories, host Docker sockets, or unnecessary privileged capabilities. Grant any task-specific execution privileges only inside a dedicated isolation boundary.
- Apply suitable network access, resource, and time limits; retain execution logs and review outputs before reuse or release.

This notice describes benchmark use and operator responsibilities; it does not replace third-party terms or the repository license.
