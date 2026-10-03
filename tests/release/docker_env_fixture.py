"""Read literal split-policy ENV defaults from the repository Dockerfiles.

This is a CPU fixture, not a Docker runtime emulator. Only policy variables
are retained; inherited CUDA image settings and runtime permissions are not
validated here. Docker -e settings must be overlaid by the calling test.
"""
import shlex

POLICY_KEYS = {"EVAL_SPLIT", "MMPTB_ROLE", "SWE_SPLIT", "MMSWE_SPLIT",
               "SWE_DATASET", "MMSWE_DATASET", "MMSWE_SPLIT_BY_OFFICIAL"}


def image_policy_env(*dockerfiles):
    result = {}
    for path in dockerfiles:
        for line in path.read_text().replace("\\\n", " ").splitlines():
            instruction, _, body = line.strip().partition(" ")
            if instruction.upper() != "ENV":
                continue
            words = shlex.split(body)
            if words and "=" not in words[0]:
                pairs = [(words[0], " ".join(words[1:]))]
            else:
                pairs = [word.split("=", 1) for word in words]
            for key, value in pairs:
                if key in POLICY_KEYS:
                    if "$" in value:
                        raise ValueError("split ENV expansion requires a Docker integration check")
                    result[key] = value
    return result
