import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ADAPTER_NAME = "POSTTRAINBENCH"
TEMPLATE_DIR = Path(__file__).parent / "template"

# PostTrainBench source directory (relative to repo root)
POSTTRAINBENCH_ROOT = Path(__file__).parent.parent.parent

# Benches whose evaluate.py is a thin shim delegating to the shared lmms-eval
# adapter (src/eval/lmms_common/evaluate.py). Their bundles must also carry
# lmms_common/ + its src/eval deps (split_util, diag_util) so the verifier image
# is self-contained — the omni-eval image bakes the lmms-eval harness + runners,
# but NOT src/eval/lmms_common.
LMMS_BENCHES = {"mmmu_pro", "video_mmmu", "videomme_v2"}

# Video benches need a fatter verifier than the audio/image default in the
# task.toml template: the runner shards the omni model across cards (one card
# can't hold it for video) and long-video frame decoding is host-RAM heavy.
# Generated here rather than hand-patched after the fact, so regenerating a
# bundle cannot silently downgrade it back to 1 GPU / 128Gi.
VIDEO_BENCHES = {"video_mmmu", "videomme_v2", "omnivideobench"}

# mmswe (SWE-bench Multimodal) is the one non-perception bench and needs a
# DIFFERENT verifier image than the other 7: its evaluate.py is a two-stage
# seam — stage 1 GENERATE (omni base python -> patches) then stage 2 GRADE
# (a SEPARATE swebench venv; Route B daemon-free: registry manifest -> mirror
# blob + sha256 -> unshare+chroot -> run tests). So the verifier image layers a
# `swebench` venv (SWE_VENV_PY) on top of the omni-eval base, and the grader
# script (dlc_native_grade.py) must be copied into the bundle. Its verifier also
# needs egress at grade time (registry/mirror/screenshot hosts) — task.toml
# already sets allow_internet=true.
MMSWE_BENCHES = {"mmswe"}

# Claude-specific instruction clause (from original get_prompt.py)
CLAUDE_CLAUSE = """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""


@dataclass
class BenchmarkInfo:
    task_id: str           # e.g., "gsm8k"
    benchmark_name: str    # e.g., "GSM8K (Grade School Math 8K)"
    setup_note: str = ""   # Additional setup instructions


@dataclass
class ModelInfo:
    model_id: str          # HuggingFace model ID, e.g., "Qwen/Qwen3-1.7B-Base"
    short_name: str        # Short name for task IDs, e.g., "qwen3-1.7b"


# mmposttrainbench: the omni benchmarks from the eval-deliver bundle, each
# wired into the verifier via a per-bench recipe (src/docker/
# bench_recipes.sh). GPU per bench: image/audio = 1, video = 4. The eval logic
# is the bundle's own scripts (lmms-eval harness / self-runners / OVB official);
# our evaluate.py adapters normalize output to {"accuracy"} for the verifier.
BENCHMARKS = {
    "mmau": BenchmarkInfo(
        task_id="mmau",
        benchmark_name="MMAU (test-mini, audio understanding)",
        setup_note="",
    ),
    "mmar": BenchmarkInfo(
        task_id="mmar",
        benchmark_name="MMAR (audio reasoning, MCQ)",
        setup_note="",
    ),
    "jointavbench": BenchmarkInfo(
        task_id="jointavbench",
        benchmark_name="JointAVBench (joint audio-video understanding)",
        setup_note="",
    ),
    "mmmu_pro": BenchmarkInfo(
        task_id="mmmu_pro",
        benchmark_name="MMMU-Pro (standard, image understanding)",
        setup_note="",
    ),
    "video_mmmu": BenchmarkInfo(
        task_id="video_mmmu",
        benchmark_name="Video-MMMU (comprehension, video understanding)",
        setup_note="",
    ),
    "videomme_v2": BenchmarkInfo(
        task_id="videomme_v2",
        benchmark_name="Video-MME v2 (video understanding, group scoring)",
        setup_note="",
    ),
    "omnivideobench": BenchmarkInfo(
        task_id="omnivideobench",
        benchmark_name="OmniVideoBench (audio-visual video QA)",
        setup_note="",
    ),
    "mmswe": BenchmarkInfo(
        task_id="mmswe",
        benchmark_name="SWE-bench Multimodal (dev split, real resolved rate)",
        setup_note="",
    ),
}

MODELS = {
    "qwen3-omni-30b": ModelInfo(
        # Display / metadata id. The actual 68G weights are NOT pulled from
        # HF — they are bind-mounted from shared storage at run time (see src/docker/), so
        # the agent/verifier images stay model-free.
        model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        short_name="qwen3-omni-30b",
    ),
}


class PostTrainBenchAdapter:
    """Adapter to generate Harbor tasks from PostTrainBench configuration."""

    def __init__(
        self,
        output_dir: Path,
        num_hours: int = 24,
        include_claude_clause: bool = True,
    ):
        """
        Initialize the adapter.

        Args:
            output_dir: Directory where Harbor tasks will be generated.
            num_hours: Number of hours for the training task (default: 24).
            include_claude_clause: Whether to include the Claude non-interactive clause.
        """
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.include_claude_clause = include_claude_clause
        self.posttrainbench_root = POSTTRAINBENCH_ROOT

    def _read_benchmark_name(self, benchmark_id: str) -> str:
        """Read the human-readable benchmark name from benchmark.txt."""
        bench_file = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "benchmark.txt"
        if bench_file.is_file():
            return bench_file.read_text(encoding="utf-8").strip()
        # Fallback to the dataclass info
        if benchmark_id in BENCHMARKS:
            return BENCHMARKS[benchmark_id].benchmark_name
        raise FileNotFoundError(f"Benchmark file not found: {bench_file}")

    def generate_task_toml(self, task_dir: Path, benchmark_id: str = "") -> None:
        """Generate task.toml for the Harbor task."""
        # Copy template and adjust timeout based on num_hours
        template_path = TEMPLATE_DIR / "task.toml"
        target_path = task_dir / "task.toml"

        content = template_path.read_text()

        # Adjust agent timeout based on num_hours
        agent_timeout = self.num_hours * 3600  # Convert hours to seconds
        content = content.replace(
            "timeout_sec = 36000.0",
            f"timeout_sec = {float(agent_timeout)}"
        )

        # Video + mmswe benches: replace the template's audio/image verifier
        # resources. The template documents the single-GPU case (MMAU); video needs
        # model parallelism + far more host RAM for frame decoding, and mmswe's
        # stage-1 generate shards the 30B omni model and its stage-2 grade unpacks
        # per-instance rootfs into /dev/shm (host-RAM heavy).
        if benchmark_id in VIDEO_BENCHES or benchmark_id in MMSWE_BENCHES:
            old_block = (
                "# Verifier container resources. MMAU is an AUDIO benchmark: the eval-deliver\n"
                "# runner loads the whole omni model onto one card (talker disabled to save\n"
                "# VRAM), so a single GPU suffices — image/audio benches are single-GPU by the\n"
                "# eval-deliver GPU strategy; the video benches (added post-pilot) will need\n"
                "# multi-GPU model parallelism here.\n"
                "[verifier.environment]\n"
                "gpus = 1\n"
                'gpu_types = ["H100", "H800"]\n'
                "cpus = 16\n"
                "memory_mb = 131072\n"
            )
            if benchmark_id in MMSWE_BENCHES:
                new_block = (
                    "# Verifier container resources. This is the mmswe (SWE-bench Multimodal)\n"
                    "# bench: stage-1 generate shards the 30B omni model across cards, and\n"
                    "# stage-2 grade unpacks each instance's official rootfs into /dev/shm and\n"
                    "# runs its test suite (host-RAM heavy) — hence 8 GPUs + 1024Gi RAM,\n"
                    "# matching what the docker orchestration layer (src/docker/) provisions.\n"
                    "[verifier.environment]\n"
                    "gpus = 8\n"
                    'gpu_types = ["H100", "H800"]\n'
                    "cpus = 64\n"
                    "memory_mb = 1048576\n"
                )
            else:
                new_block = (
                    "# Verifier container resources. This is a VIDEO benchmark: the eval-deliver\n"
                    "# runner shards the omni model across multiple cards (a single card can't hold\n"
                    "# it for video), and long-video frame decoding is host-RAM heavy — hence 8 GPUs\n"
                    "# + 1024Gi RAM, matching what the docker orchestration layer (src/docker/) actually\n"
                    "# provisions for video. (Audio/image benches stay single-GPU / 128Gi.)\n"
                    "[verifier.environment]\n"
                    "gpus = 8\n"
                    'gpu_types = ["H100", "H800"]\n'
                    "cpus = 64\n"
                    "memory_mb = 1048576\n"
                )
            if old_block not in content:
                raise RuntimeError(
                    "task.toml template drifted: cannot apply the video/mmswe verifier "
                    "resource override. Update VIDEO_BENCHES/MMSWE_BENCHES handling in "
                    "generate_task_toml() to match template/task.toml."
                )
            content = content.replace(old_block, new_block)

        # mmswe grading is per-instance (pull an official image's layers over a
        # jittery mirror, then run its test suite under unshare+chroot) across the
        # whole eval subset — the perception benches' 3h verifier budget is far too
        # short. Bump to 12h. Guarded string must match template/task.toml [verifier].
        if benchmark_id in MMSWE_BENCHES:
            if "timeout_sec = 10800.0" not in content:
                raise RuntimeError(
                    "task.toml template drifted: cannot bump the mmswe verifier "
                    "timeout. Expected '[verifier] timeout_sec = 10800.0'."
                )
            content = content.replace("timeout_sec = 10800.0", "timeout_sec = 43200.0")

        # For arenahardwriting/healthbench, agents need OPENAI_API_KEY
        # during their run (to run evaluate.py which uses OpenAI judge)
        if benchmark_id in ("arenahardwriting", "healthbench"):
            content += '\n[agent.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'

        target_path.write_text(content)

    def generate_instruction(
        self,
        task_dir: Path,
        model_info: ModelInfo,
        benchmark_info: BenchmarkInfo,
        benchmark_id: str = "",
    ) -> None:
        """Generate instruction.md for the Harbor task."""
        template_path = TEMPLATE_DIR / "instruction.md"
        target_path = task_dir / "instruction.md"

        content = template_path.read_text()

        # Fill in placeholders
        content = content.replace("{model}", model_info.model_id)
        content = content.replace("{benchmark}", benchmark_info.benchmark_name)
        content = content.replace("{num_hours}", str(self.num_hours))
        content = content.replace("{setup_other}", benchmark_info.setup_note)

        # OpenAI restriction for benchmarks that provide OPENAI_API_KEY to agents
        if benchmark_id in ("arenahardwriting", "healthbench"):
            content = content.replace(
                "{openai_restriction}",
                "- IMPORTANT: You are NOT allowed to use the OpenAI API for anything but the evaluation script.\n"
            )
        else:
            content = content.replace("{openai_restriction}", "")

        if self.include_claude_clause:
            content += CLAUDE_CLAUSE

        target_path.write_text(content)

    def generate_timer_sh(self, env_dir: Path) -> None:
        """Generate timer.sh script that tracks remaining time.

        Reads the start timestamp from the absolute path /timer_start, which
        is written by the task.toml healthcheck immediately before the agent
        launches. Using an absolute path makes the timer immune to the
        agent's `cd`s (the previous sentinel-file approach used
        `dirname "$0"` which resolved differently per cwd).
        """
        timer_script = f"""#!/bin/bash

NUM_HOURS={self.num_hours}
START_FILE="/timer_start"

if [ ! -f "$START_FILE" ]; then
    echo "Timer not initialized (healthcheck has not run yet)."
    exit 1
fi

START_DATE=$(cat "$START_FILE")
DEADLINE=$((START_DATE + NUM_HOURS * 3600))
NOW=$(date +%s)
REMAINING=$((DEADLINE - NOW))

if [ $REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=$((REMAINING / 3600))
    MINUTES=$(((REMAINING % 3600) / 60))
    printf "%d:%02d\\n" $HOURS $MINUTES
fi
"""
        timer_path = env_dir / "timer.sh"
        timer_path.write_text(timer_script)
        timer_path.chmod(0o755)

    def generate_environment(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Generate the environment/ directory: Dockerfile + agent runtime."""
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)

        # Copy Dockerfile template and .dockerignore
        shutil.copy(
            TEMPLATE_DIR / "environment" / "Dockerfile",
            env_dir / "Dockerfile"
        )
        dockerignore_src = TEMPLATE_DIR / "environment" / ".dockerignore"
        if dockerignore_src.exists():
            shutil.copy(dockerignore_src, env_dir / ".dockerignore")

        # Build-context support files (entrypoint, system monitor,
        # requirements-direct). Shared with tests/ — see _copy_build_context_support.
        self._copy_build_context_support(env_dir)

        # Eval files: evaluate.py, templates/, optional evaluation_code/
        # and task_context contents, plus metadata.json. The agent gets these in
        # /home/agent/workspace (via the Dockerfile's `COPY .`) for fast iteration
        # during training. agent_side=True withholds contamination_judge.py — the
        # verifier's own probe, which the agent has no legitimate use for.
        self._copy_eval_files(
            env_dir, benchmark_id, model_info, benchmark_info, agent_side=True
        )
        # Regeneration is in-place over an existing bundle, so a file that USED to
        # be copied here would otherwise survive as a stale leak. Delete explicitly.
        for withheld in ("contamination_judge.py",):
            (env_dir / withheld).unlink(missing_ok=True)

        # timer.sh — agent reads it during the run. Verifier doesn't need it.
        self.generate_timer_sh(env_dir)

    def _copy_build_context_support(self, target_dir: Path) -> None:
        """Copy entrypoint.sh + system_monitor.sh + requirements-direct.txt
        into a Dockerfile build context.

        Both environment/ (agent) and tests/ (verifier under harbor's
        separate-verifier mode) use the same Dockerfile structure and
        need these files at build time. The canonical sources live under
        template/environment/ and containers/.
        """
        # entrypoint.sh — Dockerfile installs it at /usr/local/bin/ and
        # sets it as ENTRYPOINT so its stdout becomes Modal's live log
        # stream (see template/environment/entrypoint.sh).
        entrypoint_src = TEMPLATE_DIR / "environment" / "entrypoint.sh"
        entrypoint_dst = target_dir / "entrypoint.sh"
        shutil.copy(entrypoint_src, entrypoint_dst)
        entrypoint_dst.chmod(0o755)

        # system_monitor.sh — kicked off by entrypoint.sh as a background
        # daemon; ports condor's src/utils/system_monitor.sh.
        monitor_src = TEMPLATE_DIR / "environment" / "system_monitor.sh"
        monitor_dst = target_dir / "system_monitor.sh"
        shutil.copy(monitor_src, monitor_dst)
        monitor_dst.chmod(0o755)

        # containers/requirements-direct.txt — the Dockerfile pins ML
        # dependencies from this file for the shared agent environment.
        reqs_src = self.posttrainbench_root / "containers" / "requirements-direct.txt"
        if not reqs_src.exists():
            raise FileNotFoundError(
                f"requirements-direct.txt not found at {reqs_src}; "
                f"the Dockerfile expects it in the build context."
            )
        shutil.copy(reqs_src, target_dir / "requirements-direct.txt")

    def _copy_eval_files(
        self,
        target_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
        agent_side: bool = False,
    ) -> None:
        """Copy the evaluation pipeline files into target_dir.

        Used for both:
          - environment/ (so the agent has them in /home/agent/workspace
            for iterative testing during training) -- pass agent_side=True
          - tests/ (so the verifier runs against an untampered copy that
            Harbor uploads only after the agent process exits)

        agent_side=True withholds the files that exist only to police the
        agent. Everything the agent legitimately needs to self-evaluate
        (evaluate.py, templates/, metadata.json, split_util.py, diag_util.py
        and for lmms benches lmms_common/) is still copied --
        instruction.md tells the agent to query the benchmark via
        evaluate.py, so withholding those would break the paradigm, not
        protect it. What it does withhold is contamination_judge.py: only
        tests/test.sh invokes it, the agent has no use for it, and handing
        an agent the exact contamination probe it will be measured by is a
        one-sided leak.

        Files copied:
          - evaluate.py            (benchmark-specific)
          - templates/             (chat templates for all model families)
          - evaluation_code/       (arenahardwriting, healthbench only)
          - task_context/<*>       (bfcl has bfcl_evaluation_code.py)
          - contamination_judge.py (judge prompt builder; verifier only)
          - metadata.json          (benchmark + model info for verifier)
        """
        # evaluate.py
        eval_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "evaluate.py"
        if not eval_src.exists():
            raise FileNotFoundError(f"evaluate.py not found: {eval_src}")
        shutil.copy(eval_src, target_dir / "evaluate.py")

        eval_root = self.posttrainbench_root / "src" / "eval"

        # split_util.py / diag_util.py go into EVERY bundle, not just the lmms ones.
        # The adapters reach them with a `dirname x3 -> src/eval` hop, which resolves
        # to "/" once the bundle is staged flat as /tests/evaluate.py. For mmau/mmar/
        # jointavbench the diag_util import is wrapped in try/except so it merely
        # skipped diagnostics; omnivideobench does a HARD `from split_util import
        # keep` and died with ModuleNotFoundError *after* generating all 1000 items
        # (3h35m of 8-GPU work). Python puts the script's own directory on sys.path,
        # so a copy at target_dir root is what makes the flat staging importable.
        for dep in ("split_util.py", "diag_util.py", "eval_util.py"):
            dep_src = eval_root / dep
            if not dep_src.exists():
                raise FileNotFoundError(f"eval helper not found: {dep_src}")
            shutil.copy(dep_src, target_dir / dep)

        # mmswe additionally needs its grader script (stage 2): evaluate.py resolves
        # it via DEFAULT_GRADER = <dir of evaluate.py>/dlc_native_grade.py, which
        # once staged flat is target_dir/dlc_native_grade.py. The omni-eval base
        # bakes the stage-1 runner (/opt/eval/runners/run_mmswe_official.py) but NOT
        # this grader, so it must travel in the bundle. Copied to both sides for
        # symmetry (public file; harmless in the agent workspace).
        if benchmark_id in MMSWE_BENCHES:
            grader_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "dlc_native_grade.py"
            if not grader_src.exists():
                raise FileNotFoundError(f"mmswe grader not found: {grader_src}")
            shutil.copy(grader_src, target_dir / "dlc_native_grade.py")

        # lmms-eval-class benches additionally need lmms_common/, which the shim
        # evaluate.py delegates to (it inserts its parent dir on sys.path, hence the
        # deps above sitting at target_dir root).
        if benchmark_id in LMMS_BENCHES:
            lc_src = eval_root / "lmms_common"
            if not lc_src.is_dir():
                raise FileNotFoundError(f"lmms_common not found: {lc_src}")
            shutil.copytree(lc_src, target_dir / "lmms_common", dirs_exist_ok=True)

        # templates/
        templates_src = self.posttrainbench_root / "src" / "eval" / "templates"
        if not templates_src.exists():
            raise FileNotFoundError(f"templates directory not found: {templates_src}")
        shutil.copytree(templates_src, target_dir / "templates", dirs_exist_ok=True)

        # evaluation_code/ (arenahardwriting, healthbench)
        eval_code_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "evaluation_code"
        if eval_code_src.is_dir():
            shutil.copytree(eval_code_src, target_dir / "evaluation_code", dirs_exist_ok=True)

        # task_context/* (bfcl has bfcl_evaluation_code.py)
        task_context_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "task_context"
        if task_context_src.is_dir():
            for item in task_context_src.iterdir():
                dst = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(item, dst)

        # contamination judge script (kept in template/environment/ as the
        # canonical source). VERIFIER ONLY: tests/test.sh is its sole caller, so
        # shipping it into the agent workspace leaked the anti-cheat probe to the
        # party being probed. See the agent_side note in the docstring.
        if not agent_side:
            judge_src = TEMPLATE_DIR / "environment" / "contamination_judge.py"
            if judge_src.exists():
                shutil.copy(judge_src, target_dir / "contamination_judge.py")

        # metadata.json
        metadata = {
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_info.benchmark_name,
            "model_id": model_info.model_id,
            "model_short_name": model_info.short_name,
            "num_hours": self.num_hours,
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def generate_tests(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Generate the tests/ directory.

        Under harbor 0.7.0's separate-verifier mode, tests/ doubles as
        the verifier image's build context: harbor builds it into a
        container the agent never touches, then transfers configured
        artifacts in at runtime. The image must self-contain test.sh and
        everything test.sh reads — harbor does not upload tests/ at
        runtime for separate verifier envs.

        Files placed here:
          - Dockerfile      builds the verifier image
          - test.sh         the verifier orchestrator (baked in via COPY .)
          - entrypoint.sh   PID-1 streamer (matches agent env)
          - system_monitor.sh  background system monitor
          - requirements-direct.txt  pinned ML deps for the Dockerfile
          - evaluate.py + templates/ + evaluation_code/ + task_context/*
            + contamination_judge.py + metadata.json — the eval pipeline
        """
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        # Verifier image Dockerfile (canonical source: template/tests/Dockerfile).
        # mmswe uses a dedicated variant that layers a `swebench` venv on the
        # omni-eval base and bakes the SWE_* env the two-stage evaluate.py reads;
        # the 7 perception benches all share the generic verifier Dockerfile.
        verifier_dockerfile = "Dockerfile.mmswe" if benchmark_id in MMSWE_BENCHES else "Dockerfile"
        shutil.copy(
            TEMPLATE_DIR / "tests" / verifier_dockerfile,
            tests_dir / "Dockerfile",
        )

        # Verifier orchestrator (test.sh)
        test_sh_src = TEMPLATE_DIR / "tests" / "test.sh"
        test_sh_dst = tests_dir / "test.sh"
        shutil.copy(test_sh_src, test_sh_dst)
        test_sh_dst.chmod(0o755)

        # Build-context support files (same set the agent env needs).
        self._copy_build_context_support(tests_dir)

        # Eval pipeline (also baked into the agent workspace via
        # environment/, but the verifier reads from /tests/ where these
        # land via the verifier Dockerfile's `COPY .`).
        self._copy_eval_files(tests_dir, benchmark_id, model_info, benchmark_info)

    def generate_task(
        self,
        benchmark_id: str,
        model_key: str,
    ) -> Path:
        """
        Generate a complete Harbor task for a benchmark + model combination.

        Args:
            benchmark_id: The benchmark ID (e.g., "gsm8k").
            model_key: The model key (e.g., "qwen3-1.7b").

        Returns:
            Path to the generated task directory.
        """
        if benchmark_id not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {list(BENCHMARKS.keys())}")
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

        benchmark_info = BENCHMARKS[benchmark_id]
        model_info = MODELS[model_key]

        # Try to get actual benchmark name from file
        try:
            benchmark_info = BenchmarkInfo(
                task_id=benchmark_info.task_id,
                benchmark_name=self._read_benchmark_name(benchmark_id),
                setup_note=benchmark_info.setup_note,
            )
        except FileNotFoundError:
            pass  # Use default from dataclass

        # Create task directory
        task_id = f"mmposttrainbench-{benchmark_id}-{model_info.short_name}"
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating task: {task_id}")

        # Generate all components
        self.generate_task_toml(task_dir, benchmark_id)
        self.generate_instruction(task_dir, model_info, benchmark_info, benchmark_id)
        self.generate_environment(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_tests(task_dir, benchmark_id, model_info, benchmark_info)

        print(f"Task generated at: {task_dir}")
        return task_dir

    def generate_all_tasks(self) -> list[Path]:
        """Generate tasks for all benchmark + model combinations."""
        tasks = []
        for benchmark_id in BENCHMARKS:
            for model_key in MODELS:
                task_dir = self.generate_task(benchmark_id, model_key)
                tasks.append(task_dir)
        return tasks


def list_available_tasks() -> list[str]:
    """List all available task combinations."""
    tasks = []
    for benchmark_id in BENCHMARKS:
        for model_key in MODELS:
            task_id = f"mmposttrainbench-{benchmark_id}-{MODELS[model_key].short_name}"
            tasks.append(task_id)
    return tasks
