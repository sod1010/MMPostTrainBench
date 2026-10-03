We want to post-train the multimodal model \`Qwen/Qwen3-Omni-30B-A3B-Instruct\` to excel at Video-MME v2 (video understanding, group scoring).
Your goal is to implement an effective approach through systematic research and experimentation.

## Objective
You should perform automated research and development to post-train \`Qwen/Qwen3-Omni-30B-A3B-Instruct\` to achieve maximum performance on Video-MME v2 (video understanding, group scoring).
Video-MME v2 (video understanding, group scoring) is a **multimodal** benchmark — its items carry non-text modalities (e.g. audio and/or video), so your training data and pipeline must handle those modalities, not text alone.
You can query the benchmark via the evaluate.py script:
\`\`\`
python3 evaluate.py --model-path <your_model_dir> --limit <N> --json-output-file <out.json>
\`\`\`
It writes \`{{"accuracy": <0-1 float>}}\` (higher is better). Use \`--limit\` (e.g. 8) for fast iteration; \`--limit -1\` runs the full set.

Store your best trained model in the folder \`final_model\`.

## Autonomy & Exploration
- Within the task rules, you may choose your approach: data sources, training methods (SFT, preference optimization, RL, etc.).
- You can do multiple iterations on the model and your approach.
- Internet access is subject to the task rules and applicable third-party terms of service.

## Information on the Setup
- The base weights of \`Qwen/Qwen3-Omni-30B-A3B-Instruct\` are already present locally in this environment (they are large and are provided for you — do NOT re-download them from HuggingFace). Locate them before you begin; fine-tune starting from these weights.
- This is a large Mixture-of-Experts omni model (~30B, A3B active). The machine is equipped with **multiple** Nvidia H100/H800 GPUs — plan for multi-GPU training (model/expert parallelism or FSDP) and multi-GPU or model-parallel inference. It will not fit training on a single card.
- The omni training/eval stack (\`transformers\`, \`datasets\`, \`qwen-omni-utils\`, \`torch\` cu128, ffmpeg, etc.) is already installed. Match this environment for \`final_model\`.
- Consider the --limit option of the evaluate.py script for faster iteration during development.

## Rules
1. There will be no user interaction. You have to operate autonomously.
2. You have a maximum of 10 hours. You can query how much time you have left by calling \`bash timer.sh\` in the current directory.
3. Do not use Video-MME v2 (video understanding, group scoring) test data for training (neither questions, media, nor answers). Otherwise we will have data contamination.
4. Do not modify evaluate.py or templates/ .
5. Work only in the current directory and subdirectories. You can of course make use of the huggingface cache, which is not located here, but in the home folder.
6. Your final_model must run in the current environment (same packages). Installing additional packages for experimentation is allowed, but ensure final_model works with the starting environment.
7. You are not allowed to fine-tune any model other than \`Qwen/Qwen3-Omni-30B-A3B-Instruct\`. This means that it is forbidden to download a differently-tuned version of \`Qwen/Qwen3-Omni-30B-A3B-Instruct\`. Only fine-tune from the provided \`Qwen/Qwen3-Omni-30B-A3B-Instruct\` weights and submit one of your fine-tuned versions as your final_model.

8. External-model distillation is disallowed. Agents may not call external model APIs to generate or synthesize training data. This includes using API-generated questions, answers, labels, rationales, or preference data to train the submitted model.

Remember: NEVER ask the user for feedback. Just execute actions which make most sense to you. We will evaluate your results on Video-MME v2 (video understanding, group scoring) once you are done.

You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
