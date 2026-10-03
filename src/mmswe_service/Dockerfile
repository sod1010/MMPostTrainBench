# Build context: repository root. Build separately with approved train/eval bases.
# Runtime image references in policy must use the resulting registry digest or local immutable image ID.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY src/mmswe_service/entry.sh src/mmswe_service/generate.py /opt/mmptb-service/
COPY src/eval/split_util.py src/eval/eval_util.py /opt/mmptb-service/eval/
COPY eval_omni/runners/run_mmswe_official.py /opt/eval/runners/run_mmswe_official.py
RUN chmod 0555 /opt/mmptb-service/entry.sh
