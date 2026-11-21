# One Dockerfile, three images, each with only the dependency groups it needs (pyproject.toml):
#   router     the ML service       (serve)                 complaint-routing/router:<commit>
#   assistant  the LLM service      (serve + llm)           complaint-routing/assistant:<commit>
#   monitor    drift job, trigger   (monitor)               complaint-routing/monitor:<commit>
# Built from the repository root by `make images` (inside Minikube), tag = the git commit.
# Versions come from uv.lock (--frozen), so the image has what the commit's lock file says.
# What is in an image besides the code: Python 3.12, the locked packages, config/ (queue mapping,
# llm.yaml defaults), prompts/ and, for the assistant, the Ask CFPB corpus (its md5 is part of the
# index version) and the embedding model. The routing model is NOT in any image: the service
# loads it from the registry by alias at start-up.

FROM python:3.12-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH PROJECT_ROOT=/app PYTHONUNBUFFERED=1 HOME=/tmp \
    GIT_PYTHON_REFRESH=quiet
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./

FROM base AS router
RUN uv sync --frozen --no-default-groups --group serve --no-install-project
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-default-groups --group serve --no-editable
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=$GIT_COMMIT org.opencontainers.image.title=router
ENV GIT_COMMIT=$GIT_COMMIT
USER 65534
EXPOSE 8000
CMD ["uvicorn", "complaint_routing.serving.router:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS assistant
RUN uv sync --frozen --no-default-groups --group serve --group llm --no-install-project
COPY src ./src
COPY config ./config
COPY prompts ./prompts
COPY data/corpus ./data/corpus
RUN uv sync --frozen --no-default-groups --group serve --group llm --no-editable
# The embedding model named in config/llm.yaml, downloaded once at build time; at run time the
# service is offline from Hugging Face (a different embedding model is a new index and image).
ENV HF_HOME=/opt/hf
RUN python -c "from complaint_routing.llm import settings, index; \
index.embedder(settings.load()['retrieval']['embedding_model'])" && chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=$GIT_COMMIT org.opencontainers.image.title=assistant
ENV GIT_COMMIT=$GIT_COMMIT
USER 65534
EXPOSE 8000
CMD ["uvicorn", "complaint_routing.serving.assistant:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS monitor
RUN uv sync --frozen --no-default-groups --group monitor --no-install-project
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-default-groups --group monitor --no-editable
ARG GIT_COMMIT=unknown
LABEL org.opencontainers.image.revision=$GIT_COMMIT org.opencontainers.image.title=monitor
ENV GIT_COMMIT=$GIT_COMMIT
USER 65534
CMD ["python", "-m", "complaint_routing.monitoring.drift"]
