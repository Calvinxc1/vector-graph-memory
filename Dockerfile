# Vector Graph Memory API Server
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies once in the shared base image.
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*


FROM base AS deps

# Resolve third-party dependencies from project metadata only so source edits
# do not invalidate the expensive dependency layer.
COPY pyproject.toml .
RUN python - <<'PY' > /tmp/requirements-api.txt
import tomllib

with open("pyproject.toml", "rb") as handle:
    project = tomllib.load(handle)["project"]

requirements = list(project.get("dependencies", []))
requirements.extend(project.get("optional-dependencies", {}).get("api", []))

print("\n".join(requirements))
PY
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements-api.txt


FROM base AS runtime

COPY --from=deps /install /usr/local
COPY pyproject.toml .
COPY src/ src/

# Install only the local package in the final image. Third-party dependencies
# come from the cached deps stage above.
RUN pip install --no-cache-dir --no-deps .

# Create directory for audit logs
RUN mkdir -p /app/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the API server
CMD ["uvicorn", "vgm.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
