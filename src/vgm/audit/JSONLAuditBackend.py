"""JSONL file-based audit backend."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from threading import Lock

from pydantic import ValidationError

from .AuditBackend import AuditBackend
from ..schemas import AuditEntry
from ..config import AuditConfig


class JSONLAuditBackend(AuditBackend):
    """File-based JSONL audit logging with rotation support."""

    def __init__(self, config: AuditConfig):
        self.config = config
        self.log_dir = Path(config.log_dir or os.path.expanduser("~/.vgm/logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / "audit.jsonl"
        self._write_lock = Lock()

    def _should_rotate(self) -> bool:
        """Check if log file should be rotated."""
        if not self.current_log_file.exists():
            return False

        file_size_mb = self.current_log_file.stat().st_size / (1024 * 1024)
        return file_size_mb >= self.config.rotation_size_mb

    def _rotate_log(self) -> None:
        """Rotate log file based on configured period."""
        if not self.current_log_file.exists():
            return

        timestamp = datetime.now(timezone.utc)
        if self.config.rotation_period == "daily":
            suffix = timestamp.strftime("%Y-%m-%d")
        elif self.config.rotation_period == "weekly":
            suffix = timestamp.strftime("%Y-W%U")
        else:  # monthly
            suffix = timestamp.strftime("%Y-%m")

        rotated_file = self.log_dir / f"audit-{suffix}.jsonl"

        # If rotated file exists, append to it; otherwise rename current
        if rotated_file.exists():
            with open(rotated_file, "a") as dest, open(self.current_log_file, "r") as src:
                dest.write(src.read())
            self.current_log_file.unlink()
        else:
            self.current_log_file.rename(rotated_file)

    def log_operation(self, entry: AuditEntry) -> None:
        """Append audit entry to JSONL file."""
        with self._write_lock:
            if self._should_rotate():
                self._rotate_log()

            with open(self.current_log_file, "a") as f:
                json_line = entry.model_dump_json() + "\n"
                f.write(json_line)

    def get_recent(self, limit: int = 50) -> List[AuditEntry]:
        """Get most recent N entries."""
        if not self.current_log_file.exists():
            return []

        entries = []
        with open(self.current_log_file, "r") as f:
            lines = f.readlines()
            # Take last N lines
            for line in lines[-limit:]:
                try:
                    entries.append(AuditEntry.model_validate_json(line))
                except (ValidationError, ValueError):
                    # Skip malformed lines - log parse errors are expected in corrupted files
                    continue
        return entries

    def get_by_session(self, session_id: str) -> List[AuditEntry]:
        """Get all entries for a session (scans entire file)."""
        if not self.current_log_file.exists():
            return []

        entries = []
        with open(self.current_log_file, "r") as f:
            for line in f:
                try:
                    entry = AuditEntry.model_validate_json(line)
                    if entry.session_id == session_id:
                        entries.append(entry)
                except (ValidationError, ValueError):
                    # Skip malformed lines
                    continue
        return entries

    def get_entity_history(self, entity_id: str) -> List[AuditEntry]:
        """Get all operations affecting an entity (scans entire file)."""
        if not self.current_log_file.exists():
            return []

        entries = []
        with open(self.current_log_file, "r") as f:
            for line in f:
                try:
                    entry = AuditEntry.model_validate_json(line)
                    if entity_id in entry.affected_entities:
                        entries.append(entry)
                except (ValidationError, ValueError):
                    # Skip malformed lines
                    continue
        return entries
