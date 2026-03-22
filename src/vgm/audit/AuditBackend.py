"""Abstract base class for audit backends."""

from abc import ABC, abstractmethod
from typing import List

from ..schemas import AuditEntry


class AuditBackend(ABC):
    """Abstract base class for audit log backends."""

    @abstractmethod
    def log_operation(self, entry: AuditEntry) -> None:
        """Log a memory operation."""
        pass

    @abstractmethod
    def get_recent(self, limit: int = 50) -> List[AuditEntry]:
        """Get recent log entries."""
        pass

    @abstractmethod
    def get_by_session(self, session_id: str) -> List[AuditEntry]:
        """Get all logs for a specific session."""
        pass

    @abstractmethod
    def get_entity_history(self, entity_id: str) -> List[AuditEntry]:
        """Get all operations affecting a specific entity."""
        pass
