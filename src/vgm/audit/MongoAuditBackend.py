"""MongoDB-based audit backend."""

from typing import List

from .AuditBackend import AuditBackend
from ..schemas import AuditEntry
from ..config import AuditConfig


class MongoAuditBackend(AuditBackend):
    """MongoDB-based audit logging with optional TTL."""

    def __init__(self, config: AuditConfig):
        try:
            from pymongo import MongoClient, ASCENDING, IndexModel
            from pymongo.errors import ConnectionFailure
        except ImportError:
            raise ImportError(
                "MongoDB backend requires pymongo. Install with: pip install vector-graph-memory[mongodb]"
            )

        self.config = config
        if not config.connection_string:
            raise ValueError("MongoDB backend requires connection_string in AuditConfig")

        self.client = MongoClient(config.connection_string)
        self.db = self.client[config.database]
        self.collection = self.db[config.collection]

        # Create indexes
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes for efficient querying."""
        from pymongo import ASCENDING, IndexModel

        indexes = [
            IndexModel([("timestamp", ASCENDING)]),
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("project_id", ASCENDING)]),
            IndexModel([("affected_entities", ASCENDING)]),
        ]
        self.collection.create_indexes(indexes)

        # Create TTL index if configured
        if self.config.ttl_days:
            self.collection.create_index(
                "timestamp",
                expireAfterSeconds=self.config.ttl_days * 24 * 60 * 60
            )

    def log_operation(self, entry: AuditEntry) -> None:
        """Insert audit entry into MongoDB."""
        doc = entry.model_dump()
        self.collection.insert_one(doc)

    def get_recent(self, limit: int = 50) -> List[AuditEntry]:
        """Get most recent N entries."""
        docs = self.collection.find().sort("timestamp", -1).limit(limit)
        return [AuditEntry.model_validate(doc) for doc in docs]

    def get_by_session(self, session_id: str) -> List[AuditEntry]:
        """Get all entries for a session."""
        docs = self.collection.find({"session_id": session_id}).sort("timestamp", 1)
        return [AuditEntry.model_validate(doc) for doc in docs]

    def get_entity_history(self, entity_id: str) -> List[AuditEntry]:
        """Get all operations affecting an entity."""
        docs = self.collection.find(
            {"affected_entities": entity_id}
        ).sort("timestamp", 1)
        return [AuditEntry.model_validate(doc) for doc in docs]

    def close(self) -> None:
        """Close MongoDB connection."""
        self.client.close()
