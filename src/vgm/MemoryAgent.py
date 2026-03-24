"""Memory-enhanced AI agent with vector-graph storage."""

from typing import Optional, List, Dict, Any, Literal
from uuid import uuid4

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic import BaseModel
from qdrant_client import QdrantClient
from gremlin_python.driver import client as gremlin_client
from pydantic_ai import EmbeddingModel

from .VectorGraphStore import VectorGraphStore, _run_async
from .schemas import NodeMetadata, EdgeMetadata, AuditEntry
from .config import MemoryConfig, MemoryTriggerConfig, AuditConfig, VectorGraphConfig
from .audit import AuditBackend, JSONLAuditBackend, MongoAuditBackend


class AgentDependencies(BaseModel):
    """Dependencies passed to agent tools."""

    memory_agent: Any  # Reference to MemoryAgent instance

    class Config:
        arbitrary_types_allowed = True


class MemoryAgent:
    """AI agent with persistent vector-graph memory.

    This agent wraps PydanticAI with memory capabilities, allowing it to:
    - Search semantic memory via vector similarity
    - Store entities and relationships in a graph
    - Track memory operations via audit logs
    - Propose memory additions with user confirmation
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        janus_client: gremlin_client.Client,
        embedding_model: EmbeddingModel,
        llm_model: str | Model,
        system_prompt: str,
        memory_config: MemoryConfig,
        audit_config: Optional[AuditConfig] = None,
        vector_config: Optional[VectorGraphConfig] = None,
        trigger_config: Optional[MemoryTriggerConfig] = None,
    ):
        """Initialize memory-enhanced agent.

        Args:
            qdrant_client: Qdrant client instance
            janus_client: JanusGraph Gremlin client instance
            embedding_model: PydanticAI EmbeddingModel for embeddings
            llm_model: LLM model identifier (e.g., 'openai:gpt-4')
            system_prompt: Base system prompt for the agent
            memory_config: Memory behavior configuration
            audit_config: Audit logging configuration (optional)
            vector_config: Vector/graph database configuration (optional)
            trigger_config: Memory trigger configuration (optional)
        """
        self.memory_config = memory_config
        self.trigger_config = trigger_config or MemoryTriggerConfig(
            mode="phrase", trigger_phrase="check for memory items"
        )
        self.audit_config = audit_config or AuditConfig()

        # Initialize storage layer
        self.store = VectorGraphStore(
            qdrant_client=qdrant_client,
            janus_client=janus_client,
            embedding_model=embedding_model,
            config=vector_config,
        )

        # Initialize audit backend
        self.audit = self._create_audit_backend(self.audit_config)

        # Session tracking
        self._current_session_id: Optional[str] = None
        self._message_count: int = 0

        # Pending memory proposals (for user confirmation)
        self.pending_proposals: Dict[str, Dict[str, Any]] = {}

        # Build system prompt with memory instructions
        full_system_prompt = self._build_system_prompt(system_prompt)

        # Create PydanticAI agent with tools
        self.agent = Agent(
            llm_model,
            deps_type=AgentDependencies,
            system_prompt=full_system_prompt,
        )

        # Register memory tools
        self._register_tools()

    @staticmethod
    def _create_audit_backend(config: AuditConfig) -> AuditBackend:
        """Create appropriate audit backend based on config."""
        if config.backend == "jsonl":
            return JSONLAuditBackend(config)
        elif config.backend == "mongodb":
            return MongoAuditBackend(config)
        else:
            raise ValueError(f"Unknown audit backend: {config.backend}")

    def _build_system_prompt(self, base_prompt: str) -> str:
        """Build full system prompt with memory instructions."""
        memory_instructions = f"""

## Memory System

You have access to a vector-graph memory system for persistent storage.

**Use Case:** {self.memory_config.use_case_description}

**Memory Threshold:** {self.memory_config.memory_threshold_description}

**Available Memory Tools:**
- `search_memory(query, limit)` - Find semantically similar content in memory
- `propose_memory_addition(content, entity_type, relationships)` - Suggest storing new information
- `get_memory_context(node_id, traversal_steps)` - Explore relationships via graph traversal

When asked about your capabilities or "what tools" you have, explain these memory management tools.

**Important Rules:**
1. Before proposing new memory, ALWAYS search for similar existing content first
2. When proposing memory, ask the user: "Should I add [description] to memory?"
3. NEVER add to memory without explicit user confirmation
4. After successful memory operation, provide a brief status: "Successfully added [item] to memory"
5. If similar content exists, ask if the user wants to update existing or create new

**Project ID:** {self.memory_config.project_id}
"""
        return base_prompt + memory_instructions

    def _register_tools(self) -> None:
        """Register memory tools with the agent."""

        @self.agent.tool
        def search_memory(
            ctx: RunContext[AgentDependencies],
            query: str,
            limit: int = 5,
        ) -> str:
            """Search the memory for semantically similar content.

            Args:
                ctx: Agent runtime context
                query: Natural language search query
                limit: Maximum number of results (default: 5)

            Returns:
                Formatted search results
            """
            results = ctx.deps.memory_agent.store.search_similar_nodes(
                content=query,
                limit=limit,
                project_id=ctx.deps.memory_agent.memory_config.project_id,
            )

            if not results:
                return f"No similar content found in memory for: '{query}'"

            formatted = []
            for i, result in enumerate(results, 1):
                formatted.append(
                    f"{i}. [{result.node_type}] {result.content[:200]}... "
                    f"(similarity: {result.similarity_score:.2f})"
                )

            return "\n".join(formatted)

        @self.agent.tool
        def propose_memory_addition(
            ctx: RunContext[AgentDependencies],
            content: str,
            entity_type: str,
            relationships: Optional[List[Dict[str, str]]] = None,
        ) -> str:
            """Propose adding content to memory (requires user confirmation).

            Args:
                ctx: Agent runtime context
                content: The content to store
                entity_type: Type of entity (e.g., 'job', 'company', 'person')
                relationships: Optional list of relationships to create
                    Each relationship should have: {"to_node_id": "...", "type": "...", "description": "..."}

            Returns:
                Proposal message for user confirmation
            """
            agent_instance = ctx.deps.memory_agent

            # Search for duplicates
            similar = agent_instance.store.search_similar_nodes(
                content=content,
                threshold=agent_instance.memory_config.similarity_threshold,
                limit=3,
                project_id=agent_instance.memory_config.project_id,
            )

            # Generate proposal ID
            proposal_id = str(uuid4())

            # Store proposal for later execution
            agent_instance.pending_proposals[proposal_id] = {
                "content": content,
                "entity_type": entity_type,
                "relationships": relationships or [],
                "similar_nodes": similar,
            }

            # Build proposal message
            if similar:
                proposal = f"Found {len(similar)} similar item(s) in memory:\n"
                for i, sim in enumerate(similar, 1):
                    proposal += f"  {i}. [{sim.node_type}] {sim.content[:100]}... (similarity: {sim.similarity_score:.2f})\n"
                proposal += f"\nShould I add '{content[:100]}...' as a new {entity_type} to memory, or update one of the existing items?"
            else:
                proposal = (
                    f"Should I add '{content[:100]}...' as a {entity_type} to memory?"
                )

            return proposal

        @self.agent.tool
        def get_memory_context(
            ctx: RunContext[AgentDependencies],
            node_id: str,
            traversal_steps: str = "both().limit(10)",
        ) -> str:
            """Get context around a memory node via graph traversal.

            Args:
                ctx: Agent runtime context
                node_id: Starting node ID
                traversal_steps: Gremlin traversal steps (default: "both().limit(10)")

            Returns:
                Formatted traversal results
            """
            try:
                results = ctx.deps.memory_agent.store.traverse_from_node(
                    node_id=node_id,
                    gremlin_steps=traversal_steps,
                )

                if not results:
                    return f"No connected nodes found for {node_id}"

                formatted = []
                for i, node in enumerate(results, 1):
                    if isinstance(node, dict) and "content" in node:
                        formatted.append(
                            f"{i}. [{node.get('node_type', 'unknown')}] {node['content'][:200]}..."
                        )
                    else:
                        formatted.append(f"{i}. {node}")

                return "\n".join(formatted)
            except Exception as e:
                return f"Error traversing from node {node_id}: {str(e)}"

    def configure_memory(
        self,
        use_case_description: Optional[str] = None,
        memory_threshold_description: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        """Update memory configuration dynamically.

        Args:
            use_case_description: New use case description
            memory_threshold_description: New memory threshold description
            similarity_threshold: New similarity threshold for duplicate detection
        """
        if use_case_description:
            self.memory_config.use_case_description = use_case_description
        if memory_threshold_description:
            self.memory_config.memory_threshold_description = (
                memory_threshold_description
            )
        if similarity_threshold is not None:
            self.memory_config.similarity_threshold = similarity_threshold

        # Rebuild system prompt
        # Note: This updates the config but doesn't re-create the agent
        # The agent will use the new config on next run

    def set_memory_trigger(
        self,
        mode: Literal["phrase", "interval", "ai_determined"],
        trigger_phrase: Optional[str] = None,
        message_interval: Optional[int] = None,
    ) -> None:
        """Update memory trigger configuration dynamically.

        Args:
            mode: Trigger mode ('phrase', 'interval', 'ai_determined')
            trigger_phrase: Phrase to trigger memory check (for 'phrase' mode)
            message_interval: Number of messages between checks (for 'interval' mode)
        """
        self.trigger_config = MemoryTriggerConfig(
            mode=mode,
            trigger_phrase=trigger_phrase,
            message_interval=message_interval,
        )

    def confirm_memory_addition(
        self,
        proposal_id: str,
        action: str = "add_new",
        update_node_id: Optional[str] = None,
    ) -> str:
        """Confirm and execute a pending memory proposal.

        Args:
            proposal_id: ID of the pending proposal
            action: Action to take ('add_new', 'update_existing', 'cancel')
            update_node_id: Node ID to update (required if action='update_existing')

        Returns:
            Status message
        """
        if proposal_id not in self.pending_proposals:
            return f"No pending proposal found with ID: {proposal_id}"

        proposal = self.pending_proposals[proposal_id]

        if action == "cancel":
            del self.pending_proposals[proposal_id]
            return "Memory addition cancelled."

        try:
            if action == "add_new":
                # Create new node
                node_metadata = NodeMetadata(
                    content=proposal["content"],
                    node_type=proposal["entity_type"],
                    source=self._current_session_id or "unknown",
                    project_id=self.memory_config.project_id,
                    embedding_model=str(self.store.embedding_model),
                )

                node_id = self.store.add_node(node_metadata)

                # Add relationships if specified
                edge_ids = []
                for rel in proposal["relationships"]:
                    edge_metadata = EdgeMetadata(
                        from_node_id=node_id,
                        to_node_id=rel["to_node_id"],
                        relationship_type=rel["type"],
                        description=rel.get("description", ""),
                        source=self._current_session_id or "unknown",
                        project_id=self.memory_config.project_id,
                    )
                    edge_id = self.store.add_edge(edge_metadata)
                    edge_ids.append(edge_id)

                # Log to audit
                audit_entry = AuditEntry(
                    session_id=self._current_session_id or "unknown",
                    project_id=self.memory_config.project_id,
                    operation_type="add_node",
                    summary=f"Added {proposal['entity_type']}: {proposal['content'][:100]}",
                    commands=[
                        f"add_node(node_id={node_id})",
                        *[f"add_edge(edge_id={eid})" for eid in edge_ids],
                    ],
                    metadata={
                        "node_type": proposal["entity_type"],
                        "relationship_count": len(edge_ids),
                    },
                    affected_entities=[node_id, *edge_ids],
                )
                self.audit.log_operation(audit_entry)

                del self.pending_proposals[proposal_id]
                return f"Successfully added {proposal['entity_type']} to memory with ID: {node_id}"

            elif action == "update_existing":
                if not update_node_id:
                    return "Error: update_node_id required for update_existing action"

                # Update existing node
                self.store.update_node(
                    node_id=update_node_id,
                    content=proposal["content"],
                )

                # Log to audit
                audit_entry = AuditEntry(
                    session_id=self._current_session_id or "unknown",
                    project_id=self.memory_config.project_id,
                    operation_type="update_node",
                    summary=f"Updated {proposal['entity_type']}: {proposal['content'][:100]}",
                    commands=[f"update_node(node_id={update_node_id})"],
                    metadata={"node_type": proposal["entity_type"]},
                    affected_entities=[update_node_id],
                )
                self.audit.log_operation(audit_entry)

                del self.pending_proposals[proposal_id]
                return f"Successfully updated existing node: {update_node_id}"

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            return f"Error executing memory operation: {str(e)}\n\nFull traceback:\n{error_details}"

    def run(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Any:
        """Run the agent with a user prompt.

        Args:
            prompt: User input
            session_id: Optional session ID (auto-generated if not provided)

        Returns:
            Agent response
        """
        # Set/update session ID
        if session_id:
            if session_id != self._current_session_id:
                # New session, reset message counter
                self._current_session_id = session_id
                self._message_count = 0
        else:
            if not self._current_session_id:
                self._current_session_id = str(uuid4())

        # Increment message count
        self._message_count += 1

        # Check if we should trigger memory review
        should_check_memory = self._should_check_memory(prompt)

        # Optionally inject memory check instruction
        if should_check_memory:
            enhanced_prompt = f"{prompt}\n\n[Assistant instruction: Review this conversation for items worth adding to memory based on the configured memory threshold. Use the memory tools to propose additions if appropriate.]"
        else:
            enhanced_prompt = prompt

        # Create dependencies
        deps = AgentDependencies(memory_agent=self)

        # Run agent (synchronous for v1)
        result = _run_async(self.agent.run(enhanced_prompt, deps=deps))

        return result

    def _should_check_memory(self, prompt: str) -> bool:
        """Determine if we should check for memory items.

        Args:
            prompt: User prompt

        Returns:
            True if memory check should be triggered
        """
        if self.trigger_config.mode == "phrase":
            if self.trigger_config.trigger_phrase:
                return self.trigger_config.trigger_phrase.lower() in prompt.lower()
            return False

        elif self.trigger_config.mode == "interval":
            if self.trigger_config.message_interval:
                return self._message_count % self.trigger_config.message_interval == 0
            return False

        elif self.trigger_config.mode == "ai_determined":
            # Always let AI decide
            return True

        return False

    def get_audit_history(
        self,
        session_id: Optional[str] = None,
        entity_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[AuditEntry]:
        """Get audit log history.

        Args:
            session_id: Filter by session ID
            entity_id: Filter by entity ID
            limit: Maximum number of entries

        Returns:
            List of audit entries
        """
        if session_id:
            return self.audit.get_by_session(session_id)
        elif entity_id:
            return self.audit.get_entity_history(entity_id)
        else:
            return self.audit.get_recent(limit)
