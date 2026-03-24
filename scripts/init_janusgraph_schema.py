#!/usr/bin/env python3
"""Initialize JanusGraph schema.

Run this script once before using the library to create the required
vertex labels, edge labels, and property keys in JanusGraph.

Usage:
    python scripts/init_janusgraph_schema.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gremlin_python.driver import client as gremlin_client  # type: ignore[import-untyped]


def init_schema():
    """Initialize JanusGraph schema."""
    # Connect to JanusGraph
    janusgraph_host = os.getenv("JANUSGRAPH_HOST", "localhost")
    janusgraph_port = int(os.getenv("JANUSGRAPH_PORT", "8182"))

    print(f"Connecting to JanusGraph at {janusgraph_host}:{janusgraph_port}...")
    janus = gremlin_client.Client(
        f"ws://{janusgraph_host}:{janusgraph_port}/gremlin", "g"
    )

    schema_script = """
mgmt = graph.openManagement()
if (mgmt.getVertexLabel('node') == null) {
    mgmt.makeVertexLabel('node').make()
}
if (mgmt.getEdgeLabel('relationship') == null) {
    mgmt.makeEdgeLabel('relationship').make()
}
if (mgmt.getPropertyKey('node_id') == null) {
    mgmt.makePropertyKey('node_id').dataType(String.class).make()
}
if (mgmt.getPropertyKey('node_type') == null) {
    mgmt.makePropertyKey('node_type').dataType(String.class).make()
}
if (mgmt.getPropertyKey('project_id') == null) {
    mgmt.makePropertyKey('project_id').dataType(String.class).make()
}
if (mgmt.getPropertyKey('created_at') == null) {
    mgmt.makePropertyKey('created_at').dataType(String.class).make()
}
if (mgmt.getPropertyKey('updated_at') == null) {
    mgmt.makePropertyKey('updated_at').dataType(String.class).make()
}
if (mgmt.getPropertyKey('edge_id') == null) {
    mgmt.makePropertyKey('edge_id').dataType(String.class).make()
}
if (mgmt.getPropertyKey('relationship_type') == null) {
    mgmt.makePropertyKey('relationship_type').dataType(String.class).make()
}
if (mgmt.getPropertyKey('description') == null) {
    mgmt.makePropertyKey('description').dataType(String.class).make()
}
if (mgmt.getPropertyKey('source') == null) {
    mgmt.makePropertyKey('source').dataType(String.class).make()
}
if (mgmt.getPropertyKey('confidence') == null) {
    mgmt.makePropertyKey('confidence').dataType(Double.class).make()
}
mgmt.commit()
'Schema initialized successfully'
"""

    print("Creating JanusGraph schema...")
    try:
        result = janus.submit(schema_script).all().result()
        print(f"✓ {result[0]}")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    finally:
        janus.close()


if __name__ == "__main__":
    init_schema()
