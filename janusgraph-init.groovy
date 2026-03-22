// JanusGraph schema initialization script
// This runs automatically when JanusGraph starts

mgmt = graph.openManagement()

// Create vertex label 'node' if it doesn't exist
if (mgmt.getVertexLabel('node') == null) {
    mgmt.makeVertexLabel('node').make()
}

// Create edge label 'relationship' if it doesn't exist
if (mgmt.getEdgeLabel('relationship') == null) {
    mgmt.makeEdgeLabel('relationship').make()
}

// Create property keys
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
