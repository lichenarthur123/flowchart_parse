"""Build a node-and-directed-edge JSON from candidate paths and arrow evidence."""
from __future__ import annotations

import math


def infer_unknown_diamond_types(graph: dict, unresolved: dict, box_audit: list[dict]) -> dict:
    """Infer a diamond's role only when recognized connections are unambiguous."""
    diamonds = {item['id'] for item in box_audit if item.get('detected_shape') == 'diamond'}
    ambiguous = {node_id for edge in unresolved['edges'] for node_id in edge['node_ids']}
    nodes = []
    for original in graph['nodes']:
        node = dict(original)
        node_id = node['id']
        if node.get('type') == 'unknown' and node_id in diamonds and node_id not in ambiguous:
            incoming = {edge['from'] for edge in graph['edges'] if edge['to'] == node_id}
            outgoing = {edge['to'] for edge in graph['edges'] if edge['from'] == node_id}
            inferred = 'merge' if len(incoming) >= 2 and len(outgoing) == 1 else (
                'decision' if len(incoming) == 1 and len(outgoing) >= 2 else None)
            if inferred:
                node.update(type=inferred, type_inference={
                    'method': 'diamond_contour_and_topology', 'original_type': 'unknown',
                    'incoming_nodes': len(incoming), 'outgoing_nodes': len(outgoing)})
        nodes.append(node)
    return {**graph, 'nodes': nodes}


def build_graph(entity_data: dict, candidates: dict) -> tuple[dict, dict, dict]:
    """Preserve entity JSON and add edges; ambiguous candidates remain separate.

    Arrow shape and connection screening are heuristics; the result must be
    reviewed. The order of node_ids is not a direction and is never used as one.
    """
    if not isinstance(entity_data.get('nodes'), list):
        raise ValueError('实体 JSON 必须包含 nodes 列表。')
    if 'edges' in entity_data and entity_data['edges'] != []:
        raise ValueError('输入实体 JSON 已有 edges，拒绝覆盖。')
    node_ids = {n['id'] for n in entity_data['nodes']}
    if len(node_ids) != len(entity_data['nodes']):
        raise ValueError('实体 JSON 包含重复节点 ID。')
    points = {p['id']: p for p in candidates['connection_points']}
    edges, unresolved, evidence = [], [], []
    for trace in candidates['traces']:
        entry = {'candidate_id': trace['id'], 'node_ids': trace['node_ids'],
                 'path_global_px': trace['path_global_px'],
                 'via_gap': trace.get('via_gap', False),
                 'via_junctions_global_px': trace.get('via_junctions_global_px', []),
                 'arrow_evidence': trace.get('arrow_evidence', []),
                 'review_flags': trace.get('review_flags', [])}
        if entry['review_flags']:
            unresolved.append({**entry, 'reason': 'candidate_flagged'})
            continue
        ends = [points[trace[key]] for key in ('start_point_id', 'end_point_id')]
        if any(len(p['possible_nodes']) != 1 for p in ends):
            unresolved.append({**entry, 'reason': 'ambiguous_node'})
            continue
        endpoint_nodes = [p['possible_nodes'][0]['id'] for p in ends]
        arrows = entry['arrow_evidence']
        if (len(arrows) != 2 or any(type(a.get('possible_arrow_toward_node')) is not bool for a in arrows)):
            unresolved.append({**entry, 'reason': 'missing_arrow_evidence'})
            continue
        indicators = [a['possible_arrow_toward_node'] for a in arrows]
        if sum(indicators) != 1:
            unresolved.append({**entry, 'reason': 'ambiguous_direction'})
            continue
        if any(node not in node_ids for node in endpoint_nodes) or endpoint_nodes[0] == endpoint_nodes[1]:
            unresolved.append({**entry, 'reason': 'invalid_node'})
            continue
        path = entry['path_global_px']
        if (not isinstance(path, list) or len(path) < 2 or
            any(not isinstance(p, list) or len(p) != 2 or
                any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in p)
                for p in path) or
            path[0] != ends[0].get('point_global_px') or path[-1] != ends[1].get('point_global_px')):
            unresolved.append({**entry, 'reason': 'invalid_path_geometry'})
            continue
        target = indicators.index(True)
        edge_id = f"edge_{trace['id'].removeprefix('candidate_')}"
        # The trace follows start_point → end_point, which may oppose from → to.
        directed_path = path if target == 1 else list(reversed(path))
        edges.append({'id': edge_id, 'from': endpoint_nodes[1-target], 'to': endpoint_nodes[target],
                      'path_global_px': directed_path})
        evidence.append({'edge_id': edge_id, **entry, 'path_global_px': directed_path})
    graph = {**entity_data, 'edges': edges}
    return graph, {'coordinate_system': candidates['coordinate_system'], 'edges': unresolved}, {
        'coordinate_system': candidates['coordinate_system'],
        'note': '边方向来自本图像素宽度启发式，分叉与断口可能有误；所有边均需检查。',
        'edges': evidence,
    }
