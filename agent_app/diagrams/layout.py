"""Deterministic layered layout (Sugiyama-style, simplified), applied twice: groups/ungrouped nodes first, then inside each group.

Groups never overlap because each group is laid out as one box in the outer pass.
"""
from __future__ import annotations

from collections import defaultdict

NODE_W, NODE_H = 150, 64
GAP_MAIN, GAP_CROSS = 90, 36
GROUP_PAD, GROUP_HEAD = 24, 34

Size = tuple[float, float]


def _layers(ids: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    succ: dict[str, list[str]] = defaultdict(list)
    for a, b in dict.fromkeys(edges):
        if a != b:
            succ[a].append(b)
    # drop back edges (cycles) found by DFS so that layering is well defined
    state: dict[str, int] = {}
    fwd: dict[str, list[str]] = defaultdict(list)
    for root in ids:
        if root in state:
            continue
        stack = [(root, iter(succ[root]))]
        state[root] = 1
        while stack:
            node, it = stack[-1]
            for nxt in it:
                if state.get(nxt) == 1:
                    continue  # back edge
                fwd[node].append(nxt)
                if nxt not in state:
                    state[nxt] = 1
                    stack.append((nxt, iter(succ[nxt])))
                    break
            else:
                state[node] = 2
                stack.pop()
    depth = {i: 0 for i in ids}
    indeg = {i: 0 for i in ids}
    for a in ids:
        for b in fwd[a]:
            indeg[b] += 1
    queue = [i for i in ids if indeg[i] == 0]
    while queue:
        a = queue.pop(0)
        for b in fwd[a]:
            depth[b] = max(depth[b], depth[a] + 1)
            indeg[b] -= 1
            if indeg[b] == 0:
                queue.append(b)
    layers: list[list[str]] = [[] for _ in range(max(depth.values(), default=0) + 1)]
    for i in ids:
        layers[depth[i]].append(i)
    return layers


def _order(layers: list[list[str]], edges: list[tuple[str, str]]) -> None:
    """Reduce crossings: a few barycenter sweeps (in place)."""
    nbrs: dict[str, list[str]] = defaultdict(list)
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    for sweep in range(4):
        seq = range(1, len(layers)) if sweep % 2 == 0 else range(len(layers) - 2, -1, -1)
        for li in seq:
            ref = layers[li - 1] if sweep % 2 == 0 else layers[li + 1]
            pos = {n: k for k, n in enumerate(ref)}
            cur = {n: k for k, n in enumerate(layers[li])}

            def bary(n: str) -> float:
                ps = [pos[m] for m in nbrs[n] if m in pos]
                return sum(ps) / len(ps) if ps else cur[n]
            layers[li].sort(key=lambda n: (bary(n), cur[n]))


def place(items: dict[str, Size], edges: list[tuple[str, str]], direction: str) -> tuple[dict[str, tuple[float, float]], Size]:
    """Positions (top-left, relative to the area's origin) for each item and the total size."""
    if not items:
        return {}, (0, 0)
    ids = list(items)
    edges = [(a, b) for a, b in edges if a in items and b in items]
    layers = _layers(ids, edges)
    _order(layers, edges)
    lr = direction == "LR"
    main = (lambda s: s[0]) if lr else (lambda s: s[1])
    cross = (lambda s: s[1]) if lr else (lambda s: s[0])
    spans = [sum(cross(items[n]) for n in L) + GAP_CROSS * (len(L) - 1) for L in layers]
    total_cross = max(spans)
    pos: dict[str, tuple[float, float]] = {}
    m = 0.0
    for L, span in zip(layers, spans):
        c = (total_cross - span) / 2  # centre each layer on the widest one
        width = max(main(items[n]) for n in L)
        for n in L:
            mp = m + (width - main(items[n])) / 2
            pos[n] = (mp, c) if lr else (c, mp)
            c += cross(items[n]) + GAP_CROSS
        m += width + GAP_MAIN
    total_main = m - GAP_MAIN
    return pos, ((total_main, total_cross) if lr else (total_cross, total_main))


def layout(node_ids: list[str], node_group: dict[str, str | None], group_ids: list[str],
           edges: list[tuple[str, str]], direction: str) -> tuple[dict[str, tuple[float, float, float, float]],
                                                                  dict[str, tuple[float, float, float, float]], Size]:
    """Returns (node boxes, group boxes, total size). Node boxes inside a group are relative to that group; others absolute."""
    members: dict[str, list[str]] = {g: [] for g in group_ids}
    for n in node_ids:
        g = node_group.get(n)
        if g in members:
            members[g].append(n)
    inner: dict[str, tuple[dict[str, tuple[float, float]], Size]] = {}
    for g, ms in members.items():
        if ms:
            inner[g] = place({n: (NODE_W, NODE_H) for n in ms}, [e for e in edges if e[0] in ms and e[1] in ms], direction)
    top_of = {n: (node_group[n] if node_group.get(n) in members else n) for n in node_ids}
    items: dict[str, Size] = {}
    for n in node_ids:
        if top_of[n] == n:
            items[n] = (NODE_W, NODE_H)
    for g in group_ids:
        if g in inner:
            w, h = inner[g][1]
            items[g] = (w + 2 * GROUP_PAD, h + GROUP_HEAD + GROUP_PAD)
        else:
            items[g] = (NODE_W + 2 * GROUP_PAD, GROUP_HEAD + GROUP_PAD + 20)  # empty group: still drawn
    outer_edges = [(top_of[a], top_of[b]) for a, b in edges if top_of[a] != top_of[b]]
    pos, size = place(items, outer_edges, direction)
    nodes: dict[str, tuple[float, float, float, float]] = {}
    groups: dict[str, tuple[float, float, float, float]] = {}
    for n in node_ids:
        if top_of[n] == n:
            x, y = pos[n]
            nodes[n] = (x, y, NODE_W, NODE_H)
    for g in group_ids:
        x, y = pos[g]
        groups[g] = (x, y, *items[g])
        if g in inner:
            for n, (nx, ny) in inner[g][0].items():
                nodes[n] = (GROUP_PAD + nx, GROUP_HEAD + ny, NODE_W, NODE_H)
    return nodes, groups, size
