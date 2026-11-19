"""dsa_kit: drawing, timing and self-checking helpers for the DSA module (Weeks 11–12).

Every lesson notebook starts with:

    from dsa_kit import *

You don't need to understand this file to do the lessons. It exists so the notebooks
can focus on the ideas. Once you're comfortable, though, read it: it's a good example
of real-world Python built from the data structures you're learning (dicts, lists,
deques, recursion on trees, and so on).
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

try:  # rich display inside Jupyter; plain printing elsewhere
    from IPython.display import HTML, Markdown, display
except ImportError:  # pragma: no cover
    HTML = Markdown = None

    def display(x):
        print(x)

__all__ = [
    "PALETTE", "draw_array", "draw_steps", "draw_linked_list", "draw_stack", "draw_queue",
    "draw_hash_table", "draw_tree", "draw_graph", "draw_grid", "draw_dp_table",
    "plot_growth", "time_it", "benchmark", "plot_benchmark", "animate_bars",
    "check", "random_list", "TreeNode", "build_tree", "callout",
]

# ---------------------------------------------------------------------------
# Visual language (the same colours mean the same thing in every lesson)
# ---------------------------------------------------------------------------
PALETTE = {
    "base": "#E8EEF7",      # an ordinary cell / node
    "edge": "#3B4A63",      # outlines, arrows, text
    "active": "#F5A623",    # the element being looked at right now
    "done": "#4CAF7A",      # finished / sorted / visited
    "special": "#E5533D",   # found, swapped, error, the "answer"
    "frontier": "#6FA8DC",  # waiting in a queue / on the frontier
    "muted": "#B8C2D1",     # out of range / discarded
    "bg": "#FFFFFF",
}
_COLOR_ALIASES = {k: v for k, v in PALETTE.items()}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor": PALETTE["bg"],
    "savefig.facecolor": PALETTE["bg"],
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
})


def _c(color: str | None, default: str = "base") -> str:
    if color is None:
        return PALETTE[default]
    return _COLOR_ALIASES.get(color, color)


def _blank_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")


def callout(text: str, kind: str = "tip"):
    """Show a coloured note box in a notebook (kind: tip | warning | key)."""
    colors = {"tip": "#E8F4FD", "warning": "#FDECEA", "key": "#EAF7EF"}
    icons = {"tip": "💡", "warning": "⚠️", "key": "🔑"}
    html = (f"<div style='background:{colors.get(kind, '#F3F3F3')};color:#1d2433;"
            f"padding:10px 14px;border-radius:8px;margin:6px 0'>"
            f"{icons.get(kind, '')} {text}</div>")
    if HTML:
        display(HTML(html))
    else:
        print(text)


# ---------------------------------------------------------------------------
# Arrays and step-by-step "comic strips"
# ---------------------------------------------------------------------------
def _draw_array_on(ax, values, highlight=None, pointers=None, title=None, show_index=True,
                   cell=1.0, label_size=13):
    highlight = highlight or {}
    pointers = pointers or {}
    n = len(values)
    for i, v in enumerate(values):
        color = _c(highlight.get(i))
        ax.add_patch(FancyBboxPatch((i * cell + 0.04, 0.04), cell - 0.08, cell - 0.08,
                                    boxstyle="round,pad=0.0,rounding_size=0.12",
                                    fc=color, ec=PALETTE["edge"], lw=1.4))
        ax.text(i * cell + cell / 2, cell / 2, str(v), ha="center", va="center",
                fontsize=label_size, color="#1d2433", fontweight="bold")
        if show_index:
            ax.text(i * cell + cell / 2, -0.22, str(i), ha="center", va="center",
                    fontsize=9, color="#6b7a90")
    # several pointers may share an index: stack their labels
    by_index: dict[int, list[str]] = {}
    for name, idx in pointers.items():
        if idx is not None and -1 <= idx <= n:
            by_index.setdefault(idx, []).append(name)
    for idx, names in by_index.items():
        x = idx * cell + cell / 2
        ax.annotate("", xy=(x, cell + 0.05), xytext=(x, cell + 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE["special"], lw=1.6))
        ax.text(x, cell + 0.75, " / ".join(names), ha="center", va="bottom", fontsize=10,
                color=PALETTE["special"], fontweight="bold")
    ax.set_xlim(-0.3, max(n, 1) * cell + 0.3)
    ax.set_ylim(-0.5, cell + 1.25)
    _blank_axes(ax)
    if title:
        ax.set_title(title, fontsize=11)


def draw_array(values: Sequence, highlight: dict | None = None, pointers: dict | None = None,
               title: str | None = None, show_index: bool = True, ax=None):
    """Draw a list as a row of boxes.

    highlight: {index: colour or palette name}, e.g. {2: "active", 5: "done"}
    pointers:  {label: index}, e.g. {"lo": 0, "hi": 7, "mid": 3}
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(max(4, 0.75 * len(values) + 1), 1.9))
    _draw_array_on(ax, list(values), highlight, pointers, title, show_index)
    if own:
        plt.tight_layout()
        plt.show()


def draw_steps(states: list[dict], ncols: int = 1, title: str | None = None,
               max_steps: int = 12):
    """Small multiples: one array picture per step.

    Each state is a dict with keys values, plus optional highlight, pointers, caption.
    """
    states = states[:max_steps]
    n = len(states)
    nrows = math.ceil(n / ncols)
    width = max(len(s["values"]) for s in states)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * max(4, 0.62 * width + 1), nrows * 1.75),
                             squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k < n:
            s = states[k]
            _draw_array_on(ax, s["values"], s.get("highlight"), s.get("pointers"),
                           s.get("caption", f"step {k + 1}"), show_index=s.get("show_index", True),
                           label_size=11)
        else:
            ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Linked lists, stacks, queues, hash tables
# ---------------------------------------------------------------------------
def draw_linked_list(values: Sequence, highlight: dict | None = None, doubly: bool = False,
                     head_label: str = "head", title: str | None = None, extra_pointers: dict | None = None):
    """Draw a singly (or doubly) linked list: [value|next] -> [value|next] -> None."""
    highlight = highlight or {}
    extra_pointers = extra_pointers or {}
    n = len(values)
    fig, ax = plt.subplots(figsize=(max(4.5, 2.0 * n + 1.8), 2.0))
    w, h, gap = 1.3, 0.8, 0.7
    for i, v in enumerate(values):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=_c(highlight.get(i)), ec=PALETTE["edge"], lw=1.4))
        ax.plot([x + w * 0.68, x + w * 0.68], [0, h], color=PALETTE["edge"], lw=1)
        ax.text(x + w * 0.34, h / 2, str(v), ha="center", va="center", fontsize=12, fontweight="bold")
        ax.plot(x + w * 0.84, h / 2, "o", color=PALETTE["edge"], ms=4)
        nxt_x = (i + 1) * (w + gap)
        ax.add_patch(FancyArrowPatch((x + w * 0.84, h * 0.62 if doubly else h / 2),
                                     (nxt_x - 0.02, h * 0.62 if doubly else h / 2),
                                     arrowstyle="-|>", mutation_scale=14, color=PALETTE["edge"], lw=1.3))
        if doubly and i > 0:
            ax.add_patch(FancyArrowPatch((x + 0.02, h * 0.3), (x - gap + 0.02, h * 0.3),
                                         arrowstyle="-|>", mutation_scale=12, color=PALETTE["frontier"], lw=1.2))
    end_x = n * (w + gap)
    ax.text(end_x + 0.25, h / 2, "None", va="center", fontsize=11, color="#6b7a90", style="italic")
    labels = {head_label: 0, **extra_pointers}
    by_index: dict[int, list[str]] = {}
    for name, idx in labels.items():
        if idx is not None and 0 <= idx < max(n, 1):
            by_index.setdefault(idx, []).append(name)
    for idx, names in by_index.items():
        x = idx * (w + gap) + w * 0.34
        ax.annotate("", xy=(x, h + 0.05), xytext=(x, h + 0.55),
                    arrowprops=dict(arrowstyle="-|>", color=PALETTE["special"], lw=1.5))
        ax.text(x, h + 0.62, " / ".join(names), ha="center", fontsize=10, color=PALETTE["special"], fontweight="bold")
    ax.set_xlim(-0.3, end_x + 1.2)
    ax.set_ylim(-0.4, h + 1.1)
    _blank_axes(ax)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def draw_stack(items: Sequence, title: str = "Stack (top at the top)", highlight_top: bool = True):
    """Draw a vertical stack. The last item in the list is the top."""
    items = list(items)
    fig, ax = plt.subplots(figsize=(2.8, 0.55 * max(len(items), 1) + 1.4))
    for i, v in enumerate(items):
        is_top = highlight_top and i == len(items) - 1
        ax.add_patch(FancyBboxPatch((0, i * 0.6), 2, 0.52, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=_c("active" if is_top else None), ec=PALETTE["edge"], lw=1.3))
        ax.text(1, i * 0.6 + 0.26, str(v), ha="center", va="center", fontsize=11, fontweight="bold")
    if items:
        ax.text(2.15, (len(items) - 1) * 0.6 + 0.26, "◀ top", va="center", color=PALETTE["special"], fontweight="bold")
    ax.plot([-0.1, -0.1, 2.1, 2.1], [len(items) * 0.6 + 0.2, -0.08, -0.08, len(items) * 0.6 + 0.2],
            color=PALETTE["edge"], lw=2)
    ax.set_xlim(-0.4, 3.2)
    ax.set_ylim(-0.4, len(items) * 0.6 + 0.6)
    _blank_axes(ax)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def draw_queue(items: Sequence, title: str = "Queue (front on the left)"):
    """Draw a queue: items leave from the front (left) and join at the back (right)."""
    items = list(items)
    hl = {0: "active"} if items else {}
    if items:
        hl[len(items) - 1] = hl.get(len(items) - 1, "frontier")
    fig, ax = plt.subplots(figsize=(max(4, 0.8 * len(items) + 2.5), 1.7))
    _draw_array_on(ax, items, hl, {"front": 0, "back": len(items) - 1} if items else {}, title, show_index=False)
    ax.annotate("", xy=(-0.75, 0.5), xytext=(-0.02, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["edge"], lw=1.5))
    ax.text(-0.85, 0.5, "dequeue", ha="right", va="center", fontsize=10)
    ax.annotate("", xy=(len(items) + 0.02, 0.5), xytext=(len(items) + 0.75, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=PALETTE["edge"], lw=1.5))
    ax.text(len(items) + 0.85, 0.5, "enqueue", ha="left", va="center", fontsize=10)
    ax.set_xlim(-2.6, len(items) + 2.6)
    plt.tight_layout()
    plt.show()


def draw_hash_table(buckets: Sequence[Sequence], title: str = "Hash table (separate chaining)",
                    highlight_bucket: int | None = None):
    """Draw a hash table as an array of buckets, each with a chain of entries."""
    m = len(buckets)
    longest = max((len(b) for b in buckets), default=0)
    fig, ax = plt.subplots(figsize=(2.6 + 1.9 * max(longest, 1), 0.62 * m + 1.0))
    for i, chain in enumerate(buckets):
        y = (m - 1 - i) * 0.62
        ax.add_patch(FancyBboxPatch((0, y), 0.8, 0.5, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=_c("active" if i == highlight_bucket else None), ec=PALETTE["edge"]))
        ax.text(0.4, y + 0.25, str(i), ha="center", va="center", fontsize=10, fontweight="bold")
        prev_x = 0.8
        for j, item in enumerate(chain):
            x = 1.3 + j * 1.9
            ax.add_patch(FancyArrowPatch((prev_x, y + 0.25), (x, y + 0.25), arrowstyle="-|>",
                                         mutation_scale=11, color=PALETTE["edge"]))
            ax.add_patch(FancyBboxPatch((x, y + 0.02), 1.5, 0.46, boxstyle="round,pad=0,rounding_size=0.06",
                                        fc=PALETTE["done"] if i == highlight_bucket else "#F7F9FC", ec=PALETTE["edge"]))
            ax.text(x + 0.75, y + 0.25, str(item), ha="center", va="center", fontsize=9)
            prev_x = x + 1.5
        if not chain:
            ax.text(1.0, y + 0.25, "empty", va="center", fontsize=9, color="#9aa6b8", style="italic")
    ax.set_xlim(-0.2, 1.6 + 1.9 * max(longest, 1))
    ax.set_ylim(-0.2, m * 0.62)
    _blank_axes(ax)
    ax.set_aspect("auto")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------
class TreeNode:
    """A binary tree node: value plus left/right children."""

    def __init__(self, val, left: "TreeNode | None" = None, right: "TreeNode | None" = None):
        self.val, self.left, self.right = val, left, right

    def __repr__(self):
        return f"TreeNode({self.val!r})"


def build_tree(level_order: Sequence) -> TreeNode | None:
    """Build a binary tree from a level-order list, using None for gaps.

    build_tree([8, 3, 10, 1, 6, None, 14]) is the classic BST picture.
    """
    if not level_order or level_order[0] is None:
        return None
    it = iter(level_order)
    root = TreeNode(next(it))
    q = deque([root])
    for left_val in it:
        node = q.popleft()
        if left_val is not None:
            node.left = TreeNode(left_val)
            q.append(node.left)
        right_val = next(it, None)
        if right_val is not None:
            node.right = TreeNode(right_val)
            q.append(node.right)
    return root


def _children(node, left_attr="left", right_attr="right", children_attr=None):
    if children_attr:
        return list(getattr(node, children_attr) or [])
    return [getattr(node, left_attr, None), getattr(node, right_attr, None)]


def draw_tree(root, highlight: dict | None = None, title: str | None = None, val_attr: str = "val",
              left_attr: str = "left", right_attr: str = "right", children_attr: str | None = None,
              node_size: float = 0.32, figsize: tuple | None = None):
    """Draw a tree. Works with TreeNode, your own node class (set the *_attr names), or
    n-ary nodes (set children_attr). highlight maps node value -> colour/palette name.
    """
    highlight = highlight or {}
    if root is None:
        print("(empty tree)")
        return
    pos: dict[int, tuple[float, float]] = {}
    nodes: dict[int, Any] = {}
    edges: list[tuple[int, int, str]] = []
    counter = [0]

    def layout(node, depth):
        """In-order x positions give a tidy, overlap-free drawing."""
        kids = _children(node, left_attr, right_attr, children_attr)
        if children_attr:
            xs = []
            for k in kids:
                xs.append(layout(k, depth + 1))
                edges.append((id(node), id(k), ""))
            x = sum(xs) / len(xs) if xs else counter[0]
            if not xs:
                counter[0] += 1
        else:
            left, right = kids
            if left is not None:
                layout(left, depth + 1)
                edges.append((id(node), id(left), "L"))
            x = counter[0]
            counter[0] += 1
            if right is not None:
                layout(right, depth + 1)
                edges.append((id(node), id(right), "R"))
        pos[id(node)] = (x, -depth)
        nodes[id(node)] = node
        return x

    layout(root, 0)
    xs = [p[0] for p in pos.values()]
    depth = -min(p[1] for p in pos.values())
    if figsize is None:
        figsize = (max(4, 0.75 * (max(xs) - min(xs) + 2)), 1.25 * (depth + 1) + 0.6)
    fig, ax = plt.subplots(figsize=figsize)
    for a, b, _ in edges:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        ax.plot([x1, x2], [y1, y2], color=PALETTE["edge"], lw=1.4, zorder=1)
    for nid, (x, y) in pos.items():
        val = getattr(nodes[nid], val_attr)
        ax.add_patch(Circle((x, y), node_size, fc=_c(highlight.get(val)), ec=PALETTE["edge"], lw=1.5, zorder=2))
        ax.text(x, y, str(val), ha="center", va="center", fontsize=11, fontweight="bold", zorder=3)
    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(-depth - 0.7, 0.7)
    _blank_axes(ax)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------
def _circle_layout(nodes):
    n = len(nodes)
    return {v: (math.cos(2 * math.pi * i / n + math.pi / 2), math.sin(2 * math.pi * i / n + math.pi / 2))
            for i, v in enumerate(nodes)}


def draw_graph(edges: Iterable, directed: bool = False, pos: dict | None = None,
               node_colors: dict | None = None, edge_colors: dict | None = None,
               weights: bool | None = None, title: str | None = None, nodes: Iterable | None = None,
               node_labels: dict | None = None, figsize=(6, 4.6), ax=None):
    """Draw a graph from an edge list.

    edges:  [(u, v)] or [(u, v, weight)]
    pos:    {node: (x, y)}. Defaults to a circle layout.
    node_colors / edge_colors: {node: colour} / {(u, v): colour}, using palette names or hex
    node_labels: {node: extra text shown under the node}, e.g. distances
    """
    edges = list(edges)
    has_w = any(len(e) == 3 for e in edges)
    weights = has_w if weights is None else weights
    all_nodes = list(dict.fromkeys(list(nodes or []) + [e[0] for e in edges] + [e[1] for e in edges]))
    pos = pos or _circle_layout(all_nodes)
    node_colors = node_colors or {}
    edge_colors = edge_colors or {}
    node_labels = node_labels or {}
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=figsize)
    r = 0.13 if max(abs(c) for p in pos.values() for c in p) <= 1.5 else 0.3
    for e in edges:
        u, v = e[0], e[1]
        (x1, y1), (x2, y2) = pos[u], pos[v]
        col = _c(edge_colors.get((u, v), edge_colors.get((v, u)) if not directed else edge_colors.get((u, v))),
                 "edge")
        lw = 2.8 if (u, v) in edge_colors or (not directed and (v, u) in edge_colors) else 1.4
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1
        sx, sy = x1 + dx / d * r, y1 + dy / d * r
        ex, ey = x2 - dx / d * r, y2 - dy / d * r
        if directed:
            ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>", mutation_scale=16,
                                         color=col, lw=lw, connectionstyle="arc3,rad=0.08", zorder=1))
        else:
            ax.plot([sx, ex], [sy, ey], color=col, lw=lw, zorder=1)
        if weights and len(e) == 3:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, str(e[2]), fontsize=10, ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    for v in all_nodes:
        x, y = pos[v]
        ax.add_patch(Circle((x, y), r, fc=_c(node_colors.get(v)), ec=PALETTE["edge"], lw=1.5, zorder=2))
        ax.text(x, y, str(v), ha="center", va="center", fontsize=11, fontweight="bold", zorder=3)
        if v in node_labels:
            ax.text(x, y - r - 0.08 * (r / 0.13), str(node_labels[v]), ha="center", va="top", fontsize=9,
                    color=PALETTE["special"], fontweight="bold", zorder=3)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    pad = r * 2.2
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad * 1.4, max(ys) + pad)
    _blank_axes(ax)
    if title:
        ax.set_title(title)
    if own:
        plt.tight_layout()
        plt.show()


def draw_grid(grid: Sequence[Sequence], path: Iterable | None = None, visited: Iterable | None = None,
              start=None, goal=None, title: str | None = None, wall="#"):
    """Draw a 2-D maze/grid. Cells equal to `wall` are walls. path / visited are (row, col) cells."""
    rows, cols = len(grid), len(grid[0])
    path, visited = set(path or []), set(visited or [])
    fig, ax = plt.subplots(figsize=(0.5 * cols + 1, 0.5 * rows + 0.8))
    for r_ in range(rows):
        for c_ in range(cols):
            cell = (r_, c_)
            if grid[r_][c_] == wall:
                col = PALETTE["edge"]
            elif cell == start:
                col = PALETTE["done"]
            elif cell == goal:
                col = PALETTE["special"]
            elif cell in path:
                col = PALETTE["active"]
            elif cell in visited:
                col = PALETTE["frontier"]
            else:
                col = "#F7F9FC"
            ax.add_patch(plt.Rectangle((c_, rows - 1 - r_), 1, 1, fc=col, ec="#D5DCE6", lw=0.8))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    _blank_axes(ax)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Dynamic programming tables
# ---------------------------------------------------------------------------
def draw_dp_table(table: Sequence[Sequence], row_labels: Sequence | None = None,
                  col_labels: Sequence | None = None, highlight: Iterable | None = None,
                  title: str | None = None, cmap: str = "Blues", fmt: str = "{}"):
    """Draw a DP table as a heat map with the numbers written in. highlight: [(r, c), ...]."""
    import numpy as np

    data = np.array([[float(x) if x is not None and x != math.inf else np.nan for x in row] for row in table])
    rows, cols = data.shape
    fig, ax = plt.subplots(figsize=(0.62 * cols + 1.8, 0.5 * rows + 1.2))
    finite = data[~np.isnan(data)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0, 1)
    # stretch the colour range so even the "hottest" cell stays light enough to read
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=lo, vmax=hi + (hi - lo) * 0.9 + 1e-9)
    hl = set(highlight or [])
    for r_ in range(rows):
        for c_ in range(cols):
            v = table[r_][c_]
            txt = "∞" if v == math.inf else ("" if v is None else fmt.format(v))
            ax.text(c_, r_, txt, ha="center", va="center", fontsize=10,
                    fontweight="bold" if (r_, c_) in hl else "normal",
                    color=PALETTE["special"] if (r_, c_) in hl else "#1d2433")
            if (r_, c_) in hl:
                ax.add_patch(plt.Rectangle((c_ - 0.5, r_ - 0.5), 1, 1, fill=False, ec=PALETTE["special"], lw=2.2))
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(col_labels if col_labels is not None else range(cols))
    ax.set_yticklabels(row_labels if row_labels is not None else range(rows))
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    if title:
        ax.set_title(title, pad=24)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Complexity: theory curves and real timing
# ---------------------------------------------------------------------------
_GROWTH = {
    "O(1)": lambda n: 1.0,
    "O(log n)": lambda n: math.log2(n) if n > 1 else 1.0,
    "O(n)": lambda n: float(n),
    "O(n log n)": lambda n: n * math.log2(n) if n > 1 else 1.0,
    "O(n²)": lambda n: float(n * n),
    "O(2ⁿ)": lambda n: 2.0 ** n,
}


def plot_growth(classes: Sequence[str] | None = None, n_max: int = 30, log_y: bool = False,
                title: str = "How the work grows as n grows"):
    """Plot the classic Big-O curves on one chart."""
    classes = classes or list(_GROWTH)
    ns = list(range(1, n_max + 1))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for name in classes:
        ys = [_GROWTH[name](n) for n in ns]
        ax.plot(ns, ys, lw=2.4, label=name)
    ax.set_xlabel("input size n")
    ax.set_ylabel("operations (relative)")
    if log_y:
        ax.set_yscale("log")
    else:
        top = max(_GROWTH["O(n²)"](n_max) if "O(n²)" in classes else n_max * math.log2(max(n_max, 2)), 10)
        ax.set_ylim(0, top * 1.05)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def time_it(fn: Callable, *args, repeat: int = 5, number: int = 1, **kwargs) -> float:
    """Best-of-`repeat` wall time in seconds for fn(*args, **kwargs) (best = least noisy)."""
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            fn(*args, **kwargs)
        best = min(best, (time.perf_counter() - t0) / number)
    return best


def benchmark(funcs: dict[str, Callable], make_input: Callable[[int], Any], sizes: Sequence[int],
              repeat: int = 3) -> dict[str, list[float]]:
    """Time several functions over growing input sizes.

    funcs:      {"label": fn}, where each fn takes the output of make_input(n)
    make_input: n -> input. A fresh input is built for each size and shared across funcs.
    Returns {"label": [seconds per size]}.
    """
    results = {name: [] for name in funcs}
    for n in sizes:
        data = make_input(n)
        for name, fn in funcs.items():
            results[name].append(time_it(fn, data, repeat=repeat))
    return results


def plot_benchmark(sizes: Sequence[int], results: dict[str, list[float]], title: str = "Measured running time",
                   log: bool = False, unit: str = "ms"):
    """Plot benchmark() output."""
    scale = {"s": 1, "ms": 1e3, "µs": 1e6}[unit]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, ts in results.items():
        ax.plot(sizes, [t * scale for t in ts], marker="o", lw=2.2, label=name)
    ax.set_xlabel("input size n")
    ax.set_ylabel(f"time ({unit})")
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Animation (for sorting and searching)
# ---------------------------------------------------------------------------
def animate_bars(frames: list[dict], title: str = "", interval: int = 250):
    """Animate a list of states as a bar chart and return a playable HTML widget.

    Each frame is {"values": [...], "highlight": {index: colour}, "caption": "..."}.
    Typical use: write a sorting function that appends a frame after each compare/swap.
    """
    if not frames:
        raise ValueError("no frames to animate")
    n = len(frames[0]["values"])
    top = max(max(f["values"]) for f in frames)
    fig, ax = plt.subplots(figsize=(min(10, 0.45 * n + 3), 3.6))
    bars = ax.bar(range(n), frames[0]["values"], color=PALETTE["base"], edgecolor=PALETTE["edge"])
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    caption = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=10)
    ax.set_title(title)

    def update(k):
        f = frames[k]
        hl = f.get("highlight", {})
        for i, (b, v) in enumerate(zip(bars, f["values"])):
            b.set_height(v)
            b.set_color(_c(hl.get(i)))
            b.set_edgecolor(PALETTE["edge"])
        caption.set_text(f.get("caption", f"frame {k + 1}/{len(frames)}"))
        return list(bars) + [caption]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    plt.close(fig)
    html = anim.to_jshtml(default_mode="once")
    return HTML(html) if HTML else html


# ---------------------------------------------------------------------------
# Self-checking exercises
# ---------------------------------------------------------------------------
def check(fn: Callable, cases: Sequence[tuple], name: str | None = None, compare: Callable | None = None,
          show_passing: bool = False) -> bool:
    """Run your function against test cases and report clearly.

    cases: [(args_tuple, expected), ...]. A single non-tuple arg can be passed bare.
    compare: optional (got, expected) -> bool, e.g. for order-insensitive answers.
    """
    import copy

    name = name or getattr(fn, "__name__", "function")
    compare = compare or (lambda got, exp: got == exp)
    passed = 0
    lines = []
    for args, expected in cases:
        call_args = args if isinstance(args, tuple) else (args,)
        try:
            got = fn(*copy.deepcopy(call_args))
            ok = compare(got, expected)
        except NotImplementedError:
            print(f"⏳ {name}: not implemented yet. Replace the `raise NotImplementedError` with your code.")
            return False
        except Exception as e:  # show the error rather than crashing the notebook
            got, ok = f"{type(e).__name__}: {e}", False
        if ok:
            passed += 1
            if show_passing:
                lines.append(f"  ✅ {name}{call_args!r} -> {got!r}")
        else:
            lines.append(f"  ❌ {name}{call_args!r}\n       expected {expected!r}\n       got      {got!r}")
    total = len(cases)
    status = "🎉 All tests passed!" if passed == total else f"{passed}/{total} tests passed"
    print(f"{name}: {status}")
    if lines:
        print("\n".join(lines))
    return passed == total


def random_list(n: int, low: int = 0, high: int | None = None, seed: int | None = 42) -> list[int]:
    """n random integers (repeatable by default, so everyone sees the same numbers)."""
    rng = random.Random(seed)
    high = high if high is not None else max(10, n * 3)
    return [rng.randint(low, high) for _ in range(n)]