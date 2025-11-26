import numpy as np
from itertools import product

def decompose_cells(mask):
    visited = np.zeros_like(mask, dtype=bool)
    cells = []
    H, W = mask.shape
    for i, j in product(range(H), range(W)):
        if mask[i, j] and not visited[i, j]:
            stack = [(i, j)]
            visited[i, j] = True
            rows, cols = [], []
            while stack:
                r, c = stack.pop()
                rows.append(r); cols.append(c)
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    rr, cc = r+dr, c+dc
                    if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            cells.append((min(rows), max(rows), min(cols), max(cols)))
    return cells


def generate_sweeps_for_cell(cell, tool_width_cells, direction='horizontal'):
    min_r, max_r, min_c, max_c = cell
    sweeps = []
    if direction == 'horizontal':
        rows = list(range(min_r, max_r+1, tool_width_cells))
        for idx, r in enumerate(rows):
            cols = list(range(min_c, max_c+1))
            if idx % 2: cols.reverse()
            sweeps.append([(r, c) for c in cols])
    else:
        cols = list(range(min_c, max_c+1, tool_width_cells))
        for idx, c in enumerate(cols):
            rows = list(range(min_r, max_r+1))
            if idx % 2: rows.reverse()
            sweeps.append([(r, c) for r in rows])
    return sweeps


def tsp_ordering(points):
    if not points:
        return []
    pts = points.copy()
    tour = [pts.pop(0)]
    while pts:
        last = tour[-1]
        dists = [abs(last[0]-p[0]) + abs(last[1]-p[1]) for p in pts]
        idx = int(np.argmin(dists))
        tour.append(pts.pop(idx))
    return tour


def plan_h_bcp(H, mask, tool_width, cell_size=0.01):
    tool_width_cells = max(1, int(round(tool_width / cell_size)))
    cells = decompose_cells(mask)
    cell_centers = [((r0+r1)/2, (c0+c1)/2) for r0,r1,c0,c1 in cells]
    ordered_centers = tsp_ordering(cell_centers)
    ordered_cells = []
    for center in ordered_centers:
        for cell in cells:
            r0,r1,c0,c1 = cell
            if abs((r0+r1)/2-center[0])<1e-6 and abs((c0+c1)/2-center[1])<1e-6:
                ordered_cells.append(cell)
                break
    path = []
    for idx, cell in enumerate(ordered_cells):
        direction = 'horizontal' if idx % 2 == 0 else 'vertical'
        for line in generate_sweeps_for_cell(cell, tool_width_cells, direction):
            for r, c in line:
                z = H[r, c]
                path.append((c, r, z))
    return path