from collections import deque


def bfs(grid, start, goal):
    """Return (path, explored_count). path is a list of (x, y) or None."""
    queue = deque([start])
    came_from = {start: None}
    explored = 0

    while queue:
        current = queue.popleft()
        explored += 1
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return path[::-1], explored
        x, y = current
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if grid[ny][nx] == 0 and (nx, ny) not in came_from:
                came_from[(nx, ny)] = current
                queue.append((nx, ny))
    return None, explored