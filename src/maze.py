import random


def generate_maze(width=21, height=21, seed=None):
    """Return a grid where 1 = wall and 0 = path. Use odd sizes."""
    if seed is not None:
        random.seed(seed)
    grid = [[1] * width for _ in range(height)]
    grid[1][1] = 0
    stack = [(1, 1)]

    while stack:
        x, y = stack[-1]
        options = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 0 < nx < width - 1 and 0 < ny < height - 1 and grid[ny][nx] == 1:
                options.append((nx, ny, dx, dy))
        if options:
            nx, ny, dx, dy = random.choice(options)
            grid[y + dy // 2][x + dx // 2] = 0  # knock down the wall between
            grid[ny][nx] = 0
            stack.append((nx, ny))
        else:
            stack.pop()  # dead end, backtrack
    return grid


def print_maze(grid):
    for row in grid:
        print("".join("#" if cell else " " for cell in row))