from maze import generate_maze, print_maze
from solvers.bfs import bfs

if __name__ == "__main__":
    grid = generate_maze(21, 21)
    start, goal = (1, 1), (19, 19)
    path, explored = bfs(grid, start, goal)
    print_maze(grid, path)
    print(f"BFS: path length {len(path)}, nodes explored {explored}")