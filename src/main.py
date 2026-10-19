from maze import generate_maze, print_maze
from solvers.bfs import bfs
from solvers.dfs import dfs

if __name__ == "__main__":
    grid = generate_maze(21, 21)
    start, goal = (1, 1), (19, 19)

    for name, solver in (("BFS", bfs), ("DFS", dfs)):
        path, explored = solver(grid, start, goal)
        print(f"\n{name}:")
        print_maze(grid, path)
        print(f"{name}: path length {len(path)}, nodes explored {explored}")