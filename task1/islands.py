"""The solution for the first task

Usage (example):
    python islands.py 2 2 1 0 1 0
"""

import argparse
import numpy as np


def isSafe(matrix:np.array, r:int, c:int, visited:np.array) -> bool:
    """Checking if the algorithm moved out of matrix greed and if the value is unvisited island"""
    m = len(matrix)
    n = len(matrix[0])
    return (0 <= r < m and 0 <= c < n and matrix[r][c] == 1 and not visited[r][c])

def dfs(matrix:np.array, r:int, c:int, visited:np.array):
    """DFS algorithm to search for land in the neighboring cells"""
    visited[r][c] = True

    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    for k in range(4):
        nr = r + directions[k][0]
        nc = c + directions[k][1]
        if isSafe(matrix, nr, nc, visited):
            dfs(matrix, nr, nc, visited)
     

def count_islands(m:int, n:int, matrix:np.array) -> int:
    """The function that combines isSafe and dfs and counts number of islands"""
    visited = np.zeros((m, n), dtype=bool)
    islands = 0
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1 and not visited[i][j]:
                    dfs(matrix, i, j, visited)
                    islands += 1
    return islands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument('Matrix', nargs='*', type=int, choices=[0, 1])
    a = parser.parse_args()
    m = a.M
    n = a.N
    matrix = np.array(a.Matrix).reshape(a.M, a.N)

    n_islands = count_islands(m, n, matrix)

    print(n_islands)

if __name__ == "__main__":
    main()