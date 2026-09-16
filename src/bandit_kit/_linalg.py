"""Tiny dense linear-algebra helpers so LinUCB can stay numpy-free."""

from __future__ import annotations

from typing import Sequence


Vector = list[float]
Matrix = list[list[float]]


def identity(n: int, scale: float = 1.0) -> Matrix:
    """Return ``scale * I_n``."""
    if n < 1:
        raise ValueError("dimension must be at least 1")
    return [[scale if i == j else 0.0 for j in range(n)] for i in range(n)]


def zeros(n: int) -> Vector:
    if n < 1:
        raise ValueError("dimension must be at least 1")
    return [0.0] * n


def copy_vector(vector: Sequence[float]) -> Vector:
    return [float(value) for value in vector]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot product requires equal-length vectors")
    return sum(a * b for a, b in zip(left, right))


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    if not matrix:
        raise ValueError("matrix must not be empty")
    if len(matrix[0]) != len(vector):
        raise ValueError("matrix-vector size mismatch")
    return [dot(row, vector) for row in matrix]


def add_vectors(left: Sequence[float], right: Sequence[float]) -> Vector:
    if len(left) != len(right):
        raise ValueError("vector size mismatch")
    return [a + b for a, b in zip(left, right)]


def scale_vector(vector: Sequence[float], scalar: float) -> Vector:
    return [scalar * value for value in vector]


def outer(left: Sequence[float], right: Sequence[float] | None = None) -> Matrix:
    right_vec = left if right is None else right
    return [[a * b for b in right_vec] for a in left]


def add_scaled_matrix(
    matrix: Sequence[Sequence[float]],
    other: Sequence[Sequence[float]],
    scalar: float,
) -> Matrix:
    rows = len(matrix)
    cols = len(matrix[0])
    if len(other) != rows or len(other[0]) != cols:
        raise ValueError("matrix size mismatch")
    return [
        [matrix[i][j] + scalar * other[i][j] for j in range(cols)]
        for i in range(rows)
    ]


def sherman_morrison_update(
    inverse: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> Matrix:
    """Return ``(A + x x^T)^{-1}`` given ``A^{-1}`` via Sherman-Morrison.

    ``(A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)``
    """
    mapped = matvec(inverse, vector)
    denom = 1.0 + dot(vector, mapped)
    if abs(denom) < 1e-18:
        raise ValueError("Sherman-Morrison denominator is degenerate")
    return add_scaled_matrix(inverse, outer(mapped, mapped), -1.0 / denom)


def quadratic_form(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> float:
    """Return ``x^T M x``."""
    return dot(vector, matvec(matrix, vector))


__all__ = [
    "add_scaled_matrix",
    "add_vectors",
    "copy_vector",
    "dot",
    "identity",
    "matvec",
    "outer",
    "quadratic_form",
    "scale_vector",
    "sherman_morrison_update",
    "zeros",
]
