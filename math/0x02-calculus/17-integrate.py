#!/usr/bin/env python3
"""Derivative"""


def poly_integral(poly, C=0):
    """Performs integrate of a polynomial.
    Ej:
    ∫(5 + 3x + x³)dx ----> C + 5x + (3/2)x² + (1/4)x⁴
    [5, 3, 0, 1]     ----> [C, 5, 1.5, 0, 0.25]

    Args:
        poly (list): polynomial.
        C: integral constant.

    Returns:
        list: polynomial integrated, included the constant C.
    """
    # Check if poly is a valid list.
    if poly == [] or type(poly) is not list:
        return None
    for n in poly:
        if type(n) is not int and type(n) is not float:
            return None

    coefficient = [1] + poly
    integrals = [C] + [1 / (i + 1) for i in range(len(poly))]

    result = [coefficient[i] * integrals[i] for i in range(len(coefficient))]
    return [int(n) if n % 1 == 0 else n for n in result]
