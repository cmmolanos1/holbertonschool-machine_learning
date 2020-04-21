#!/usr/bin/env python3
"""Derivative"""


def poly_derivative(poly):
    """Performs a derivative of a polynomial.
    Ej:
    d(5 + 3x + x³)/dx ----> 3 + 3x²
    [5, 3, 0, 1]       ----> [3, 0, 3]

    Args:
        poly (list): the polynomial ordered by coefficients.

    Returns:
        list: the derivative polynomial.
    """
    # Check if poly is a valid list.
    if poly == [] or type(poly) is not list:
        return None
    for n in poly:
        if type(n) is not int and type(n) is not float:
            return None

    # Create a list with X-exponents
    exponents = [n for n in range(len(poly))]

    if len(poly) == 1:
        return [0]
    else:
        # Multiply exponents and coefficients, and delete the constant.
        result = [exponents[i] * poly[i] for i in range(1, len(poly))]

        return result if any(result) else [0]
