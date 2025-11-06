import numpy as np

def rect(f, a, b, n):
    """ Estimate integral of f(x) over interval [a,b] using left rectangle rule with n sub-intervals."""

    x = np.linspace(a, b - (b - a)/n, n)
    return ((b - a) / n) * np.sum(f(x))

def trap(f, a, b, n):
    """ Estimate integral of f(x) over [a,b] using trapezoidal rule with n sub-intervals."""

    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (0.5 * (y[0] + y[-1]) + np.sum(y[1:-1]))

def simpson(f, a, b, n):
    """ Estimate integral of f(x) over [a,b] using simpson's rule and n sub-intervals."""
    if n % 2 == 1:
        raise ValueError("n must be even for Simpson's rule")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return (h / 3) * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))