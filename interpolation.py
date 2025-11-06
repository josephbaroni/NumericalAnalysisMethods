import numpy as np

def divdif(xs,ys):
    """ Compute Newton divided differences table. """
    n = len(xs)
    dd = np.copy(ys).astype(float)
    for i in range(1, n):
        dd[i:n] = (dd[i:n]-dd[i-1:n-1]) / (xs[i:n]-xs[0:n-i])
    return dd

def dd_interp(xs,dd,t):
    """ Evaluate divided difference polynomial at points t. """
    n = len(xs)-1
    val = dd[n]
    for i in range(n-1,-1,-1):
        val = dd[i] + (t - xs[i]) * val
    return val

def lagrange_interp(xs, ys, t):
    """ Evaluate Lagrange interpolation polynomial at points t."""
    n = len(xs)
    t = np.atleast_1d(t)
    vals = np.zeros_like(t, dtype=float)
    for i in range(n):
        li = np.ones_like(t)
        for j in range(n):
            if i != j:
                li *= (t - xs[j]) / (xs[i] - xs[j])
        vals += ys[i] * li
    return vals

def hermite_interp(xs, ys, dys, t):
    """ Cubic Hermite interpolation. """
    n = len(xs)
    t = np.atleast_1d(t)
    vals = np.zeros_like(t, dtype=float)
    for i in range(n-1):
        x0, x1 = xs[i], xs[i+1]
        y0, y1 = ys[i], ys[i+1]
        dy0, dy1 = dys[i], dys[i+1]
        h = x1 - x0
        s = (t - x0) / h
        h00 = (1 + 2*s)*(1 - s)**2
        h10 = s*(1 - s)**2
        h01 = s**2*(3 - 2*s)
        h11 = s**2*(s - 1)
        vals += (h00*y0 + h10*h*dy0 + h01*y1 + h11*h*dy1) * ((t >= x0) & (t <= x1))
    return vals


def cubic_spline(xs, ys):
    """Compute natural cubic spline coefficients."""
    n = len(xs) - 1
    h = np.diff(xs)  
    # set up tridiagonal system for c coefficients
    A = np.zeros((n+1, n+1))
    b = np.zeros(n+1)
    # Natural spline boundary conditions
    A[0, 0] = 1
    A[n, n] = 1  
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * ((ys[i+1] - ys[i]) / h[i] - (ys[i] - ys[i-1]) / h[i-1])
    # Solve for c coefficients
    c = np.linalg.solve(A, b)
    # compute b_i and d_i
    a = ys[:-1]
    b_ = np.zeros(n)
    d = np.zeros(n)
    for i in range(n):
        b_[i] = (ys[i+1] - ys[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    return a, b_, c[:-1], d, xs

def spline_eval(a, b, c, d, xs, t):
    """ Evaluate spline at points t."""
    t = np.atleast_1d(t)
    vals = np.zeros_like(t, dtype=float)
    for j, tj in enumerate(t):
        # Find interval i such that xs[i] <= tj < xs[i+1]
        i = np.searchsorted(xs, tj) - 1
        i = np.clip(i, 0, len(xs)-2)
        dx = tj - xs[i]
        vals[j] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    return vals
