import numpy as np


def jacobi(A, b, x0=None, tol=1e-10, max_iter=1000):
    """ Solve Ax = b using Jacobi method. """
    x = np.zeros_like(b) if x0 is None else x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)
    for _ in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new
        x = x_new
    return x


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    """ Solve Ax = b using Gauss-Seidel method. """
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i])
                        - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new
        x = x_new
    return x


def sor(A, b, omega=1.0, x0=None, tol=1e-10, max_iter=1000):
    """ Solve Ax = b using SOR method. """
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sigma = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = x[i] + omega * ((b[i] - sigma) / A[i, i] - x[i])
        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new
        x = x_new
    return x


def gradient_method(A, b, x0=None, tol=1e-10, max_iter=None):
    """ SolveAx = b for A symmetric positive definite using Steepest Gradient Descent method. """
    n = len(b)
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()

    if max_iter is None:
        max_iter = n * 10

    r = b - A @ x
    for k in range(max_iter):
        Ar = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x_new = x + alpha * r
        r_new = r - alpha * Ar

        if np.linalg.norm(r_new) < tol:
            return x_new

        x = x_new
        r = r_new
    return x


def conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=None):
    """ Solve Ax=b for A symmetric positive definite using Conjugate Gradient method. """
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - A @ x
    p = r.copy()
    rsold = np.dot(r, r)
    if max_iter is None:
        max_iter = n
    for _ in range(max_iter):
        Ap = A @ p
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x