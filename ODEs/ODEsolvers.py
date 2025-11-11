import numpy as np


# Forward Euler Method (Explicit)
# solve y_{n+1} = y_n + h * f(t_n, y_n) 
def forward_euler(f, y0, t0, tf, h):
    steps = int((tf-t0)/h) + 1
    t = np.linspace(t0, tf, steps)
    y = np.zeros(steps)
    y[0] = y0

    for n in range(1, steps):
        y[n] = y[n-1] + h * f(t[n-1], y[n-1])  # Forward Euler update
    return t, y


# Backward Euler Method (Implicit)
# solve y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
def backward_euler(f, y0, t0, tf, h, max_iter=100, tol=1e-6):
    steps = int((tf-t0)/h) + 1
    t = np.linspace(t0, tf, steps)
    y = np.zeros(steps)
    y[0] = y0

    for n in range(1, steps):
        yn1 = y[n-1]
        for _ in range(max_iter):   # Fixed point iteration
            yn1_new = y[n-1] + h * f(t[n], yn1)
            if np.abs(yn1_new - yn1) < tol:
                break
            yn1 = yn1_new
        y[n] = yn1
    return t, y


# Crank-Nicolson Method (Implicit)
# solve y_{n+1} = y_n + (h / 2) * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))
def crank_nicolson(f, y0, t0, tf, h, max_iter=100, tol=1e-6):
    steps = int((tf-t0)/h) + 1
    t = np.linspace(t0, tf, steps)
    y = np.zeros(steps)
    y[0] = y0

    for n in range(1, steps):
        yn1 = y[n-1]
        for _ in range(max_iter):   # fixed-point iteration
            yn1_new = y[n-1] + 0.5 * h * (f(t[n-1], y[n-1]) + f(t[n], yn1))
            if np.abs(yn1_new - yn1) < tol:
                break
            yn1 = yn1_new
        y[n] = yn1
    return t, y


# Heun's Method (Improved Euler)
# solve y_{n+1} = y_n + h/2 * (k1 + k2) for k1 = f(t_n, y_n), k2 = f(t_n + h, y_n + h*k1)
def heun(f, y0, t0, tf, h):
    steps = int((tf-t0)/h) + 1
    t = np.linspace(t0, tf, steps)
    y = np.zeros(steps)
    y[0] = y0

    for n in range(1, steps):
        k1 = f(t[n-1], y[n-1])
        k2 = f(t[n-1] + h, y[n-1] + h * k1)
        y[n] = y[n-1] + 0.5 * h * (k1 + k2)   # Update solution
    return t, y


# Runge-Kutta 4th Order Method
# solve y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2k3 + k4)
def runge_kutta4(f, y0, t0, tf, h):
    steps = int((tf-t0)/h) + 1
    t = np.linspace(t0, tf, steps)
    y = np.zeros(steps)
    y[0] = y0

    for n in range(1, steps):
        k1 = f(t[n-1], y[n-1])
        k2 = f(t[n-1] + h / 2, y[n-1] + h * k1 / 2)
        k3 = f(t[n-1] + h / 2, y[n-1] + h * k2 / 2)
        k4 = f(t[n-1] + h, y[n-1] + h * k3)
        y[n] = y[n-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # RK4 update
    return t, y


# Adams-Bashforth 2nd Order Method
def adams_bashforth(f, y0, t0, tf, h):
    steps = int((tf-t0)/h) + 1
    t = np.linspace(t0, tf, steps)
    y = np.zeros(steps)
    y[0] = y0
    _, y_euler = forward_euler(f, y0, t0, t0 + h, h)    # Use Forward Euler for first step
    y[1] = y_euler[1]          # Set the second value

    # Apply Adams-Bashforth 2nd order method for the rest of the steps
    for n in range(1, steps-1):
        y[n+1] = y[n] + h / 2 * (3 * f(t[n], y[n]) - f(t[n-1], y[n-1]))  # Update
    return t, y

# Adams-Moulton 2nd order method
def adams_moulton(f, y0, t0, tf, h):
    steps = int((tf-t0)/h) + 1
    t = np.linspace(t0, tf, steps)
    y = np.zeros(steps)
    y[0] = y0
    _, y_euler = forward_euler(f, y0, t0, t0 + h, h)    # Use Forward Euler for first step
    y[1] = y_euler[1]          # Set the second value
    
    # Apply Adams-Moulton 2nd order method for the rest of the steps
    for n in range(1, steps-1):
        yn1 = y[n] + h / 12 * (5 * f(t[n+1], y[n]) + 8 * f(t[n], y[n]) - f(t[n-1], y[n-1]))
        y[n+1] = yn1  # Update
    return t, y

