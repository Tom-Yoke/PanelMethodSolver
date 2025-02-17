# Pure Source Panel Method Implementation
# This script solves the potential flow around an airfoil using constant strength source panels.
# Outputs: Streamlines of the flow and surface pressure coefficient (Cp) distribution.
#
# Assigned Parameters
N_PANELS = 50  # Number of panels for airfoil discretization
U_INF = 1.0  # Free-stream velocity
ALPHA = 0.0  # Angle of attack in degrees
X_MIN, X_MAX = -0.5, 1.5  # Grid limits for velocity field
Y_MIN, Y_MAX = -0.5, 0.5
GRID_RES = 50  # Grid resolution

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------
# Airfoil Definition
# -----------------------------------------------
def define_airfoil(airfoil_type='NACA0012', N=N_PANELS):
    if airfoil_type == 'NACA0012':
        theta = np.linspace(0, np.pi, N)
        x = 0.5 * (1 - np.cos(theta))  # Cosine spacing
        y = 0.1 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)
        x = np.concatenate([x, x[::-1]])  # Symmetric
        y = np.concatenate([y, -y[::-1]])
    elif airfoil_type == 'ellipse':
        a, b = 1.0, 0.15  # Semi-major and semi-minor axis
        theta = np.linspace(0, 2 * np.pi, N)
        x, y = a * np.cos(theta), b * np.sin(theta)
    elif airfoil_type == 'data':
        x, y = np.loadtxt('airfoil_data.txt', unpack=True)
    else:
        raise ValueError("Invalid airfoil type. Choose 'NACA0012', 'ellipse', or 'data'.")
    return x, y


# -----------------------------------------------
# Panel Discretization
# -----------------------------------------------
def define_panels(x, y, alpha=ALPHA):
    N = len(x) - 1
    panels = []
    alpha_rad = np.radians(alpha)
    cos_alpha, sin_alpha = np.cos(alpha_rad), np.sin(alpha_rad)

    for i in range(N):
        x_rot = cos_alpha * x[i] - sin_alpha * y[i]
        y_rot = sin_alpha * x[i] + cos_alpha * y[i]
        x_next_rot = cos_alpha * x[i + 1] - sin_alpha * y[i + 1]
        y_next_rot = sin_alpha * x[i + 1] + cos_alpha * y[i + 1]
        panels.append(((x_rot, y_rot), (x_next_rot, y_next_rot)))

    return np.array(panels)


# -----------------------------------------------
# Influence Coefficients & Solver
# -----------------------------------------------
def compute_influence_coeffs(panels):
    N = len(panels)
    A = np.zeros((N, N))
    rhs = np.zeros(N)

    for i in range(N):
        xi, yi = (panels[i][0][0] + panels[i][1][0]) / 2, (panels[i][0][1] + panels[i][1][1]) / 2
        ni = np.array([panels[i][1][1] - panels[i][0][1], -(panels[i][1][0] - panels[i][0][0])])
        ni /= np.linalg.norm(ni)  # Normal vector

        for j in range(N):
            if i != j:
                A[i, j] = compute_velocity_contributions(xi, yi, panels[j]) @ ni

        rhs[i] = -U_INF  # Free-stream velocity normal component

    return A, rhs


def solve_source_strengths(A, rhs):
    return np.linalg.solve(A, rhs)


# -----------------------------------------------
# Flow Computation
# -----------------------------------------------
def compute_velocity_field(X, Y, panels, sigma):
    u, v = np.ones_like(X) * U_INF, np.zeros_like(Y)

    for i in range(len(panels)):
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                V_contrib = compute_velocity_contributions(X[j, k], Y[j, k], panels[i]) * sigma[i]
                u[j, k] += V_contrib[0]
                v[j, k] += V_contrib[1]

    return u, v


def compute_pressure_coefficient(panels, sigma):
    Cp = np.zeros(len(panels))
    for i in range(len(panels)):
        V = np.linalg.norm(compute_velocity_contributions((panels[i][0][0] + panels[i][1][0]) / 2,
                                                          (panels[i][0][1] + panels[i][1][1]) / 2, panels[i]) * sigma[
                               i] + np.array([U_INF, 0]))
        Cp[i] = 1 - V ** 2
    return Cp


# -----------------------------------------------
# Post-Processing: Visualization
# -----------------------------------------------
def plot_results(x, y, X, Y, u, v, Cp):
    plt.figure(figsize=(10, 5))
    plt.streamplot(X, Y, u, v, density=1.5, linewidth=0.5, color='b')
    plt.plot(x, y, 'k-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Streamlines for Pure Source Panel Method')
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(x[:-1], Cp, 'ro-', markersize=4)
    plt.gca().invert_yaxis()
    plt.xlabel('x')
    plt.ylabel('$C_p$')
    plt.title('Surface Pressure Coefficient')
    plt.show()
