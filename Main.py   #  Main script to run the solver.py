import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Header: Simulation Parameters
# ==========================

# Airfoil Selection
AIRFOIL_TYPE = "NACA4"  # Options: "NACA4", "ELLIPSE", "FILE"
NACA4_PARAMS = {"m": 0.02, "p": 0.4, "t": 0.12}  # Only used if AIRFOIL_TYPE is "NACA4"
AIRFOIL_FILENAME = "airfoil.dat"  # Only used if AIRFOIL_TYPE is "FILE"

# Geometry
NUM_PANELS = 100  # Number of points/panels for discretization
ELLIPSE_PARAMS = {"a": 1.0, "b": 0.2}  # Only used if AIRFOIL_TYPE is "ELLIPSE"

# Flow Conditions
ANGLE_OF_ATTACK = 10.0  # Degrees

# Display Settings
SHOW_AIRFOIL_SHAPE = True

# ==========================
# Functions for Airfoil Generation
# ==========================

def naca4(m, p, t, num_points=100):
    """Generate a NACA 4-digit airfoil."""
    x = np.linspace(0, 1, num_points)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)

    yc = np.piecewise(x, [x < p, x >= p],
                      [lambda x: (m / p ** 2) * (2 * p * x - x ** 2),
                       lambda x: (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x ** 2)])
    dyc_dx = np.piecewise(x, [x < p, x >= p],
                          [lambda x: (2 * m / p ** 2) * (p - x),
                           lambda x: (2 * m / (1 - p) ** 2) * (p - x)])
    theta = np.arctan(dyc_dx)

    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    xu, xl = x - yt * sin_theta, x + yt * sin_theta
    yu, yl = yc + yt * cos_theta, yc - yt * cos_theta

    x_coords = np.concatenate((xu[::-1], xl[1:]))
    y_coords = np.concatenate((yu[::-1], yl[1:]))
    return x_coords, y_coords


def load_airfoil(filename):
    """Load airfoil coordinates from a file."""
    data = np.loadtxt(filename, skiprows=1)
    return data[:, 0], data[:, 1]


def ellipse_airfoil(a, b, num_points=100):
    """Generate an ellipse-shaped airfoil with semi-major axis a and semi-minor axis b."""
    theta = np.linspace(0, np.pi, num_points)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    x_coords = np.concatenate((x[::-1], x[1:]))
    y_coords = np.concatenate((y[::-1], -y[1:]))
    return x_coords, y_coords


def rotate_airfoil(x, y, angle):
    """Rotate airfoil by a given angle in degrees."""
    theta = np.radians(angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta
    return x_rot, y_rot


def plot_airfoil(x, y, title="Airfoil Shape"):
    """Plot the airfoil shape."""
    plt.figure(figsize=(8, 3))
    plt.plot(x, y, 'k-', linewidth=2)
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    # Generate airfoil based on selected type
    if AIRFOIL_TYPE == "NACA4":
        x, y = naca4(**NACA4_PARAMS, num_points=NUM_PANELS)
        title = f"NACA {int(NACA4_PARAMS['m']*100)}{int(NACA4_PARAMS['p']*10)}{int(NACA4_PARAMS['t']*100)} Airfoil"

    elif AIRFOIL_TYPE == "ELLIPSE":
        x, y = ellipse_airfoil(**ELLIPSE_PARAMS, num_points=NUM_PANELS)
        title = "Ellipse Airfoil"

    elif AIRFOIL_TYPE == "FILE":
        x, y = load_airfoil(AIRFOIL_FILENAME)
        title = "Loaded Airfoil from File"

    else:
        raise ValueError(f"Invalid AIRFOIL_TYPE: {AIRFOIL_TYPE}")

    # Rotate the airfoil
    x, y = rotate_airfoil(x, y, ANGLE_OF_ATTACK)

    # Plot the airfoil
    if SHOW_AIRFOIL_SHAPE:
        plot_airfoil(x, y, title=f"{title} at {ANGLE_OF_ATTACK} Degrees")
