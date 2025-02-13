import numpy as np
import matplotlib.pyplot as plt


def naca4(m, p, t, num_points=100):
    """Generate a NACA 4-digit airfoil."""
    x = np.linspace(0, 1, num_points)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)

    yc = np.where(x < p, (m / p ** 2) * (2 * p * x - x ** 2), (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x ** 2))
    dyc_dx = np.where(x < p, (2 * m / p ** 2) * (p - x), (2 * m / (1 - p) ** 2) * (p - x))
    theta = np.arctan(dyc_dx)

    xu, xl = x - yt * np.sin(theta), x + yt * np.sin(theta)
    yu, yl = yc + yt * np.cos(theta), yc - yt * np.cos(theta)

    x_coords = np.concatenate([xu[::-1], xl[1:]])
    y_coords = np.concatenate([yu[::-1], yl[1:]])
    return x_coords, y_coords

def ellipse_airfoil(a, b, num_points=100):
    """Generate an ellipse-shaped airfoil with semi-major axis a and semi-minor axis b."""
    theta = np.linspace(0, np.pi, num_points)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    x_coords = np.concatenate([x[::-1], x[1:]])
    y_coords = np.concatenate([y[::-1], -y[1:]])
    return x_coords, y_coords


def load_airfoil(filename):
    """Load airfoil coordinates from a file."""
    data = np.loadtxt(filename, skiprows=1)
    return data[:, 0], data[:, 1]


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


if __name__ == "__main__":

    # Example: Generate NACA 2412
    x, y = naca4(m=0.02, p=0.4, t=0.12)
    #  plot_airfoil(x, y)


    # Example: Generate an ellipse airfoil
    x_ellipse, y_ellipse = ellipse_airfoil(a=1.0, b=0.08)
    plot_airfoil(x_ellipse, y_ellipse)

    # Example: Rotate airfoil by 10 degrees
    x_rot, y_rot = rotate_airfoil(x, y, angle=-10)

    plot_airfoil(x_rot, y_rot, "NACA 2412 Airfoil at 10 Degrees Angle of Attack")