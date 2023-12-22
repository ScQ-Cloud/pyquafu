import matplotlib.pyplot as plt
import numpy as np

"""
Plotting state of single qubit on the Bloch sphere.

TODO:
1. Plot by density matrix, say, from single-qubit sub-system.
2. Plot geometrical representation of quantum operations.
3. Plot a chain of qubits.
"""


def angles_to_xyz(thetas, phis):
    """Transform angles to cartesian coordinates."""
    xs = np.sin(thetas) * np.cos(phis)
    ys = np.sin(thetas) * np.sin(phis)
    zs = np.cos(thetas)
    return xs, ys, zs


def xyz_to_angles(xs, ys, zs):
    """Transform cartesian coordinates to angles."""
    phis = np.arctan2(ys, xs)
    thetas = np.arccos(zs)
    return thetas, phis


def hex_to_rgb(hex_color):
    """Transform a hex color code to RGB (normalized float)."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Invalid hex color code")

    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return r, g, b


def plot_bloch_vector(v_x, v_y, v_z, title=""):
    """
    Plot the Bloch vector on the Bloch sphere.

    Args:
        v_x (float): x coordinate of the Bloch vector.
        v_y (float): y coordinate of the Bloch vector.
        v_z (float): z coordinate of the Bloch vector.
        title (str): title of the plot.

    Returns:
        ax: matplotlib axes of the Bloch sphere plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # surface of Bloch sphere
    theta = np.linspace(0, np.pi, 21)
    phi = np.linspace(0, 2 * np.pi, 21)
    theta, phi = np.meshgrid(theta, phi)
    x, y, z = angles_to_xyz(theta, phi)

    surf = ax.plot_surface(x, y, z, color="white", alpha=0.2)
    edge_color = hex_to_rgb("#000000")  # #ff7f0e
    edge_alpha = 0.05
    surf.set_edgecolor((edge_color[0], edge_color[1], edge_color[2], edge_alpha))

    # coordinate axes inside the sphere span
    span = np.linspace(-1.0, 1.0, 2)
    ax.plot(span, 0 * span, zs=0, zdir="z", label="X", lw=1, color="black", alpha=0.5)
    ax.plot(0 * span, span, zs=0, zdir="z", label="Y", lw=1, color="black", alpha=0.5)
    ax.plot(0 * span, span, zs=0, zdir="y", label="Z", lw=1, color="black", alpha=0.5)

    # coordinate values
    ax.text(1.4, 0, 0, "x", color="black")
    ax.text(0, 1.2, 0, "y", color="black")
    ax.text(0, 0, 1.2, "z", color="black")

    # Bloch vector
    ax.quiver(0, 0, 0, v_x, v_y, v_z, color="r")
    v_theta, v_phi = xyz_to_angles(v_x, v_y, v_z)

    # coordinates value text
    ax.text(
        0,
        0,
        1.6,
        "Bloch vector: ($\\theta=${:.2f}, $\\varphi$={:.2f})".format(v_theta, v_phi),
        fontsize=8,
        color="red",
    )
    # ax.text(0, 0, 1.6, 'Bloch vector: ({:.2f}, {:.2f}, {:.2f})'.format(v_x, v_y, v_z), fontsize=8, color='red')

    # Set the range of the axes
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(32, 32)
    ax.set_axis_off()
    ax.set_title(title)
    return ax
