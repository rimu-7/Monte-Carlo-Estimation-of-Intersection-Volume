import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define the shapes
def is_inside_box(point, vertices):
    min_x = min(vertices[:, 0])
    max_x = max(vertices[:, 0])
    min_y = min(vertices[:, 1])
    max_y = max(vertices[:, 1])
    min_z = min(vertices[:, 2])
    max_z = max(vertices[:, 2])

    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y and min_z <= point[2] <= max_z

def is_inside_cylinder(point, c1, c2, radius):
    height_vector = c2 - c1
    height = np.linalg.norm(height_vector)
    axis = height_vector / height
    projection = np.dot(point - c1, axis)

    if projection < 0 or projection > height:
        return False

    perpendicular_vector = point - (c1 + projection * axis)
    distance = np.linalg.norm(perpendicular_vector)

    return distance <= radius

def is_inside_pyramid(point, vertices):
    apex = vertices[0]
    base_vertices = vertices[1:]

    if point[2] > apex[2]:
        return False

    base_plane = np.cross(base_vertices[1] - base_vertices[0], base_vertices[2] - base_vertices[0])
    base_plane = base_plane.astype(float)  # Ensure base_plane is a float array
    base_plane /= np.linalg.norm(base_plane)
    distance_to_plane = np.dot(point - base_vertices[0], base_plane)

    if distance_to_plane > 1e-6:
        return False

    min_x = min(base_vertices[:, 0])
    max_x = max(base_vertices[:, 0])
    min_y = min(base_vertices[:, 1])
    max_y = max(base_vertices[:, 1])

    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y

def is_inside_all_shapes(point, box_vertices, cyl_c1, cyl_c2, cyl_radius, pyramid_vertices):
    return (is_inside_box(point, box_vertices) and
            is_inside_cylinder(point, cyl_c1, cyl_c2, cyl_radius) and
            is_inside_pyramid(point, pyramid_vertices))

def monte_carlo_estimation(num_points, box_vertices, cyl_c1, cyl_c2, cyl_radius, pyramid_vertices):
    bounding_box_min, bounding_box_max = calculate_bounding_box(box_vertices, cyl_c1, cyl_c2, cyl_radius, pyramid_vertices)
    points = np.random.rand(num_points, 3) * (bounding_box_max - bounding_box_min) + bounding_box_min

    with Pool() as pool:
        results = pool.starmap(is_inside_all_shapes, [(point, box_vertices, cyl_c1, cyl_c2, cyl_radius, pyramid_vertices) for point in points])

    volume_bounding_box = np.prod(bounding_box_max - bounding_box_min)
    volume_intersection = (sum(results) / num_points) * volume_bounding_box
    return volume_intersection

def calculate_bounding_box(box_vertices, cyl_c1, cyl_c2, cyl_radius, pyramid_vertices):
    all_points = np.vstack([box_vertices, cyl_c1, cyl_c2, pyramid_vertices])
    min_bounds = np.min(all_points, axis=0)
    max_bounds = np.max(all_points, axis=0)

    min_bounds -= cyl_radius
    max_bounds += cyl_radius

    return min_bounds, max_bounds

def plot_shapes(box_vertices, cyl_c1, cyl_c2, cyl_radius, pyramid_vertices):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the box
    box_edges = [
        [box_vertices[i] for i in [0, 1, 5, 4]],
        [box_vertices[i] for i in [1, 2, 6, 5]],
        [box_vertices[i] for i in [2, 3, 7, 6]],
        [box_vertices[i] for i in [3, 0, 4, 7]],
        [box_vertices[i] for i in [0, 1, 2, 3]],
        [box_vertices[i] for i in [4, 5, 6, 7]]
    ]
    for edge in box_edges:
        poly = Poly3DCollection([edge], alpha=0.25)
        poly.set_facecolor((0, 1, 0, 0.1))
        ax.add_collection3d(poly)

    # Plot the cylinder
    cyl_height = np.linalg.norm(cyl_c2 - cyl_c1)
    cyl_axis = (cyl_c2 - cyl_c1) / cyl_height
    theta = np.linspace(0, 2 * np.pi, 100)
    cyl_bottom = np.array([
        [cyl_c1[0] + cyl_radius * np.cos(t), cyl_c1[1] + cyl_radius * np.sin(t), cyl_c1[2]]
        for t in theta
    ])
    cyl_top = cyl_bottom + cyl_height * cyl_axis

    # Reshape Z coordinates for plot_surface
    cyl_bottom_x, cyl_bottom_y, cyl_bottom_z = cyl_bottom.T
    cyl_bottom_z = cyl_bottom_z.reshape((1, -1))
    cyl_top_x, cyl_top_y, cyl_top_z = cyl_top.T
    cyl_top_z = cyl_top_z.reshape((1, -1))

    ax.plot_surface(cyl_bottom_x, cyl_bottom_y, cyl_bottom_z, color='r', alpha=0.1)
    ax.plot_surface(cyl_top_x, cyl_top_y, cyl_top_z, color='r', alpha=0.1)
    for i in range(len(cyl_bottom)):
        ax.plot([cyl_bottom[i, 0], cyl_top[i, 0]], [cyl_bottom[i, 1], cyl_top[i, 1]], [cyl_bottom[i, 2], cyl_top[i, 2]], color='r', alpha=0.1)

    # Plot the pyramid
    pyramid_edges = [
        [pyramid_vertices[0], pyramid_vertices[i]] for i in range(1, 5)
    ] + [
        [pyramid_vertices[i], pyramid_vertices[i % 4 + 1]] for i in range(1, 5)
    ]
    for edge in pyramid_edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [edge[0][2], edge[1][2]], color='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Example usage
box_vertices = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 0.0, 1.0]
])
cyl_c1 = np.array([0.5, 0.5, 1.0])
cyl_c2 = np.array([0.5, 0.5, 0.0])
cyl_radius = 0.5
pyramid_vertices = np.array([
    [0, 0, 1],
    [1, 1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0]
])

# Visualize the shapes
plot_shapes(box_vertices, cyl_c1, cyl_c2, cyl_radius, pyramid_vertices)

# Test points
test_points = np.array([
    [0.5, 0.5, 0.5],  # Inside the box
    [0.5, 0.5, 0.75],  # Inside the cylinder
    [0.0, 0.0, 0.5]    # Inside the pyramid
])

# Test is_inside_box
print("Box test:", [is_inside_box(point, box_vertices) for point in test_points])

# Test is_inside_cylinder
print("Cylinder test:", [is_inside_cylinder(point, cyl_c1, cyl_c2, cyl_radius) for point in test_points])

# Test is_inside_pyramid
print("Pyramid test:", [is_inside_pyramid(point, pyramid_vertices) for point in test_points])

# Estimate the volume of intersection
volume = monte_carlo_estimation(10000000, box_vertices, cyl_c1, cyl_c2, cyl_radius, pyramid_vertices)
print(f"Estimated volume of intersection: {volume}")
