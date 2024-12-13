import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim


def gaussian_3d(x, y, z, x0, y0, z0, sigma, amplitude):
    """
    Computes the 3D Gaussian function.
    x, y, z: 3D tensor grid coordinates.
    x0, y0, z0: Center of the Gaussian.
    sigma: Standard deviation (spread).
    amplitude: Peak intensity of the Gaussian.
    """
    return amplitude * torch.exp(-((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / (2 * sigma**2))


def generate_3d_gaussian_volume(device, grid_size, points, sigma):
    """
    Creates a 3D volume by adding Gaussian splats for each point using PyTorch tensors.
    device: 
    grid_size: Tuple (x_size, y_size, z_size) defining the grid size.
    points: Tensor of shape (n_points, 4), where each row is (x0, y0, z0, intensity).
    sigma: Standard deviation for the Gaussian spread.
    """
    x_size, y_size, z_size = grid_size

    # Create a 3D grid of coordinates
    x = torch.linspace(0, x_size - 1, x_size)
    y = torch.linspace(0, y_size - 1, y_size)
    z = torch.linspace(0, z_size - 1, z_size)
    X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
    X = X.to(device)
    Y = Y.to(device)
    Z = Z.to(device)

    # Initialize a 3D volume
    volume = torch.zeros((x_size, y_size, z_size), dtype=torch.float32).to(device)

    # Add Gaussian splats for each point
    for point in points:
        x0, y0, z0, intensity = point
        volume += gaussian_3d(X, Y, Z, x0, y0, z0, sigma, intensity)

    return volume


# Function to project 3D points with a world matrix and a projection matrix
def project_points_with_world_matrix(points, world_matrix, projection_matrix):
    """
    Projects 3D points into 2D image space using a world matrix and a projection matrix.

    Parameters:
    - points: List of tuples [(x, y, z, intensity), ...] in local coordinates.
    - world_matrix: 4x4 transformation matrix for world space.
    - projection_matrix: 3x4 projection matrix for camera projection.

    Returns:
    - List of projected Gaussians [(px, py, intensity, sigma), ...].
    """
    projected_gaussians = []
    
    for x, y, z, intensity in points:
        # Convert to homogeneous coordinates
        x = world_matrix[0][0] * x + world_matrix[0][1] * y + world_matrix[0][2] * z + world_matrix[0][3]
        y = world_matrix[1][0] * x + world_matrix[1][1] * y + world_matrix[1][2] * z + world_matrix[1][3]
        z = world_matrix[2][0] * x + world_matrix[2][1] * y + world_matrix[2][2] * z + world_matrix[2][3]
        x = x / z
        y = y / z
        
        ## Transform to world space
        #point_world = world_matrix @ point_local
        #
        ## Apply the projection matrix
        #projected_point = projection_matrix @ point_world
        #
        ## Normalize to get 2D coordinates
        #px = projected_point[0] / projected_point[2]
        #py = projected_point[1] / projected_point[2]
        projected_gaussians.append((x, y, intensity, torch.tensor(10.0)))  # Default sigma = 10

    return projected_gaussians


# Function to render 2D Gaussians into an image
def render_gaussians_to_image(gaussians_2d, image_shape, device):
    """
    Renders 2D Gaussians into an image grid.
    
    Parameters:
    - gaussians_2d: List of projected Gaussians [(px, py, intensity, sigma), ...].
    - image_shape: Tuple (height, width) of the output image.
    - device: device
    
    Returns:
    - A torch tensor representing the rendered image.
    """
    height, width = image_shape
    x = torch.linspace(0, width - 1, width)
    y = torch.linspace(0, height - 1, height)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    X = X.to(device)
    Y = Y.to(device)
    # Initialize the image
    image = torch.zeros((height, width), dtype=torch.float32).to(device)
    
    # Render each Gaussian onto the grid
    for px, py, intensity, sigma in gaussians_2d:
        gaussian = intensity * torch.exp(-((X - px)**2 + (Y - py)**2) / (2 * sigma**2))
        image += gaussian
    
    return torch.clamp(image, 0, 1)


def move_matrices_to_device(matrices, device):
    """
    Moves a list of matrices to the specified device (e.g., GPU).

    Args:
        matrices (list of torch.Tensor or numpy.ndarray): List of matrices to move.
        device (torch.device): Target device (e.g., 'cuda' or 'cpu').

    Returns:
        list of torch.Tensor: List of matrices on the target device.
    """
    gpu_matrices = []
    for matrix in matrices:
        # Ensure the matrix is a PyTorch tensor
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix)  # Convert to tensor if needed
        
        # Move the tensor to the target device
        gpu_matrices.append(matrix.to(device))
    
    return gpu_matrices


def optimize_gaussians(input_images, initial_gaussians, world_matrices, projection_matrix, image_shape, sigma, device):
    """
    Optimizes Gaussian parameters to match input images from multiple views.

    Parameters:
    - input_images: List of ground truth images (2D tensors).
    - initial_gaussians: Tensor of shape (n_gaussians, 4) for (x, y, z, intensity).
    - world_matrices: List of world matrices, one for each view.
    - projection_matrix: Camera projection matrix (3x4).
    - image_shape: Shape of the rendered images (height, width).
    - sigma: Standard deviation for the Gaussian spread.

    Returns:
    - Optimized Gaussian parameters (x, y, z, intensity).
    """
    # Initialize optimizer    
    initial_gaussians = initial_gaussians.to(device)
    initial_gaussians = initial_gaussians.detach().clone().requires_grad_(True)
    initial_gaussians.requires_grad_(True)
    optimizer = optim.Adam([initial_gaussians], lr=0.01)

    # Optimization loop
    num_epochs = 3000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0

        # Loop through views
        for i, (input_image, world_matrix) in enumerate(zip(input_images, world_matrices)):
            # Project Gaussians for the current view
            params = initial_gaussians
            #x, y, z, intensity = params[0], params[1], params[2], params[3]
            projected_gaussians = project_points_with_world_matrix(
                params, world_matrix.to(device), projection_matrix
            )

            # Render the current view
            rendered_image = render_gaussians_to_image(projected_gaussians, image_shape, device)

            # Compute MSE loss
            loss = torch.nn.functional.mse_loss(rendered_image, input_image)
            total_loss += loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Logging
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item()}")

    return initial_gaussians


def generate_tree_point_cloud(device, scale=1.0, n_points_trunk=500, n_points_branches=2000, n_points_leaves=5000):
    """
    Generates a sparse 3D point cloud representing a tree.
    
    Args:
        n_points_trunk (int): Number of points for the trunk.
        n_points_branches (int): Number of points for the branches.
        n_points_leaves (int): Number of points for the leaves.
    
    Returns:
        torch.Tensor: Sparse point cloud as a tensor of shape (N, 3), where N is the total number of points.
    """
    # Trunk (vertical cylinder)
    trunk_radius = 0.2
    trunk_height = 3.0
    trunk_points = []
    trunk_densities = []

    for _ in range(n_points_trunk):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, trunk_radius)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = np.random.uniform(0, trunk_height)
        
        # Higher density for trunk points
        density = 1.0
        trunk_points.append((x * scale, y * scale, z * scale, density))
    
    # Branches (random smaller cylinders extending outward)
    branches_points = []
    branches_densities = []
    n_branches = 5
    branch_length = 1.5
    for i in range(n_branches):
        branch_angle = np.random.uniform(0, 2 * np.pi)
        branch_height = np.random.uniform(trunk_height * 0.5, trunk_height)  # Branch starts mid-trunk or higher
        branch_dir = np.array([np.cos(branch_angle), np.sin(branch_angle), 1.0])  # Upward direction with tilt
        branch_dir /= np.linalg.norm(branch_dir)
        
        for _ in range(n_points_branches // n_branches):
            t = np.random.uniform(0, branch_length)
            offset = t * branch_dir + np.random.normal(scale=0.05, size=3)  # Add random scatter
            # Moderate density for branches
            density = 0.6
            branches_points.append((offset[0] * scale, offset[1] * scale, (branch_height + offset[2]) * scale, density))
    
    # Leaves (ellipsoid canopy)
    leaves_center = np.array([0, 0, trunk_height + 0.5])  # Above the trunk
    leaves_radii = np.array([1.5, 1.5, 2.0])  # Ellipsoid radii
    leaves_points = []
    leaves_densities = []
    for _ in range(n_points_leaves):
        u = np.random.uniform(0, 2 * np.pi)
        v = np.random.uniform(0, np.pi)
        x = leaves_radii[0] * np.sin(v) * np.cos(u)
        y = leaves_radii[1] * np.sin(v) * np.sin(u)
        z = leaves_radii[2] * np.cos(v)
        leaves_center + np.array([x, y, z])
        # Lower density for leaves points
        density = 0.2
        leaves_points.append((leaves_center[0] * scale, leaves_center[1] * scale, leaves_center[2] * scale, density))
    
    # Combine all points and densities
    all_points = np.vstack([trunk_points, branches_points, leaves_points])
    
    # Convert to PyTorch tensor
    point_cloud = torch.tensor(all_points, dtype=torch.float32).to(device)
    return point_cloud



def visualize_tree_point_cloud_with_density(point_cloud):
    """
    Visualizes a 3D point cloud with density values using PyVista.
    
    Args:
        point_cloud (torch.Tensor): Point cloud with coordinates and density values (N, 4).
    """
    # Convert to NumPy for visualization
    point_cloud_np = point_cloud.cpu().numpy()
    
    # Create a PyVista point cloud
    cloud = pv.PolyData(point_cloud_np[:, :3])  # Only use the first 3 columns (x, y, z)
    
    # Add density values as point data
    cloud.point_data["Density"] = point_cloud_np[:, 3]
    
    # Plot the point cloud with color based on density
    plotter = pv.Plotter()
    plotter.add_mesh(cloud, scalars="Density", cmap="viridis", point_size=5, render_points_as_spheres=True)
    plotter.show()


def gaussian_splat(x, y, z, density, sigma=0.5):
    """
    Generate a Gaussian splat centered at (x, y, z) with the given density and sigma (standard deviation).
    The density will affect the size and intensity of the Gaussian.
    
    Args:
        x, y, z (float): The coordinates of the point.
        density (float): The density value of the point, affecting the strength of the splat.
        sigma (float): Standard deviation of the Gaussian kernel.
        
    Returns:
        float: The intensity of the splat at the point (x, y, z).
    """
    return density * np.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))


def gaussian_splatting(point_cloud, image_size=(64, 64), sigma=0.5):
    """
    Perform Gaussian splatting on a point cloud to generate a 2D image.
    
    Args:
        point_cloud (torch.Tensor): The point cloud with coordinates (N, 4), where N is the number of points.
        image_size (tuple): The size of the rendered image.
        sigma (float): The standard deviation for the Gaussian kernels.
        
    Returns:
        np.ndarray: A 2D image of Gaussian splats.
    """
    # Initialize an empty image
    image = np.zeros(image_size)

    # Iterate through the point cloud
    for point in point_cloud:
        x, y, z, density = point.cpu().numpy()
        
        # Map 3D coordinates (x, y, z) to 2D image coordinates
        screen_x = int((x + 1) * (image_size[0] / 2))  # Normalize and map to image space
        screen_y = int((y + 1) * (image_size[1] / 2))  # Normalize and map to image space

        # Make sure the point is within the image bounds
        if 0 <= screen_x < image_size[0] and 0 <= screen_y < image_size[1]:
            # Add the Gaussian splat to the image
            intensity = gaussian_splat(x, y, z, density, sigma)
            image[screen_y, screen_x] += intensity

    # Normalize the image to [0, 1] for visualization
    image = np.clip(image, 0, 1)
    
    return image


def visualize_gaussian_splatting_image(image):
    """
    Visualize the Gaussian splatting result as a 3D surface using PyVista.
    
    Args:
        image (np.ndarray): The 2D Gaussian splat image.
    """
    # Create a mesh grid for the image
    y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    points = np.vstack([x.ravel(), y.ravel(), np.zeros_like(x.ravel())]).T
    
    # Create the point cloud for the image
    surface = pv.PolyData(points)
    
    # Add the image as the texture
    surface.point_data["scalars"] = image.ravel()
    
    # Visualize the Gaussian splat image as a surface
    plotter = pv.Plotter()
    plotter.add_mesh(surface, scalars="scalars", cmap="viridis", point_size=10, render_points_as_spheres=True)
    plotter.show()
    

# --- --- --- --- ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
# Example 3D points in local space
points = [
    (55, 55, 2, 1.0),  # Initial points
    (105, 105, 2, 0.6),
    (40, 100, 2, 0.6)
]
tree_points = generate_tree_point_cloud(device, 1.0, 500, 500, 1000)
visualize_tree_point_cloud_with_density(tree_points)
splat_image = gaussian_splatting(tree_points, image_size=(64, 64), sigma=0.5)
# Visualize the Gaussian splatting result
visualize_gaussian_splatting_image(splat_image)


# Function to dynamically add more points
def add_points(new_points):
    global points
    points.extend(new_points)  # Add new points to the existing list

points = torch.tensor(points, dtype=torch.float32)
#points = tree_points

# Image shape
image_shape = (100, 100)

# World matrix for identity transform
# Rotated and translated world matrix for second view
rotation = torch.tensor([
    [1.0, 0.0, 0, 0],
    [0.0, 1.0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)
translation = torch.tensor([
    [1, 0, 0, 20],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)
world_matrix_rotated_translated = translation @ rotation

# Rotated and translated world matrix for second view
rotation = torch.tensor([
    [0.866, -0.5, 0, 0],
    [0.5, 0.866, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)
translation = torch.tensor([
    [1, 0, 0, 60],
    [0, 1, 0, -10],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=torch.float32)
world_matrix_rotated_translated2 = translation @ rotation

# Projection matrix (identity for simplicity)
projection_matrix = torch.cat((torch.eye(3), torch.zeros((3, 1))), dim=1)

# Project points for View 1 and View 2
projected_view_1 = move_matrices_to_device(project_points_with_world_matrix(points, world_matrix_rotated_translated, projection_matrix), device)
projected_view_2 = move_matrices_to_device(project_points_with_world_matrix(points, world_matrix_rotated_translated2, projection_matrix), device)

# Render the two views as images
rendered_image_1 = render_gaussians_to_image(projected_view_1, image_shape, device)
rendered_image_2 = render_gaussians_to_image(projected_view_2, image_shape, device)

# Visualize the rendered images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("View 1: Identity World Matrix")
plt.imshow(rendered_image_1.cpu().numpy(), cmap="hot", origin="lower")
plt.colorbar(label="Intensity")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.subplot(1, 2, 2)
plt.title("View 2: Rotated & Translated World Matrix")
plt.imshow(rendered_image_2.cpu().numpy(), cmap="hot", origin="lower")
plt.colorbar(label="Intensity")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.tight_layout()
plt.show()


# Parameters
grid_size = (50, 50, 50)
grid_size = torch.tensor(grid_size).to(device)
sigma = torch.tensor(5).to(device)  # Spread of the Gaussian

# Generate the 3D Gaussian volume with updated points
volume = generate_3d_gaussian_volume(device, grid_size, points, sigma)

# Convert PyTorch tensor to NumPy array
#volume_np = volume.cpu().numpy()  # Ensure it's on the CPU and convert to NumPy
#
## Create a PyVista grid
#grid = pv.ImageData()
#grid.dimensions = volume_np.shape
#grid.spacing = (1, 1, 1)  # Specify spacing
#grid.origin = (0, 0, 0)  # Set the origin
#grid.point_data["values"] = volume_np.flatten(order="F")  # Add the volume data
#
## Threshold the volume for better visualization
#threshold = 0.01  # Visualization threshold
#grid = grid.threshold(threshold)
#
## Render the volume
#plotter = pv.Plotter()
#plotter.add_volume(grid, cmap="coolwarm", opacity="sigmoid")
#plotter.show()


input_images = [rendered_image_1.to(device), rendered_image_2.to(device)]
world_matrices = [world_matrix_rotated_translated, world_matrix_rotated_translated2]

# Initial guess for Gaussian parameters (x, y, z, intensity)
initial_gaussians = torch.tensor([
    [56.0, 57.0, 2.0, 1.0],  # Close guess to true points
    [109.0, 101.0, 2.2, 0.4],
    [39.0, 100.0, 1.9, 0.5]
], requires_grad=True)

optimized_gaussians = optimize_gaussians(
    input_images=input_images,
    initial_gaussians=initial_gaussians,
    world_matrices=world_matrices,
    projection_matrix=projection_matrix,
    image_shape=image_shape,
    sigma=sigma,
    device=device
)

print(optimized_gaussians)
optimized = project_points_with_world_matrix(optimized_gaussians, world_matrix_rotated_translated, projection_matrix)
# Render the two views as images
optimized_image = render_gaussians_to_image(optimized, image_shape, device)

# Visualize the rendered images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("View 1: Ground Truth")
plt.imshow(rendered_image_1.cpu().numpy(), cmap="hot", origin="lower")
plt.colorbar(label="Intensity")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.subplot(1, 2, 2)
plt.title("View 2: Optimized")
plt.imshow(optimized_image.cpu().detach().numpy(), cmap="hot", origin="lower")
plt.colorbar(label="Intensity")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.tight_layout()
plt.show()



## --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
## Define a simple quadratic function f(x) = (x - 3)^2
#def objective_function(x):
#    return (x - 3)**2
#
## Initialize x as a tensor with requires_grad=True, meaning we'll compute gradients for it
#x = torch.tensor([2.0], requires_grad=True)  # Initial guess with requires_grad=True
#
## Create an Adam optimizer to minimize the objective function
#optimizer = torch.optim.Adam([x], lr=0.1)  # learning rate = 0.1
#
## Number of optimization steps
#num_steps = 100
#
#for step in range(num_steps):
#    # Zero gradients from the previous step
#    optimizer.zero_grad()
#
#    # Compute the objective function
#    loss = objective_function(x)
#
#    # Backpropagate to compute gradients
#    loss.backward()
#
#    # Perform an optimization step
#    optimizer.step()
#
#    # Print the progress
#    if step % 10 == 0:  # Print every 10 steps
#        print(f"Step {step}, x = {x.item()}, loss = {loss.item()}")
#
#    # Gradient Check: Print gradient information every few epochs
#    if step % 10 == 0:
#        print(f"Gradients: {params.grad}")
#
## Final value of x
#print(f"Optimized x: {x.item()}")

