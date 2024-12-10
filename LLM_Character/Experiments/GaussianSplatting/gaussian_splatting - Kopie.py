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

def generate_3d_gaussian_volume(grid_size, points, sigma):
    """
    Creates a 3D volume by adding Gaussian splats for each point using PyTorch tensors.
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

    # Initialize a 3D volume
    volume = torch.zeros((x_size, y_size, z_size), dtype=torch.float32)

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
            loss = torch.nn.functional.mse_loss(rendered_image, input_image.to(device))
            total_loss += loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Logging
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.item()}")

    return initial_gaussians

# --- --- --- --- ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
# Example 3D points in local space
points = [
    (55, 55, 2, 1.0),  # Initial points
    (105, 105, 2, 0.6),
    (40, 100, 2, 0.6)
]

# Function to dynamically add more points
def add_points(new_points):
    global points
    points.extend(new_points)  # Add new points to the existing list

points = torch.tensor(points, dtype=torch.float32)

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
sigma = 5  # Spread of the Gaussian

# Generate the 3D Gaussian volume with updated points
volume = generate_3d_gaussian_volume(grid_size, points, sigma)

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


input_images = [rendered_image_1, rendered_image_2]
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

