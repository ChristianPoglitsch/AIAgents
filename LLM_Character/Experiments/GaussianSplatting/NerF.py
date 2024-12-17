import numpy as np
import pyvista as pv
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class NeRF(nn.Module):
    def __init__(self, input_dim=3, direction_dim=3, hidden_dim=256):
        super(NeRF, self).__init__()
        # MLP for density and intermediate features
        self.density_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # Output: (N, hidden_dim)
            nn.Softplus()  # Constrain density to be non-negative
        )
        # MLP for color
        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_dim + direction_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # Output: (R, G, B)
            nn.Sigmoid()  # Constrain colors to [0, 1]
        )

    def forward(self, x, d):
        """
        Forward pass for the NeRF model.
        Args:
            x: 3D coordinates (N, 3)
            d: Viewing directions (N, 3)
        Returns:
            density: Volume density (N, 1)
            color: RGB color (N, 3)
        """
        # Predict density and intermediate features
        features = self.density_mlp(x)  # Shape: (N, hidden_dim)
        density = features[:, :1]  # Extract density (N, 1)

        # Concatenate features and viewing directions
        color_input = torch.cat([features, d], dim=-1)  # Shape: (N, hidden_dim + 3)
        color = self.color_mlp(color_input)  # Shape: (N, 3)
        return density, color
    

def generate_rays_with_projection(world_matrix, projection_matrix, image_size):
    """
    Generate rays cast from the camera origin through every pixel on the view plane,
    incorporating the projection matrix.

    Args:
    - world_matrix (torch.Tensor): 4x4 matrix defining the camera pose in world space.
    - projection_matrix (torch.Tensor): 4x4 matrix defining the camera projection.
    - image_size (tuple): (height, width) of the image.

    Returns:
    - rays_o (torch.Tensor): Origins of the rays (N, 3).
    - rays_d (torch.Tensor): Directions of the rays (N, 3), pointing through each pixel.
    """
    H, W = image_size  # Image dimensions
    aspect_ratio = W / H

    # Step 1: Create pixel coordinates in NDC
    i, j = torch.meshgrid(
        torch.linspace(-1, 1, W),  # NDC x-coordinates
        torch.linspace(-1, 1 / aspect_ratio, H),  # NDC y-coordinates
        indexing="ij"
    )
    i, j = i.flatten(), j.flatten()  # Flatten the grid
    ndc_points = torch.stack([i, j, torch.ones_like(i), torch.ones_like(i)], dim=-1)  # (N, 4)

    # Step 2: Transform from NDC to camera space using the inverse projection matrix
    inv_proj = torch.linalg.inv(projection_matrix)  # Invert the projection matrix
    points_camera = (ndc_points @ inv_proj.T)  # Transform to camera space
    points_camera = points_camera[:, :3] / points_camera[:, 3:]  # Convert from homogeneous to Euclidean coordinates

    # Step 3: Transform from camera space to world space using the world matrix
    R = world_matrix[:3, :3]  # Rotation
    t = world_matrix[:3, 3]   # Translation (camera position)
    points_world = (points_camera @ R.T) + t  # Transform to world space

    # Step 4: Compute ray origins and directions
    rays_o = t.expand(points_world.shape)  # Ray origins (camera position repeated)
    rays_d = points_world - rays_o  # Ray directions (from origin to each pixel)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)  # Normalize directions

    return rays_o, rays_d


# Volume rendering function
def volume_rendering(rays_o, rays_d, nerf_model, n_samples=64, near=0.1, far=4.0):
    """
    Perform volume rendering for a batch of rays.
    Args:
        rays_o: Ray origins (B, 3)
        rays_d: Ray directions (B, 3)
        nerf_model: NeRF model
        n_samples: Number of samples per ray
        near: Near plane
        far: Far plane
    Returns:
        rendered_colors: Rendered pixel colors (B, 3)
    """
    B = rays_o.shape[0]
    t_vals = torch.linspace(near, far, n_samples).to(rays_o.device)
    z_vals = t_vals.expand(B, n_samples)  # Sampled depths
    points = rays_o[:, None, :] + z_vals[..., None] * rays_d[:, None, :]  # Points along each ray
    points = points.reshape(-1, 3)  # Flatten points for batch processing
    directions = rays_d.repeat_interleave(n_samples, dim=0)  # Match points
    
    # Predict density and color
    density, color = nerf_model(points, directions)
    density = density.view(B, n_samples)
    color = color.view(B, n_samples, 3)
    
    # Compute weights for volume rendering
    delta = z_vals[:, 1:] - z_vals[:, :-1]
    delta = torch.cat([delta, torch.tensor([1e10]).expand(delta[:, :1].shape).to(delta.device)], dim=-1)  # Last interval
    alpha = 1 - torch.exp(-density * delta)  # Opacity
    T = torch.cumprod(1 - alpha + 1e-10, dim=-1)  # Accumulated transmittance
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=-1)  # Add initial transmittance of 1
    weights = T * alpha
    
    # Compute final color
    rendered_colors = torch.sum(weights[..., None] * color, dim=1)
    
    #print(f"Density: {density.min().item()}, {density.max().item()}")
    #print(f"Color: {color.min().item()}, {color.max().item()}")

    return rendered_colors
            

def render_image(nerf_model, rays_o, rays_d, image_size=(64, 64), n_samples=64, batch_size=1024, near=1.0, far=4.0):
    """
    Render an image using a NeRF model.
    
    Args:
    - nerf_model (torch.nn.Module): Trained NeRF model.
        rays_o: Ray origins (B, 3)
        rays_d: Ray directions (B, 3)
    - image_size (tuple): (height, width) of the output image.
    - n_samples (int): Number of samples per ray.
    - batch_size (int): The batch size for processing rays.
    - near (float): Near plane distance.
    - far (float): Far plane distance.

    Returns:
    - Rendered image as a torch.Tensor of shape (H, W, 3).
    """
    H, W = image_size

    # Debug: Check ray origins and directions
    #print(f"Ray origins (first 5): {rays_o[:5]}")
    #print(f"Ray directions (first 5): {rays_d[:5]}")

    # Step 1: Render colors for all rays
    num_rays = rays_o.shape[0]
    rendered_colors = torch.zeros((num_rays, 3), dtype=torch.float32)

    # Process rays in batches
    for i in range(0, num_rays, batch_size):
        rays_o_batch = rays_o[i:i + batch_size]
        rays_d_batch = rays_d[i:i + batch_size]
        
        # Debug: Check batch inputs
        #print(f"Processing batch {i} to {i + batch_size}")

        # Perform volume rendering
        rendered_batch = volume_rendering(
            rays_o_batch, rays_d_batch, nerf_model, n_samples, near, far
        )
        
        # Debug: Check rendered colors
        #print(f"Rendered batch colors (first 5): {rendered_batch[:5]}")
        rendered_colors[i:i + batch_size] = rendered_batch

    # Step 2: Reshape the flat output into the desired image shape
    rendered_image = rendered_colors.reshape(H, W, 3)  # (H, W, 3)

    # Debug: Check the final image
    #print(f"Rendered image min: {rendered_image.min()}, max: {rendered_image.max()}")
    return rendered_image


# Define the optimization loop
def train_nerf(nerf_model, rays_os, rays_ds, target_images, n_samples=64, near=0.1, far=4.0, batch_size=1024, num_epochs=1000, learning_rate=1e-4):
    """
    Train the NeRF model on an image of size 64x64 pixels.
    
    Args:
    - nerf_model (torch.nn.Module): The NeRF model.
    - rays_o (torch.Tensor): Origins of the rays (N, 3).
    - rays_d (torch.Tensor): Directions of the rays (N, 3).
    - target_image (torch.Tensor): Target image (64x64, 3).
    - n_samples (int): Number of samples per ray.
    - near (float): Near plane for ray sampling.
    - far (float): Far plane for ray sampling.
    - batch_size (int): The batch size for processing rays.
    - num_epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate for the optimizer.
    """
    
    # Number of rays is the same as the number of pixels in the image
    N = rays_os[0].shape[0]
    
    # Initialize the optimizer for the NeRF model
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        nerf_model.train()
        optimizer.zero_grad()
        total_loss = 0

        for i, (rays_o, rays_d, target_image) in enumerate(zip(rays_os, rays_ds, target_images)):

            # Create a transformed (cloned) copy of target_image
            transformed_target_image = target_image.clone()  # Make a deep copy of target_image
            transformed_target_image = transformed_target_image.view(N, 3)  # Reshape it to (N, 3) where N = 64*64
            
            # Initialize a tensor to hold the rendered image
            target_image = torch.zeros_like(transformed_target_image)
            target_image = target_image.to(device)

            # Process rays in batches to avoid memory overload
            for i in range(0, N, batch_size):
                # Get the current batch of rays
                rays_o_batch = rays_o[i:i + batch_size]
                rays_d_batch = rays_d[i:i + batch_size]
            
                # Perform volume rendering for the current batch
                rendered_batch = volume_rendering(rays_o_batch, rays_d_batch, nerf_model, n_samples, near, far)
            
                # Store the rendered batch in the appropriate place in the full image
                target_image[i:i + batch_size] = rendered_batch

            # Compute the loss (e.g., Mean Squared Error)
            loss = torch.nn.functional.mse_loss(target_image, transformed_target_image)
            total_loss += loss
        
        # Backpropagate and optimize (only once per epoch)
        total_loss.backward()  # No need for retain_graph=True here
        optimizer.step()

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")



# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a toy NeRF model
nerf_model = NeRF()
nerf_model.to(device)

# Define a world matrix and projection matrix (as before)
world_matrix = torch.tensor([
    [1.0, 0.0, 0, 4],  # 
    [0.0, 1.0, 0, 1],  # 
    [0, 0, 1, 2],      # Camera is 2 units in front of the scene
    [0, 0, 0, 1]
], dtype=torch.float32)

projection_matrix = torch.eye(4)  # Identity projection for simplicity

# Image resolution
image_size = (64, 64)  # 64x64 pixels

# Render the image using the NeRF model
#rendered_image = render_image(nerf_model, world_matrix, projection_matrix, image_size)
#rendered_image_np = rendered_image.detach().cpu().numpy()
#rendered_image = torch.from_numpy(rendered_image_np)

image_size = (64, 64)
image = Image.new("RGB", image_size, color=(255, 255, 255))  # White background
draw = ImageDraw.Draw(image)

# Cube corners in 2D (simplified perspective projection of a cube)
cube_points = [
    (20, 20),  # Front top-left
    (40, 20),  # Front top-right
    (40, 40),  # Front bottom-right
    (20, 40),  # Front bottom-left
    (25, 25),  # Back top-left (offset for perspective)
    (45, 25),  # Back top-right (offset for perspective)
    (45, 45),  # Back bottom-right (offset for perspective)
    (25, 45),  # Back bottom-left (offset for perspective)
]

# Draw the edges of the cube by connecting the points
# Front square
draw.line([cube_points[0], cube_points[1]], fill="red", width=2)
draw.line([cube_points[1], cube_points[2]], fill="green", width=2)
draw.line([cube_points[2], cube_points[3]], fill="blue", width=2)
draw.line([cube_points[3], cube_points[0]], fill="black", width=2)

# Back square
draw.line([cube_points[4], cube_points[5]], fill="black", width=2)
draw.line([cube_points[5], cube_points[6]], fill="black", width=2)
draw.line([cube_points[6], cube_points[7]], fill="black", width=2)
draw.line([cube_points[7], cube_points[4]], fill="black", width=2)

# Connect the front and back squares to form the cube
draw.line([cube_points[0], cube_points[4]], fill="blue", width=2)
draw.line([cube_points[1], cube_points[5]], fill="black", width=2)
draw.line([cube_points[2], cube_points[6]], fill="green", width=2)
draw.line([cube_points[3], cube_points[7]], fill="black", width=2)

# Convert the image to a NumPy array
rendered_image = np.array(image)
rendered_image = torch.tensor(rendered_image, dtype=torch.float32)
rendered_image /= 255.0  # Scale to [0, 1]
rendered_image = rendered_image.to(device)

n_samples = 24
rays_o, rays_d = generate_rays_with_projection(world_matrix, projection_matrix, image_size)
rays_o = rays_o.to(device)
rays_d = rays_d.to(device)
train_nerf(nerf_model, [rays_o], [rays_d], [rendered_image], num_epochs=3500, n_samples=n_samples, batch_size=1024*8)

# Visualize the rendered image
optimized_image = render_image(nerf_model=nerf_model, rays_o=rays_o, rays_d=rays_d, batch_size=1024*8, image_size=image_size, n_samples=n_samples)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("View 1: Ground Truth")
plt.imshow(rendered_image.cpu().detach().numpy(), cmap="hot", origin="lower")
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



# Other view
world_matrix = torch.tensor([
    [1.0, 0.0, 0, 4.5],  # 
    [0.0, 1.0, 0, 0.5],  # 
    [0, 0, 1, 2],      # Camera is 2 units in front of the scene
    [0, 0, 0, 1]
], dtype=torch.float32)
rays_o, rays_d = generate_rays_with_projection(world_matrix, projection_matrix, image_size)
rays_o = rays_o.to(device)
rays_d = rays_d.to(device)

optimized_image_new = render_image(nerf_model=nerf_model, rays_o=rays_o, rays_d=rays_d, batch_size=1024*8, image_size=image_size, n_samples=n_samples)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("View 1: Rotated optimized")
plt.imshow(optimized_image_new.cpu().detach().numpy(), cmap="hot", origin="lower")
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
