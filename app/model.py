import torch
import torch.nn.functional as F
import torch.nn as nn
from monai.networks.nets import UNETR

class UNETRWithPhysics(nn.Module):
    def __init__(self, unetr_model, reaction_rate, diffusion_rate, time_step, device):
        super(UNETRWithPhysics, self).__init__()
        self.unetr = unetr_model
        self.reaction_rate = reaction_rate
        self.diffusion_rate = diffusion_rate
        self.time_step = time_step
        self.device = device

        # Define the 3D Laplacian kernel for diffusion (5x5x5 kernel for 3D space)
        # Ensure the kernel has the correct dimensions
        kernel_size = (5, 5, 5)
        self.laplacian_kernel = torch.tensor([[[[0, 0, 1, 0, 0],
                                               [0, 1, 2, 1, 0],
                                               [1, 2, -24, 2, 1],
                                               [0, 1, 2, 1, 0],
                                               [0, 0, 1, 0, 0]]]], dtype=torch.float32).to(device)

        # Ensure kernel has the correct dimensions
        self.laplacian_kernel = self.laplacian_kernel.unsqueeze(0)  # Add out_channels dimension
        self.laplacian_kernel = self.laplacian_kernel.repeat(1, 1, *kernel_size)  # Repeat for in_channels

    def compute_laplacian(self, segmentation):
        """
        Compute the Laplacian of the segmentation using a 3D convolution to model diffusion.
        The Laplacian is applied separately to each channel.
        """
        batch_size, channels, depth, height, width = segmentation.shape

        # Expand the Laplacian kernel to match the number of channels
        laplacian_kernel = self.laplacian_kernel.repeat(channels, 1, 1, 1, 1)

        # Compute the padding based on the kernel size (to maintain input size)
        kernel_depth, kernel_height, kernel_width = self.laplacian_kernel.shape[2:5]
        padding = (
            (kernel_depth - 1) // 2,   # Padding for depth
            (kernel_height - 1) // 2,  # Padding for height
            (kernel_width - 1) // 2    # Padding for width
        )

        # Perform 3D convolution with appropriate padding
        laplacian = F.conv3d(segmentation, laplacian_kernel, padding=padding, groups=channels)

        # Print shape of the result for debugging
        #         print(f"Laplacian shape: {laplacian.shape}")

        return laplacian



    def reaction_diffusion_step(self, segmentation):
        # Compute Laplacian for diffusion term
        laplacian = self.compute_laplacian(segmentation)

        # Reaction term: Logistic growth (reaction_rate * segmentation * (1 - segmentation))
        reaction_term = self.reaction_rate * segmentation * (1 - segmentation)

        # Diffusion term: (diffusion_rate * Laplacian)
        diffusion_term = self.diffusion_rate * laplacian

        # Ensure dimensions match
        if reaction_term.shape != diffusion_term.shape:
            raise ValueError(f"Dimension mismatch: reaction_term shape {reaction_term.shape}, diffusion_term shape {diffusion_term.shape}")

        # Update the segmentation with both reaction and diffusion dynamics
        updated_segmentation = segmentation + self.time_step * (reaction_term + diffusion_term)
        
        return updated_segmentation

    def forward(self, x):
        # Forward pass through the UNETR model for initial segmentation
        segmentation = self.unetr(x)

        # Apply the reaction-diffusion step to the segmentation
        segmentation = self.reaction_diffusion_step(segmentation)

        return segmentation

def load_model():
    roi = (128, 128, 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reaction_rate = 0.1
    diffusion_rate = 0.01
    time_step = 0.01
    unetr_model = UNETR(
        in_channels=4,            # Number of input channels (e.g., 4 for BraTS)
        out_channels=3,           # Number of output channels or classes
        img_size=roi,             # Image size for 3D input
        feature_size=16,          # Size of network features
        hidden_size=768,          # Hidden size for transformer layers
        mlp_dim=3072,             # MLP dimensionality in transformer layers
        num_heads=12,             # Number of self-attention heads
        proj_type="conv",         # Replace pos_embed with proj_type; use 'conv' or 'perceptron'
        norm_name="instance",     # Normalization type
        dropout_rate=0.0,         # Dropout rate
    ).to(device)
    model = UNETRWithPhysics(unetr_model, reaction_rate, diffusion_rate, time_step, device).to(device)
    # Load your trained model here
    model.load_state_dict(torch.load("model.pth", map_location=device), strict=False)
    model.eval()  # Set the model to evaluation mode
    return model
