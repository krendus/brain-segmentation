import nibabel as nib
import numpy as np
import torch
import base64
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
import tempfile
import os
from io import BytesIO  # Import for base64 encoding of images
import torch.nn as nn
import torch.nn.functional as F
from flask import current_app
from io import BytesIO
import io
from monai.networks.nets import UNETR
from skimage.transform import resize
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(file):
    # Determine the file extension
    filename = file.filename
    ext = os.path.splitext(filename)[-1]  # Get the file extension (e.g., .nii, .nii.gz)

    # Define the path to the tmp directory
    tmp_dir = os.path.join(current_app.root_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)  # Ensure the tmp directory exists

    # Create a temporary file with the correct extension in the tmp directory
    with tempfile.NamedTemporaryFile(suffix=ext, dir=tmp_dir, delete=False) as tmp:
        file.save(tmp.name)
        temp_file_path = tmp.name

    try:
        # Load the image using nibabel
        image = nib.load(temp_file_path)
        image_data = image.get_fdata()

        # Check if the image has only one channel
        if image_data.ndim == 3:  # [D, H, W] -> Convert to [C, D, H, W]
            image_data = np.expand_dims(image_data, axis=0)  # Add channel dimension

        # Repeat the single channel to match the expected input channels (4 channels)
        input_data = np.repeat(image_data, 4, axis=0)  # Repeat along the channel axis

        # Convert the image data to a tensor
        input_tensor = torch.from_numpy(input_data).unsqueeze(0).float()  # Shape: [1, 4, D, H, W]

        # Resize the input tensor to the expected dimensions [1, 4, 128, 128, 128]
        input_tensor = F.interpolate(input_tensor, size=(128, 128, 128), mode='trilinear', align_corners=True)

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

    return input_tensor

def simulate_tumor_growth(shape=(240, 240, 160), num_iterations=100, diffusion_rate=0.1):
    # Initialize tumor tensor on CPU
    tumor = torch.zeros(shape, dtype=torch.float32, device='cpu')
    
    # Seed initial tumor at center
    tumor[shape[0]//2, shape[1]//2, shape[2]//2] = 1.0
    
    for _ in range(num_iterations):
        # Apply diffusion equation (simplified)
        tumor_new = tumor.clone()
        tumor_new[1:-1, 1:-1, 1:-1] += diffusion_rate * (
            tumor[:-2, 1:-1, 1:-1] + tumor[2:, 1:-1, 1:-1] +
            tumor[1:-1, :-2, 1:-1] + tumor[1:-1, 2:, 1:-1] +
            tumor[1:-1, 1:-1, :-2] + tumor[1:-1, 1:-1, 2:] -
            6 * tumor[1:-1, 1:-1, 1:-1]
        )
        
        # Clip values to [0, 1]
        tumor = torch.clamp(tumor_new, 0, 1)
    
    return tumor



def preprocess_images(zip_ref):
    """
    Preprocess multiple modalities from a zip file, resize them to (128, 128, 128),
    and return a 4-channel image tensor.
    
    Parameters:
    - zip_ref: The reference to the zip file containing the modalities (flair, t1ce, t1, t2).
    
    Returns:
    - image_tensor: A tensor containing the preprocessed 4-channel image.
    """
    tmp_dir = os.path.join(current_app.root_path, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Extract the zip file into the tmp directory
    zip_ref.extractall(tmp_dir)

    # List of required modalities
    required_modalities = ['flair', 't1ce', 't1', 't2']
    modality_files = {modality: None for modality in required_modalities}

    # Process each extracted file
    for file_name in os.listdir(tmp_dir):
        for modality in required_modalities:
            if file_name.endswith(f"_{modality}.nii") or file_name.endswith(f"_{modality}.nii.gz"):
                modality_files[modality] = os.path.join(tmp_dir, file_name)

    # Check if all required modalities are present
    if not all(modality_files.values()):
        raise ValueError('ZIP file must contain all four modalities: flair, t1ce, t1, t2')

    images = []
    for modality, file_path in modality_files.items():
        try:
            # Load the image using nibabel
            img = nib.load(file_path)
            img_data = img.get_fdata()

            # Normalize the image
            img_data = (img_data - np.mean(img_data)) / np.std(img_data)

            # Resize the image to (128, 128, 128)
            img_data_resized = resize(img_data, (128, 128, 128), anti_aliasing=True)

            images.append(img_data_resized.astype(np.float32))

        except nib.filebasedimages.ImageFileError as e:
            print(f"Error loading image for modality {modality}: {e}")
            raise ValueError(f"Failed to load NIfTI image for modality '{modality}'. Please check the file format.")

    # Ensure all modalities have the same dimensions after resizing
    shapes = [img.shape for img in images]
    if len(set(shapes)) != 1:
        raise ValueError("All modalities must have the same dimensions after resizing.")

    # Stack the 4 modalities into a single tensor (4-channel image)
    image_stack = np.stack(images, axis=0)  # Stack along the channel axis

    # Convert the stacked image into a PyTorch tensor and add the batch dimension
    image_tensor = torch.tensor(image_stack, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    return image_tensor



def inference(input_tensor):
    try:
        # Load the model
        model = load_model()
        logger.debug("Model loaded successfully")

        # Determine the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Using device: %s", device)

        # Move the input tensor to the appropriate device
        input_tensor = input_tensor.to(device)
        logger.debug("Input tensor moved to device")
        logger.debug("Input tensor shape: %s", input_tensor.shape)
        # Run inference
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output_tensor = model(input_tensor)
            logger.debug("Inference completed")

        return output_tensor

    except Exception as e:
        logger.exception("An error occurred during inference")
        raise e  # Re-raise the exception after logging

def visualize_results(image_tensor, output_tensor, filename="result_image.png"):
    # Convert tensors to NumPy arrays
    image_np = image_tensor[0].cpu().numpy()  # Convert input image to NumPy array
    output_np = output_tensor[0].cpu().numpy()  # Convert output to NumPy array

    # Determine the number of slices along the depth (D) dimension
    depth = image_np.shape[1]  # Assuming [C, D, H, W] format
    slice_idx = depth // 2  # Middle slice for visualization

    # Extract ET, TC, WT from the output tensor
    et_mask = output_np[0]  # Enhancing Tumor (ET)
    tc_mask = output_np[1]  # Tumor Core (TC)
    wt_mask = output_np[2]  # Whole Tumor (WT)

    # Create subplots to visualize input, output, and overlay side by side
    fig, axs = plt.subplots(4, 4, figsize=(24, 24))

    # Plot input image channels
    for i in range(4):
        if i < image_np.shape[0]:  # Ensure within bounds
            axs[0, i].imshow(image_np[i, slice_idx, :, :], cmap="gray")
            axs[0, i].set_title(f"Input Channel {i+1}")
        else:
            axs[0, i].axis('off')  # Hide unused subplots

    # Plot output segmentation channels
    for i in range(4):
        if i < output_np.shape[0]:  # Ensure within bounds
            axs[1, i].imshow(output_np[i, slice_idx, :, :], cmap="gray")
            axs[1, i].set_title(f"Output Channel {i+1}")
        else:
            axs[1, i].axis('off')  # Hide unused subplots

    # Create overlay by combining input and output images
    for i in range(4):
        if i < image_np.shape[0] and i < output_np.shape[0]:  # Ensure within bounds
            overlay = np.copy(image_np[i, slice_idx, :, :])
            axs[2, i].imshow(overlay, cmap="gray", alpha=0.7)  # Input in grayscale
            axs[2, i].imshow(output_np[i, slice_idx, :, :], cmap="jet", alpha=0.3)  # Output in color
            axs[2, i].set_title(f"Overlay Channel {i+1}")
        else:
            axs[2, i].axis('off')  # Hide unused subplots

    # Plot overlays for each tumor class (ET, TC, WT)
    tumor_classes = [("Enhancing Tumor (ET)", et_mask), 
                     ("Tumor Core (TC)", tc_mask), 
                     ("Whole Tumor (WT)", wt_mask)]

    for j, (title, mask) in enumerate(tumor_classes):
        axs[3, j].imshow(image_np[0, slice_idx, :, :], cmap="gray", alpha=0.7)  # Input in grayscale
        axs[3, j].imshow(mask[slice_idx, :, :], cmap="jet", alpha=0.3)  # Tumor class in color
        axs[3, j].set_title(f"{title} Overlay")

    axs[3, 3].axis('off')  # Hide unused subplot

    # Define the path for the uploads directory
    uploads_dir = os.path.join(current_app.root_path, 'static/uploads')  # 'static/uploads'
    os.makedirs(uploads_dir, exist_ok=True)  # Ensure the uploads directory exists

    # Save the figure to the uploads folder
    filepath = os.path.join(uploads_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)

    return 'static/uploads/' + filename





def visualize_prediction(inputs, prediction, selected_slices=None):
    """
    Visualize the input modalities and the model's prediction.

    Parameters:
    - inputs: A 4D tensor or numpy array of shape (4, H, W, D), where each channel is a different modality.
    - prediction: The model's output, assumed to be a 4D tensor or numpy array of shape (C, H, W, D) 
                  where C is the number of classes.
    - selected_slices: A list of slice indices to visualize. If None, all slices are shown.
    """

    # Convert tensors to numpy arrays if necessary
    if torch.is_tensor(inputs):
        inputs = inputs.cpu().numpy()
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()

    # Remove batch dimension if present
    inputs = inputs[0] if inputs.ndim == 5 else inputs
    prediction = prediction[0] if prediction.ndim == 5 else prediction

    selected_slices = [10, 30, 50, 70]

    # If selected slices are not provided, visualize all slices
    if selected_slices is None:
        selected_slices = range(inputs.shape[3])

    # Adjust the number of rows to accommodate all inputs and class predictions + overlay
    total_rows = 4 + prediction.shape[0] + 1  # 4 for input modalities, C for predictions, 1 for overlay
    fig, axes = plt.subplots(total_rows, len(selected_slices), figsize=(15, 8))

    # Plot input modalities (FLAIR, T1CE, T1, T2)
    for i, modality in enumerate(["FLAIR", "T1CE", "T1", "T2"]):
        for j, slice_idx in enumerate(selected_slices):
            axes[i, j].imshow(inputs[i, :, :, slice_idx], cmap='gray')
            axes[i, j].set_title(f"{modality} Slice {slice_idx}", fontsize=8)
            axes[i, j].axis('off')

    # Plot model predictions for each class (ET, WT, TC)
    classes = ["ET", "WT", "TC"]
    for class_idx in range(prediction.shape[0]):
        for j, slice_idx in enumerate(selected_slices):
            axes[4 + class_idx, j].imshow(prediction[class_idx, :, :, slice_idx], cmap='hot', alpha=0.5)
            axes[4 + class_idx, j].set_title(f"Prediction {classes[class_idx]} Slice {slice_idx}", fontsize=8)
            axes[4 + class_idx, j].axis('off')

    # Plot overlayed input and prediction
    for j, slice_idx in enumerate(selected_slices):
        overlay = np.zeros_like(inputs[0, :, :, slice_idx])
        for class_idx in range(prediction.shape[0]):
            overlay += prediction[class_idx, :, :, slice_idx] * (class_idx + 1)

        axes[4 + prediction.shape[0], j].imshow(inputs[0, :, :, slice_idx], cmap='gray')
        axes[4 + prediction.shape[0], j].imshow(overlay, cmap='jet', alpha=0.3)
        axes[4 + prediction.shape[0], j].set_title(f"Overlay Slice {slice_idx}", fontsize=8)
        axes[4 + prediction.shape[0], j].axis('off')

    plt.tight_layout(pad=2.0)

    # Define the path for the uploads directory
    uploads_dir = os.path.join(current_app.root_path, 'static/uploads')  
    os.makedirs(uploads_dir, exist_ok=True)  

    # Save the figure to the uploads folder
    filename = "filename.png"
    filepath = os.path.join(uploads_dir, filename)
    plt.savefig(filepath)
    plt.close(fig)

    return 'static/uploads/' + filename


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
# Safely loading the model state dict
    model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=False), strict=False)

    model.eval()  # Set the model to evaluation mode
    return model
