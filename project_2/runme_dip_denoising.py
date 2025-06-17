### Libraries ###
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from models import *
from utils.denoising_utils import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

### Hyper Parameters ###
num_iter = 4000  # Max iterations for training (ensure it's long enough to see overfitting)
OPTIMIZER = 'adam'

# Loss Function Flag
USE_SURE_LOSS = True  # Set to True for SURE, False for MSE
# --- Adjust LR and reg_noise_std based on current loss type ---
if USE_SURE_LOSS:
    current_LR = 0.001
    current_reg_noise_std = 1.0
    current_loss_type_name = "SURE"
else:
    current_LR = 0.01
    current_reg_noise_std = 1. / 30.
    current_loss_type_name = "MSE"

### Parameters ###
imsize = -1
PLOT = False  # We will handle plotting separately at the end
sigma = 0.1  # standard deviation of Gaussian noise
show_every = 100  # Print progress every X iterations
exp_weight = 0.99  # For averaging output
input_depth = 1
OPT_OVER = 'net'
INPUT = 'noise'
pad = 'reflection'

# List of test images
image_files = ['1_Cameraman256.png', '2_house.png', '3_peppers256.png', '4_Lena512.png',
               '5_barbara.png', '6_boat.png', '7_hill.png', '8_couple.png']
image_files = [os.path.join('test_set/', f) for f in image_files]

### Settings ###
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.cuda.FloatTensor
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    dtype = torch.FloatTensor
    print("Using CPU")

# Visualization Specific Settings
VISUAL_EXAMPLES_IMAGE_FNAME = '1_Cameraman256.png'  # The image to show detailed visual examples for

# Dictionary to store *all* iteration results for the selected image
# Format: {iteration: {'out_np': ..., 'psnr': ...}}
full_history_for_selected_image = {}

# This will be populated after training to select specific points
dynamic_visual_iterations_global = []

# Store PSNRs for all images to find average peak
psnrs_per_iter_all_images = []

### Main Code ###
print(f"\n--- Starting Training with {current_loss_type_name} Loss ---")
print(f"Loss: {current_loss_type_name}, LR: {current_LR}, reg_noise_std: {current_reg_noise_std}")

for fname in image_files:
    print(f"\nProcessing {os.path.basename(fname)} with {current_loss_type_name} Loss")

    # Fix random seed for reproducibility per image
    np.random.seed(0)
    torch.manual_seed(0)

    # Load GT image
    img_pil = crop_image(get_image(fname, imsize)[0].convert('L'), d=32)
    img_np = pil_to_np(img_pil)

    # Add Gaussian noise
    noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
    img_noisy_np = img_np + noise
    img_noisy_np = np.clip(img_noisy_np, 0, 1)

    # Convert to torch
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

    # Initialize DIP network
    net = get_net(input_depth, 'skip', pad,
                  skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5,
                  upsample_mode='bilinear',
                  n_channels=1).type(dtype)

    net_input_saved = None
    noise_input = None

    if USE_SURE_LOSS:
        net_input_saved = img_noisy_torch.detach().clone()
        noise_input = torch.zeros_like(img_noisy_torch).type(dtype)
    else:
        net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
        net_input_saved = net_input.detach().clone()
        noise_input = net_input.detach().clone()

    mse_loss_fn = torch.nn.MSELoss().type(dtype)

    out_avg_wrapper = [None]
    psnrs_wrapper = []


    def closure():
        # Zero gradients for network parameters
        for param in net.parameters():
            if param.grad is not None:
                param.grad.zero_()

        current_net_input = None

        if USE_SURE_LOSS:
            # When using SURE, net_input is based on the noisy image.
            current_net_input = net_input_saved.clone()
            if current_reg_noise_std > 0:
                current_net_input += (torch.randn_like(noise_input) * current_reg_noise_std)

            current_net_input.requires_grad_(True)

            if current_net_input.grad is not None:
                current_net_input.grad.zero_()

            out = net(current_net_input)

            # SURE Loss: ||out - noisy_image||^2 + 2 * sigma^2 * div(out) - sigma^2 * N

            # Find the divergence
            divergence_term = torch.autograd.grad(out, current_net_input,
                                                  grad_outputs=torch.ones_like(out),
                                                  retain_graph=True, allow_unused=True)[0]
            divergence_term_sum = divergence_term.sum()

            # SURE Loss
            loss = torch.norm(out - img_noisy_torch) ** 2 + \
                   2 * (sigma ** 2) * divergence_term_sum - \
                   (sigma ** 2) * img_noisy_torch.numel()

        else:  # Use MSE Loss
            if current_reg_noise_std > 0:
                current_net_input = net_input_saved + (noise_input.normal_() * current_reg_noise_std)
            else:
                current_net_input = net_input_saved

            out = net(current_net_input)
            loss = mse_loss_fn(out, img_noisy_torch)

        loss.backward()

        if out_avg_wrapper[0] is None:
            out_avg_wrapper[0] = out.detach()
        else:
            out_avg_wrapper[0] = out_avg_wrapper[0] * exp_weight + out.detach() * (1 - exp_weight)

        out_np = torch_to_np(out.detach())
        psnr_gt = compare_psnr(img_np, out_np)
        psnrs_wrapper.append(psnr_gt)

        # Store full history for the selected image
        current_iter_num = len(psnrs_wrapper)
        if os.path.basename(fname) == VISUAL_EXAMPLES_IMAGE_FNAME:
            full_history_for_selected_image[current_iter_num] = {
                'out_np': out_np.copy(),
                'psnr': psnr_gt
            }

        # Iterations status
        if len(psnrs_wrapper) % show_every == 0:
            print(f"Iter {len(psnrs_wrapper):04d} | PSNR_GT: {psnr_gt:.2f} | Loss: {loss.item():.4f}")

        return loss


    p = get_params(OPT_OVER, net, net_input_saved if not USE_SURE_LOSS else None)
    optimize(OPTIMIZER, p, closure, current_LR, num_iter)
    psnrs_per_iter_all_images.append(psnrs_wrapper)

# Final Analysis
min_len_all_images = min(len(p) for p in psnrs_per_iter_all_images)
psnrs_per_iter_all_images = [p[:min_len_all_images] for p in psnrs_per_iter_all_images]
avg_psnr_across_all_images = np.mean(psnrs_per_iter_all_images, axis=0)

max_avg_psnr_iter_idx = np.argmax(avg_psnr_across_all_images)
max_avg_psnr_value = avg_psnr_across_all_images[max_avg_psnr_iter_idx]

print(f"\n--- {current_loss_type_name} Loss Summary (All Images) ---")
print(f"Maximum average PSNR: {max_avg_psnr_value:.2f} dB (achieved at iteration {max_avg_psnr_iter_idx + 1}).")
print("PSNR for each image at this peak iteration:")

# Plotting
for i, fname in enumerate(image_files):
    image_name = os.path.basename(fname)
    psnr_at_max_avg = psnrs_per_iter_all_images[i][max_avg_psnr_iter_idx]
    print(f"- {image_name}: {psnr_at_max_avg:.2f} dB")

plt.figure(figsize=(8, 5))
plt.plot(avg_psnr_across_all_images)
plt.xlabel("Iteration")
plt.ylabel(f"Average PSNR (dB) with {current_loss_type_name} Loss")
plt.title(f"DIP Denoising: Average PSNR vs Iteration ({current_loss_type_name} Loss)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"dip_avg_psnr_vs_iteration_{current_loss_type_name.lower()}.png")
plt.show()

# Determine specific visualization iterations
peak_iter = max_avg_psnr_iter_idx + 1

dynamic_visual_iterations_global = []

# 1. The first iteration (1-indexed)
if 1 <= num_iter:
    dynamic_visual_iterations_global.append(1)

# 2. Iteration 1000, if available
if 1000 <= num_iter:
    dynamic_visual_iterations_global.append(1000)
else:  # If num_iter is less than 1000 but more than 1, add a middle point as a fallback
    if num_iter > 1 and (num_iter // 2) not in dynamic_visual_iterations_global:
        dynamic_visual_iterations_global.append(num_iter // 2)

# 3. The best iteration (peak_iter)
if peak_iter not in dynamic_visual_iterations_global and peak_iter <= num_iter:
    dynamic_visual_iterations_global.append(peak_iter)

# 4. The last iteration
if num_iter not in dynamic_visual_iterations_global:
    dynamic_visual_iterations_global.append(num_iter)

# Ensure unique and sorted iterations
dynamic_visual_iterations_global = sorted(list(set(dynamic_visual_iterations_global)))

# Filter out iterations for which we do not have data (e.g., if num_iter is very small or iterations were skipped)
dynamic_visual_iterations_global = sorted([
    iter_val for iter_val in dynamic_visual_iterations_global
    if iter_val in full_history_for_selected_image
])

visual_data_for_selected_image = {
    iter_val: full_history_for_selected_image[iter_val]
    for iter_val in dynamic_visual_iterations_global
}

print(f"\nSelected iterations for visual examples: {dynamic_visual_iterations_global}")


# Plot Visual Examples for the selected image
def plot_visual_examples(gt_img_np, noisy_img_np, visual_data, iterations_to_plot, image_title, loss_type):
    num_iterations = len(iterations_to_plot)
    if num_iterations == 0:
        print("No iterations to plot for visual examples.")
        return

    fig, axes = plt.subplots(num_iterations, 3, figsize=(12, 4 * num_iterations))

    # Ensure axes is always 2D for consistent indexing
    if num_iterations == 1:
        axes = np.array([axes])

    for i, iteration in enumerate(iterations_to_plot):
        # GT Image - Squeeze the channel dimension (1) to make it (H, W)
        axes[i, 0].imshow(gt_img_np.squeeze(), cmap='gray')
        axes[i, 0].set_title(f"GT\n(Iter {iteration})")
        axes[i, 0].axis('off')

        # Noisy Image - Squeeze the channel dimension (1) to make it (H, W)
        axes[i, 1].imshow(noisy_img_np.squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Noisy (PSNR: {compare_psnr(gt_img_np.squeeze(), noisy_img_np.squeeze()):.2f})")
        axes[i, 1].axis('off')

        # DIP Estimated Image - Squeeze the channel dimension (1) to make it (H, W)
        estimated_data = visual_data.get(iteration)
        if estimated_data:
            axes[i, 2].imshow(estimated_data['out_np'].squeeze(), cmap='gray')
            axes[i, 2].set_title(f"DIP ({loss_type})\nPSNR: {estimated_data['psnr']:.2f}")
        else:
            axes[i, 2].set_title(f"DIP ({loss_type})\nData missing for {iteration}")
        axes[i, 2].axis('off')

    plt.suptitle(f"Denoising Examples for {image_title} ({loss_type} Loss)", y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f"denoising_examples_{os.path.splitext(image_title)[0]}_{loss_type.lower()}.png")
    plt.show()


# Prepare data for visualization and call plotting function
print(f"\n--- Preparing Visual Examples for {VISUAL_EXAMPLES_IMAGE_FNAME} ({current_loss_type_name} Loss) ---")
# Load GT and Noisy image for the selected example only once for plotting
gt_img_pil = crop_image(get_image(os.path.join('../test_Set/', VISUAL_EXAMPLES_IMAGE_FNAME), imsize)[0].convert('L'),
                        d=32)
gt_img_np = pil_to_np(gt_img_pil)

# Re-add noise just for the noisy image display, ensuring it's consistent for the plot
np.random.seed(0)
temp_noise = np.random.normal(0, sigma, gt_img_np.shape).astype(np.float32)
noisy_img_for_display_np = np.clip(gt_img_np + temp_noise, 0, 1)

plot_visual_examples(gt_img_np, noisy_img_for_display_np, visual_data_for_selected_image,
                     dynamic_visual_iterations_global, VISUAL_EXAMPLES_IMAGE_FNAME, current_loss_type_name)