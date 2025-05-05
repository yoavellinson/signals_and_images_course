import numpy as np
import matplotlib.pyplot as plt
from bm3d import bm3d

np.random.seed(0)

def psnr(x, x_gt):
    mse = np.mean((x - x_gt) ** 2)
    return 10 * np.log10(1.0 / mse)

def create_mask(shape, percent):
    """
    Randomly sample a binary mask with `percent` percent of pixels observed.
    """
    total_pixels = np.prod(shape)
    num_samples = int(total_pixels * percent)
    mask = np.zeros(total_pixels, dtype=bool)
    mask[np.random.choice(total_pixels, num_samples, replace=False)] = True
    return mask.reshape(shape)

def inpainting_admm_pnp(y, mask, denoiser, rho=0.8, beta=0.01, num_iters=25):
    """
    ADMM-PnP for image inpainting using a given denoiser.
    Inputs:
        y        : Observed image with only known pixels in mask.
        mask     : Boolean mask of known pixels.
        denoiser : Function to apply as plug-and-play denoiser.
        rho      : ADMM penalty parameter.
        beta     : Denoiser strength parameter.
        num_iters: Number of ADMM iterations.
    Output:
        x_hat    : Reconstructed image.
    """
    x = np.copy(y)
    z = np.copy(x)
    u = np.zeros_like(x)

    for k in range(num_iters):
        # x-update via denoising
        x = denoiser(z - u, sigma_psd=np.sqrt(beta / rho))

        # z-update (proximal step)
        z_new = np.where(mask,
                         (y + rho * (x + u)) / (1 + rho),
                         x + u)

        # u-update (dual update)
        u = u + x - z_new
        z = z_new

    return x

def main():
    image_names = ['1_Cameraman256', '2_house', '3_peppers256', '4_Lena512',
                   '5_barbara', '6_boat', '7_hill', '8_couple']
    observed_fraction = 0.2
    rho = 0.2
    beta = 0.006
    num_iters = 25

    input_psnrs = []
    recon_psnrs = []
    images_gt = []
    images_observed = []
    images_reconstructed = []

    for name in image_names:
        try:
            img = plt.imread(f'{name}.png')
        except FileNotFoundError:
            print(f"Error: File {name}.png not found.")
            return -1

        if img.ndim == 3:
            img = np.mean(img, axis=2)
        if img.dtype != np.float32 and img.max() > 1.0:
            img = img.astype(np.float32) / 255.0

        mask = create_mask(img.shape, observed_fraction)
        y = img * mask  # observed pixels only

        x_hat = inpainting_admm_pnp(y, mask, bm3d, rho, beta, num_iters)

        psnr_input = psnr(img, y)
        psnr_output = psnr(img, x_hat)

        input_psnrs.append(psnr_input)
        recon_psnrs.append(psnr_output)
        images_gt.append(img)
        images_observed.append(y)
        images_reconstructed.append(x_hat)

        print(f"{name}: Input PSNR = {psnr_input:.2f}, Reconstructed PSNR = {psnr_output:.2f}")

    print("\nAverage Input PSNR: {:.2f}".format(np.mean(input_psnrs)))
    print("Average Reconstructed PSNR: {:.2f}".format(np.mean(recon_psnrs)))

    # Display images
    fig, axs = plt.subplots(len(image_names), 3, figsize=(10, 2 * len(image_names)))
    fig.suptitle('Inpainting Results: x_gt, y, x̂', fontsize=16)

    for row_idx, name in enumerate(image_names):
        for col_idx, image in enumerate([images_gt[row_idx], images_observed[row_idx], images_reconstructed[row_idx]]):
            axs[row_idx, col_idx].imshow(image, cmap='gray', vmin=0, vmax=1)
            axs[row_idx, col_idx].axis('off')
            if row_idx == 0:
                axs[row_idx, col_idx].set_title(["x_gt", "y (observed)", "x̂ (reconstructed)"][col_idx])

        axs[row_idx, 0].text(-0.1, 0.5, name, fontsize=10, va='center', ha='right',
                             transform=axs[row_idx, 0].transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()
