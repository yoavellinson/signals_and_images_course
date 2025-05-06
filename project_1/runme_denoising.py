import numpy as np
import matplotlib.pyplot as plt
from bm3d import bm3d

### ============================
###        Functions
### ============================

np.random.seed(0)

def psnr(x, x_gt):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Inputs:
        x     : Estimated image (numpy array, float32, values in [0, 1])
        x_gt  : Ground truth image (same size and type as x)

    Output:
        PSNR value in decibels (float)
    """
    mse = np.mean((x - x_gt) ** 2)
    return 10 * np.log10(1 / mse)

def add_noise(img, sigma_e):
    """
    Add Gaussian noise to an image with fixed random seed for reproducibility.

    Inputs:
        img     : Clean image (numpy array, float32, values in [0, 1])
        sigma_e : Standard deviation of Gaussian noise

    Output:
        Noisy image y = x_gt + e (numpy array, float32)
    """
    noise = np.random.normal(0, sigma_e, img.shape)
    return img + noise

def Compute_spatial_weights(half_w, sigma_s):
    """
    Precompute the spatial (distance-based) Gaussian weights for the bilateral filter.

    Inputs:
        half_w  : Half of the filter window size (int)
        sigma_s : Standard deviation of the spatial Gaussian kernel (float)

    Output:
        spatial_weights : 2D numpy array of shape (2*half_w+1, 2*half_w+1)
                          containing spatial Gaussian weights centered at (0,0)
    """
    x = np.arange(-half_w, half_w + 1)
    y = np.arange(-half_w, half_w + 1)
    X, Y = np.meshgrid(x, y)
    spatial_weights = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma_s ** 2))
    return spatial_weights

def bilateral_filter(img, sigma_s, sigma_r):
    """
    Apply bilateral filtering to a grayscale image.

    Inputs:
        img      : Noisy grayscale image (numpy array, float32, values in [0, 1])
        sigma_s  : Standard deviation for spatial kernel
        sigma_r  : Standard deviation for range kernel

    Output:
        Denoised image (numpy array, float32, same shape as img)
    """
    H, W = img.shape
    half_w = int(3 * sigma_s)  # Size of the window is 6*sigma_s + 1
    padded_img = np.pad(img, half_w, mode='reflect')
    output = np.zeros_like(img)

    # Compute spatial weights (Gaussian based on distance)
    spatial_weights = Compute_spatial_weights(half_w, sigma_s)

    # Iterate over each pixel
    for i in range(H):
        for j in range(W):
            i1 = i + half_w
            j1 = j + half_w

            # Extract local patch
            patch = padded_img[i1 - half_w:i1 + half_w + 1, j1 - half_w:j1 + half_w + 1]

            # Compute range weights (Gaussian based on intensity difference)
            intensity_diff = patch - padded_img[i1, j1]
            range_weights = np.exp(-(intensity_diff ** 2) / (2 * sigma_r ** 2))

            # Combine weights
            weights = spatial_weights * range_weights
            weights /= np.sum(weights) # Normalizing

            # Apply weighted average
            output[i, j] = np.sum(patch * weights)

    return output

### ============================
###            Main
### ============================

def main():
    """
    Denoise a set of grayscale images using bilateral filtering,
    compute and print PSNR values, and display visual comparisons.
    """
    image_names = ['1_Cameraman256', '2_house', '3_peppers256', '4_Lena512',
                   '5_barbara', '6_boat', '7_hill', '8_couple']
    sigma_e = 0.1  # Noise standard deviation
    sigma_s = 1.7    # Spatial kernel std (tuning)
    sigma_r = 0.26  # Range kernel std (tuning)

    input_psnrs = []
    denoised_psnrs = []
    bm3d_psnrs = []
    # Store results for visualization later
    images_gt = []
    images_noisy = []
    images_denoised = []
    images_bm3d_denoised = []

    dir_path = './test_set'
    for name in image_names:
        try:
            img = plt.imread(f'{dir_path}/{name}.png')
        except FileNotFoundError:
            print(f"Error: File {dir_path}+/{name}.png not found.")
            return -1
        except Exception as e:
            print(f"Could not load {name}.png due to error: {e}")
            return -1

        # Convert to grayscale and normalize
        if img.ndim == 3:
            img = np.mean(img, axis=2)  # convert RGB to grayscale
        if img.dtype != np.float32 and img.max() > 1.0:
            img = img.astype(np.float32) / 255.0  # normalize to [0, 1]

        y = add_noise(img, sigma_e)
        x_hat = bilateral_filter(y, sigma_s, sigma_r)
        x_bm3d_hat = bm3d(y, sigma_psd=sigma_e)

        psnr_input = psnr(img, y)
        psnr_output = psnr(img, x_hat)
        psnr_bm3d_output= psnr(img, x_bm3d_hat)

        input_psnrs.append(psnr_input)
        denoised_psnrs.append(psnr_output)
        bm3d_psnrs.append(psnr_bm3d_output)

        print(f"{name}: Input PSNR = {psnr_input:.2f}, Bilateral Output PSNR = {psnr_output:.2f}, Output BM3D PSNR = {psnr_bm3d_output:.2f}")

        # Save for visualization
        images_gt.append(img)
        images_noisy.append(y)
        images_denoised.append(x_hat)
        images_bm3d_denoised.append(x_bm3d_hat)

    print("\nAverage Input PSNR: {:.2f}".format(np.mean(input_psnrs)))
    print("Average Bilateral Output PSNR: {:.2f}".format(np.mean(denoised_psnrs)))
    print("Average BM3D Output PSNR: {:.2f}".format(np.mean(bm3d_psnrs)))

    # ===============================
    #  Plot: x_gt, y (noisy), x̂ (denoised)
    # ===============================
    fig, axs = plt.subplots(len(image_names), 4, figsize=(10, 2 * len(image_names)))
    fig.suptitle('Denoising Results: x_gt, y, x̂,x̂_BM3D', fontsize=16)

    for row_idx, name in enumerate(image_names):
        img = images_gt[row_idx]  # Ground truth
        y = images_noisy[row_idx]  # Noisy image
        x_hat = images_denoised[row_idx]  # Denoised output
        x_bm3d_hat = images_bm3d_denoised[row_idx]
        for col_idx, image in enumerate([img, y, x_hat,x_bm3d_hat]):
            axs[row_idx, col_idx].imshow(image, cmap='gray', vmin=0, vmax=1)
            axs[row_idx, col_idx].axis('off')

            # Set column titles only on first row
            if row_idx == 0:
                if col_idx == 0:
                    axs[row_idx, col_idx].set_title("x_gt")
                elif col_idx == 1:
                    axs[row_idx, col_idx].set_title("y (noisy)")
                elif col_idx ==2 :
                    axs[row_idx, col_idx].set_title("x̂ (denoised)")
                else: 
                    axs[row_idx, col_idx].set_title("x̂_BM3D (denoised)")
                    

        # Add image name label on the left of each row
        axs[row_idx, 0].text(-0.1, 0.5, name, fontsize=10, va='center', ha='right',
                             transform=axs[row_idx, 0].transAxes, rotation=0)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'./plots/denoising_results.png')

    plt.show()



# Run the main script
if __name__ == "__main__":
    main()
