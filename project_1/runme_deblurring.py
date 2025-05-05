import numpy as np
from bm3d import bm3d
from runme_denoising import bilateral_filter,psnr,add_noise
import matplotlib.pyplot as plt
from tqdm import tqdm

def cconv2_by_fft2_numpy(A, B, flag_invertB=False, eta=1e-2):
    """
    Circular 2D convolution or deconvolution using FFT (NumPy version).
    
    Args:
        A (np.ndarray): 2D input image (H x W)
        B (np.ndarray): 2D kernel (h x w)
        flag_invertB (bool): If True, performs deconvolution
        eta (float): Regularization parameter for deconvolution
    
    Returns:
        np.ndarray: Output after convolution or deconvolution
    """
    m, n = A.shape
    mb, nb = B.shape

    # Pad kernel to image size
    bigB = np.zeros_like(A)
    bigB[:mb, :nb] = B

    # Roll to center the PSF
    bigB = np.roll(bigB, shift=(-mb // 2, -nb // 2), axis=(0, 1))

    # FFT of kernel and input
    fft2B = np.fft.fft2(bigB)
    fft2A = np.fft.fft2(A)

    if flag_invertB:
        # Tikhonov regularization for inverse filtering
        fft2B = np.conj(fft2B) / (np.abs(fft2B)**2 + eta)

    # Convolution or deconvolution
    result = np.real(np.fft.ifft2(fft2A * fft2B))

    return result

class PnPADMMDeBlurr:
    def __init__(self,denoiser,max_iter,rho,sigmas,kernel,reduce_rho=True,reduce_sigma=True,tol=1e-6):
        '''
        denosiser (string): denosing function
        max_iter (int): maximum ADMM itterations
        rho(float): ADMM penalty parameter
        sigmas (list): [sigma_psd] for bm3d denoiser and [sigma_r,sigma_s] for BF denosier
        '''
        self.kernel = kernel
        self.denoiser = denoiser
        self.max_iter = max_iter
        self.rho = rho
        self.sigmas = sigmas
        self.reduce_rho = reduce_rho
        self.reduce_sigma=reduce_sigma
        self.tol = tol

    def denoise_sample(self,y):
        if self.denoiser =='bm3d':
            return bm3d(y,sigma_psd=self.sigmas[0])
        else: #blf
            sigma_s = self.sigmas[0]
            sigma_r = self.sigmas[-1]
            return bilateral_filter(y,sigma_s,sigma_r)

    def __call__(self, y):
        '''
        y: blurred image (2D numpy array)
        Returns: Deblurred image
        '''
        # Initialization
        N = y.shape[0]*y.shape[1]
        z_k = y.copy()
        x_k = y.copy()
        u_k = np.zeros_like(y)
        rho_tmp = self.rho
        sigma_tmp = self.sigmas[0]

        i = 0
        res = 10
        pbar = tqdm(total=self.max_iter,desc='Residuals')
        while (res > self.tol) and i < self.max_iter:
            z_k_1, x_k_1, u_k_1 = self.pnp_admm_step(y, z_k, x_k, u_k)
            res_x = (1/np.sqrt(N)) * np.sqrt(np.sum((x_k_1-x_k)**2,axis=(0,1)))
            res_z = (1/np.sqrt(N)) * np.sqrt(np.sum((z_k_1-z_k)**2,axis=(0,1)))
            res_u = (1/np.sqrt(N)) * np.sqrt(np.sum((u_k_1-u_k)**2,axis=(0,1)))
            res_tmp = res_u+res_x+res_z
            if res_tmp>res:
                break
            res= res_tmp
            z_k=z_k_1
            x_k=x_k_1
            u_k=u_k_1
            i+=1
            pbar.update(1)
            pbar.set_description(f'Res={res:.8f}')
        self.rho = rho_tmp
        self.sigmas[0] = sigma_tmp
        return x_k
    
    #based on Tom's FFT2 function
    #https://github.com/tirer-lab/DDPG/blob/fcd17382d6b2d084b4bd2686c531b61e392cc1a9/functions/fft_operators.py#L222
    def z_update_fft2(self,y,x,u):
        m, n = y.shape[:2]
        mb, nb = self.kernel.shape[:2]

        # pad, multiply and transform back
        bigB = np.zeros((m, n))
        bigB[:mb,:nb] = self.kernel
        bigB = np.roll(bigB, (-int((mb-1)/2), -int((mb-1)/2)), axis=(0,1))  # pad PSF with zeros to whole image domain, and center it

        H = np.fft.fft2(bigB)

        f_k_f_y = np.fft.fft2(y) * np.conj(H)
        f_x_plus_u = np.fft.fft2(x-u)
        f_k_spec = np.abs(H)**2

        numerator = f_k_f_y + self.rho * f_x_plus_u
        denominator = f_k_spec + self.rho

        Zk = numerator/denominator
        zk = np.real(np.fft.ifft2(Zk))
        if self.reduce_rho:
            self.rho *=1.1
        if self.reduce_sigma:
            self.sigmas[0] *= 0.8
        return zk
    
    def pnp_admm_step(self, y, z, x, u):
        # --- z-update: data fidelity ---
        z = self.z_update_fft2(y,x,u)
        # --- x-update: denoising ---
        x = self.denoise_sample(z + u)
        # --- u-update: dual variable ---
        u = u + z - x
        return z, x, u
    
def main():
    """
    Denoise a set of grayscale images using bilateral filtering,
    compute and print PSNR values, and display visual comparisons.
    """
    image_names = ['1_Cameraman256', '2_house', '3_peppers256', '4_Lena512',
                   '5_barbara', '6_boat', '7_hill', '8_couple']
    # image_names = ['1_Cameraman256','2_house']
    #hyper parameters
    denoiser = 'bm3d'
    max_iter=4
    rho=0.055
    sigmas=[0.085]

    #blurring kernel
    
    i = np.arange(-7, 8)
    j = np.arange(-7, 8)
    ii, jj = np.meshgrid(i, j, indexing='ij')
    kernel = 1 / (1 + ii**2 + jj**2)
    kernel /= np.sum(kernel)

    input_psnrs = []
    denoised_psnrs = []
    images_gt = []
    images_noisy = []
    images_denoised = []

    dir_path = './test_set'
    deblurrer = PnPADMMDeBlurr(denoiser=denoiser,max_iter=max_iter,rho=rho,sigmas=sigmas,kernel=kernel)
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

        y = cconv2_by_fft2_numpy(img, kernel)
        y = add_noise(y,sigma_e=0.01)
        x_hat = deblurrer(y)

        psnr_input = psnr(img, y)
        psnr_output = psnr(img, x_hat)

        input_psnrs.append(psnr_input)
        denoised_psnrs.append(psnr_output)

        print(f"{name}: Input PSNR = {psnr_input:.2f}, Output PSNR = {psnr_output:.2f}")

        # Save for visualization
        images_gt.append(img)
        images_noisy.append(y)
        images_denoised.append(x_hat)

    input_psnr_mean =np.mean(input_psnrs)
    output_psnr_mean =np.mean(denoised_psnrs)

    print("\nAverage Input PSNR: {:.2f}".format(input_psnr_mean))
    print("Average Deblurred PSNR: {:.2f}".format(output_psnr_mean))

    if denoiser =='bm3d':
        sigma_txt = f'sigma={deblurrer.sigmas[0]:.4f}'
    else:
        sigma_txt = f'sigma_s={deblurrer.sigmas[0]:.4f},sigma_r={deblurrer.sigmas[-1]:.4f}'
    
    if deblurrer.reduce_sigma:
        sigma_txt = f'sigma_reduced'
    
    hyperparams = f'{sigma_txt},rho={rho if not deblurrer.reduce_rho else 'increased'}'
    # ===============================
    #  Plot: x_gt, y (noisy), x̂ (denoised)
    # ===============================
    fig, axs = plt.subplots(len(image_names), 3, figsize=(10, 2 * len(image_names)))
    fig.suptitle(f'Deblurring Results:PSNR-Input={input_psnr_mean:.2f},PSNR-Output={output_psnr_mean:.2f}\nHyperparams:{hyperparams}', fontsize=16)

    for row_idx, name in enumerate(image_names):
        img = images_gt[row_idx]  # Ground truth
        y = images_noisy[row_idx]  # Noisy image
        x_hat = images_denoised[row_idx]  # Denoised output
        for col_idx, image in enumerate([img, y, x_hat]):
            axs[row_idx, col_idx].imshow(image, cmap='gray', vmin=0, vmax=1)
            axs[row_idx, col_idx].axis('off')

            # Set column titles only on first row
            if row_idx == 0:
                if col_idx == 0:
                    axs[row_idx, col_idx].set_title("x_gt")
                elif col_idx == 1:
                    axs[row_idx, col_idx].set_title("y (Blurred)")
                elif col_idx ==2 :
                    axs[row_idx, col_idx].set_title("x̂ (Deblurred)")
                  
        # Add image name label on the left of each row
        axs[row_idx, 0].text(-0.1, 0.5, name, fontsize=10, va='center', ha='right',
                             transform=axs[row_idx, 0].transAxes, rotation=0)
    rho_fixed_txt = '_rho_fixed' if not deblurrer.reduce_rho else '_rho_reduced'
    sigma_fixed_txt = '_sigma_fixed' if not deblurrer.reduce_sigma else '_sigma_reduced'

    if deblurrer.denoiser!='bm3d':
        rho_fixed_txt=''
        sigma_fixed_txt=''
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'./plots/pnp_admm_results_max_{deblurrer.denoiser}{rho_fixed_txt}{sigma_fixed_txt}.png')
    plt.show()
# Run the main script
if __name__ == "__main__":
    main()