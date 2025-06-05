import numpy as np
from bm3d import bm3d
import matplotlib.pyplot as plt
from tqdm import tqdm
from ircnn import IRCNN
import torch

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
def cconv2_invAAt_by_fft2_numpy(A,B,eta=0.01):
    # assumes that A (2D image) is bigger than B (2D kernel)

    m, n = A.shape
    mb, nb = B.shape

    # Pad kernel to image size
    bigB = np.zeros_like(A)
    bigB[:mb, :nb] = B

    # Roll to center the PSF
    bigB = np.roll(bigB, shift=(-mb // 2, -nb // 2), axis=(0, 1))

    # FFT of kernel and input
    fft2B = np.fft.fft2(bigB)

    fft2B_norm2 = np.abs(fft2B)**2
    inv_fft2B_norm = 1 / (fft2B_norm2 + eta)

    result = np.real(np.fft.ifft2(np.fft.fft2(A) * inv_fft2B_norm))

    return result



# y_k = AtA_add_eta_inv(At(y) + rho(x -u))

def cconv2_by_fft2_numpy(A, B,flag_conjB=False, eta=1e-2):
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

    if flag_conjB:
        # Tikhonov regularization for inverse filtering
        fft2B = np.conj(fft2B)# / (np.abs(fft2B)**2 + eta)

    result = np.real(np.fft.ifft2(fft2A * fft2B))

    return result

class PnPADMMDeBlurr:
    def __init__(self,denoiser,max_iter,rho,sigmas,kernel,gamma=1,eta=1,tol=1e-6):
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
        self.gamma = gamma
        self.eta=eta
        self.tol = tol
        if denoiser == 'ircnn':
            # self.model = IRCNN(in_nc=1,out_nc=1,nc=64)
            self.model25 = torch.load('/home/workspace/yoavellinson/signals_and_images_course/project_2/ircnn_gray.pth')
            # current_idx = min(int(np.ceil(sigmas[0] * 255. / 2.) - 1),24)
            # former_idx = 0
            # if current_idx != former_idx:
            #     self.model.load_state_dict(model25[str(current_idx)], strict=True)
            #     self.model.eval()
            #     for _, v in self.model.named_parameters():
            #         v.requires_grad = False
            # #     model = model.to(device)
            # # self.model.load_state_dict(model25,strict=True)
    def get_txt(self):
        return f'rho_{self.rho}_sigma_{self.sigmas},eta_{self.eta}_gamma_{self.gamma}'
    
    def denoise_sample(self,y):
        if self.denoiser =='bm3d':
            return bm3d(y,sigma_psd=self.sigmas[0])
        elif self.denoiser =='ircnn':
            model = IRCNN(in_nc=1,out_nc=1,nc=64)
            current_idx = min(int(np.ceil(self.sigmas[0] * 255. / 2.) - 1),24)
            former_idx = 0
            if current_idx != former_idx:
                model.load_state_dict(self.model25[str(current_idx)], strict=True)
                model.eval()
                for _, v in model.named_parameters():
                    v.requires_grad = False
            y = torch.tensor(y,dtype=torch.float).unsqueeze(0).unsqueeze(0)
            return model(y).detach().cpu().squeeze().numpy()
        
    def AtA_add_eta_inv_numpy(self, vec,):
        I = vec.reshape(vec.shape[0], vec.shape[1])
        
        out = cconv2_invAAt_by_fft2_numpy(I, self.kernel, eta=self.rho)

        return out.reshape(vec.shape[0], -1)

    def At_numpy(self,vec):
        I = vec.reshape(vec.shape[0], vec.shape[1])
        out = cconv2_by_fft2_numpy(I,self.kernel, flag_conjB=True)
        return out.reshape(vec.shape[0], -1)
    
    def __call__(self, y,img):
        '''
        y: blurred image (2D numpy array)
        Returns: Deblurred image
        '''
        # Initialization
        N = y.shape[0]*y.shape[1]
        #as in the paper

        x_k = y.copy()
        v_k = y.copy()
        u_k = np.zeros_like(y)

        rho_tmp = self.rho
        sigma_tmp = self.sigmas[0]

        i = 0
        res = 10
        pbar = tqdm(total=self.max_iter,desc='Residuals',leave=False)
        psnr_old = 0
        while (res > self.tol) and i < self.max_iter:
            x_k_1,v_k_1,u_k_1 = self.pnp_admm_step(y,x_k,v_k,u_k)
            psnr_mid = psnr(x_k_1,img)
            # print(f'PSNR:{psnr_mid}')
            delta_psnr = psnr_mid - psnr_old
            res_x = (1/np.sqrt(N)) * np.sqrt(np.sum((x_k_1-x_k)**2,axis=(0,1)))
            res_z = (1/np.sqrt(N)) * np.sqrt(np.sum((v_k_1-v_k)**2,axis=(0,1)))
            res_u = (1/np.sqrt(N)) * np.sqrt(np.sum((u_k_1-u_k)**2,axis=(0,1)))

            res = res_u+res_x+res_z
            # if res < 0 and i>10: 
            #     break
            if delta_psnr >0:
                self.rho *= self.gamma
            elif delta_psnr<0 and i > 5:
                break
            v_k=v_k_1
            x_k=x_k_1
            u_k=u_k_1
            psnr_old = psnr_mid
            # self.rho*=1.5
            self.sigmas[0] *=self.eta
            i+=1
            pbar.update(1)
            pbar.set_description(f'PSNR={psnr_mid:.8f}')

        self.rho = rho_tmp
        self.sigmas[0] = sigma_tmp
        return x_k
    
    def prox(self,y,x_tilde):
        a = self.At_numpy(y) + self.rho*(x_tilde)
        return self.AtA_add_eta_inv_numpy(a)
    
    def pnp_admm_step(self,y,x,v,u):
        x_tilde = v-u

        x = self.prox(y,x_tilde)

        v_tilde = x+u

        v= self.denoise_sample(v_tilde)

        u += x-v

        return x,v,u
    
def main(denoiser='bm3d'):
    """
    Denoise a set of grayscale images using bilateral filtering,
    compute and print PSNR values, and display visual comparisons.
    """
    image_names = ['1_Cameraman256', '2_house', '3_peppers256', '4_Lena512',
                   '5_barbara', '6_boat', '7_hill', '8_couple']
    #hyper parameters

    max_iter=35
    sigmas = [0.09] if denoiser =='bm3d' else [0.11]
    rho = 0.013 if denoiser =='bm3d' else 0.008

    eta = 0.99 if denoiser =='bm3d' else 0.99
    gamma=1.01 if denoiser =='bm3d' else 1.01

    #blurring kernel
    
    i = np.arange(-7, 8)
    j = np.arange(-7, 8)
    kernel = np.zeros((len(i),len(j)))
    for ii in range(len(i)):
        for jj in range(len(j)):
            kernel[ii,jj] = 1/(1+i[ii]**2+j[jj]**2)
    kernel /= np.sum(kernel)

    input_psnrs = []
    denoised_psnrs = []
    images_gt = []
    images_noisy = []
    images_denoised = []

    dir_path = './test_set'
    deblurrer = PnPADMMDeBlurr(denoiser=denoiser,max_iter=max_iter,rho=rho,sigmas=sigmas,kernel=kernel,eta=eta,gamma=gamma,tol=1e-5)
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
        x_hat = deblurrer(y,img)

        psnr_input = psnr(y, img)
        psnr_output = psnr(x_hat,img)

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

    if denoiser =='bm3d' or denoiser=='ircnn':
        sigma_txt = f'sigma={deblurrer.sigmas[0]:.4f}'
    else:
        sigma_txt = f'sigma_s={deblurrer.sigmas[0]:.4f},sigma_r={deblurrer.sigmas[-1]:.4f}'
    
   
    
    hyperparams = f'{sigma_txt},rho={rho},eta={eta},gamma={gamma}'
    # ===============================
    #  Plot: x_gt, y (noisy), x̂ (denoised)
    # ===============================
    fig, axs = plt.subplots(len(image_names), 3, figsize=(10, 2 * len(image_names)))
    fig.suptitle(f'Deblurring Results:Denoiser:{denoiser}\nPSNR-Input={input_psnr_mean:.2f},PSNR-Output={output_psnr_mean:.2f}\nHyperparams:{hyperparams}', fontsize=16)

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

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f'./plots/pnp_admm_results_max_{deblurrer.get_txt()}.png'
    plt.savefig(filename)
    print(f'Saved: {filename}')
    plt.show()

from itertools import product

def run_grid_search():
    sigmas_list_s = [0.1,0.08,0.06]
    # sigmas_list_r = [0.01,0.04,0.08, 0.1,0.5]

    rhos_list = [0.009,0.01,0.1]

    eta_list = [1,0.99,0.999,0.95,0.9]

    gamma_list = [1]

    best_psnr = -np.inf
    best_config = None

    for sigma_s, rho, eta, gamma in tqdm(product(sigmas_list_s, rhos_list, eta_list, gamma_list),total=len(sigmas_list_s)*len(rhos_list)*len(eta_list)*len(gamma_list)):
        print(f"\nRunning: sigma={sigma_s}, rho={rho}, reduce_sigma={eta}, increase_rho={gamma}")
        denoiser = 'ircnn'
        max_iter = 25
        kernel = make_kernel()

        deblurrer = PnPADMMDeBlurr(
            denoiser=denoiser,
            max_iter=max_iter,
            rho=rho,
            sigmas=[sigma_s],
            kernel=kernel,
            gamma=gamma,
            eta=eta
        )

        avg_psnr = evaluate_deblurrer(deblurrer)
        print(f'PSNR:{avg_psnr}')
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_config = (sigma_s, rho, eta, gamma)

    print("\n==== Best Configuration ====")
    print(f"Sigma: {best_config[0]}, Rho: {best_config[1]}, Reduce Sigma: {best_config[2]}, Increase Rho: {best_config[3]}")
    print(f"Average PSNR: {best_psnr:.2f}")

def make_kernel():
    i = np.arange(-7, 8)
    j = np.arange(-7, 8)
    kernel = np.zeros((len(i), len(j)))
    for ii in range(len(i)):
        for jj in range(len(j)):
            kernel[ii, jj] = 1 / (1 + i[ii] ** 2 + j[jj] ** 2)
    return kernel / np.sum(kernel)

def evaluate_deblurrer(deblurrer):
    image_names = ['1_Cameraman256','5_barbara','3_peppers256']
    dir_path = './test_set'
    input_psnrs = []
    denoised_psnrs = []

    for name in image_names:
        try:
            img = plt.imread(f'{dir_path}/{name}.png')
        except Exception as e:
            print(f"Failed to load {name}: {e}")
            return -1

        if img.ndim == 3:
            img = np.mean(img, axis=2)
        if img.dtype != np.float32 and img.max() > 1.0:
            img = img.astype(np.float32) / 255.0

        y = cconv2_by_fft2_numpy(img, deblurrer.kernel)
        y = add_noise(y, sigma_e=0.01)
        x_hat = deblurrer(y,img)

        psnr_output = psnr(img, x_hat)
        denoised_psnrs.append(psnr_output)

    return np.mean(denoised_psnrs)


if __name__ == "__main__":
    # main(denoiser='BL') #for bilateral filter denoiser
    main(denoiser='ircnn')
    # run_grid_search()
