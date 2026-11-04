import numpy as np

def bw_correction(sample: np.ndarray, dark: np.ndarray, white: np.ndarray):
    bw_corrected = (sample - dark) /   (white - dark + np.finfo(np.float64).eps) #
    bw_corrected = bw_corrected.clip(0, 1)
    return bw_corrected


def eval_target_distr(led_wavelengths_np, spectrums_wavelength_gt):
    # ==========================================
    # STEP 2: Formation of ideal rectangular distributions (W)
    # ==========================================
    # The interval boundaries lie exactly halfway between adjacent LEDs.
    bounds = np.zeros(len(led_wavelengths_np) + 1)
    bounds[1:-1] = (led_wavelengths_np[:-1] + led_wavelengths_np[1:]) / 2.0
    # We are extending the outer limits to the target boundaries of the range.
    bounds[0] = spectrums_wavelength_gt[0]
    bounds[-1] = spectrums_wavelength_gt[-1]

    # We create a matrix W of size (N x 37).
    W = np.zeros((len(spectrums_wavelength_gt), len(led_wavelengths_np)))
    W_avg = np.zeros_like(W)  # Matrix for computing the MEAN value inside an interval

    for i, lam in enumerate(spectrums_wavelength_gt):
        # We find the index of the interval into which the given wavelength falls.
        if lam > 350 and lam < 1050:
            idx = np.searchsorted(bounds, lam) - 1
            idx = np.clip(idx, 0, len(led_wavelengths_np) - 1)
            W[i, idx] = 1.0

    # We normalize W_avg so that it yields the arithmetic mean over the interval.
    for j in range(len(led_wavelengths_np)):
        mask = W[:, j] == 1.0
        count = np.sum(mask)
        if count > 0:
            W_avg[mask, j] = 1.0 / count
    return W_avg



def ridge_regression_lstsq(X, y, alpha, rcond=None):
    """
    Solves Ridge Regression using data augmentation with np.linalg.lstsq
    alpha: Regularization strength (lambda)
    """
    # Number of features
    n_features = X.shape[1]

    # Square root of alpha because lstsq minimizes the squared loss
    penalty_matrix = np.sqrt(alpha) * np.identity(n_features)
    zero_vector = np.zeros_like(penalty_matrix)

    # Augment X and y
    X_augmented = np.vstack([X, penalty_matrix])
    y_augmented = np.vstack([y, zero_vector])

    # Solve regularized system
    # rcond=None uses the modern machine precision threshold
    weights, residuals, rank, s = np.linalg.lstsq(X_augmented, y_augmented, rcond=rcond)

    return weights, residuals, rank, s


def data_prep(spectrums_gt, spectrums_cam_cor):
    # ==========================================
    # STEP 1: Data preparation
    # ==========================================
    R_cal = spectrums_gt.reshape(-1, *spectrums_gt.shape[2:])
    Y_cal = spectrums_cam_cor.reshape(-1, *spectrums_cam_cor.shape[2:])
    return R_cal, Y_cal


def reconstruct_spectrum(M, data_cube):
    Y_cube = data_cube.copy()
    C, H, W = Y_cube.shape
    Y_cube_chlast = np.moveaxis(Y_cube, 0, -1)
    Y_cube_2d = Y_cube_chlast.reshape(-1, C)
    Y_ideal_2d = Y_cube_2d @ M.astype(np.float32)
    Y_ideal_2d = np.clip(Y_ideal_2d, 0, None)
    Y_ideal_chlast = Y_ideal_2d.reshape(H, W, C)
    Y_corrected = np.moveaxis(Y_ideal_chlast, -1, 0)
    return Y_corrected