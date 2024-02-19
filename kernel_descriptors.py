import numpy as np
from tqdm import tqdm
from scipy import linalg
from global_features import KernelPCA
from kernels import GaussianKernel, GaussianKernelForAngle

class FeatureVectorProjection:
    def __init__(self, kernel):
        self.kernel = kernel
        self.basis = None
        self.G = None
        self.ndim = None

    def fit(self, X):
        K = self.kernel.build_K(X)
        K_inv = linalg.inv(K)
        self.G = linalg.cholesky(K_inv, lower=True).real
        self.basis = X
        self.ndim = X.shape[0]

    def predict(self, X):
        K = self.kernel.build_K(X, self.basis)
        return K @ self.G

class KernelDescriptorsExtractor:
    def __init__(self, gamma_o=5, gamma_c=4, gamma_b=2, gamma_p=3,
                 grid_o_dim=25, grid_c_dims=(5, 5, 5), grid_p_dims=(5, 5),
                 epsilon_g=0.8, epsilon_s=0.2):
        self.projector_o = self.init_orientation_basis(gamma_o, grid_o_dim)
        self.projector_c = self.init_color_basis(gamma_c, grid_c_dims)
        self.projector_b = self.init_binary_pattern_basis(gamma_b)
        self.projector_p = self.init_position_basis(gamma_p, grid_p_dims)
        self.epsilon_g = epsilon_g
        self.epsilon_s = epsilon_s
        self.kpca_op = self.init_kpca(self.projector_o, self.projector_p, GaussianKernel(0.4))
        self.kpca_cp = self.init_kpca(self.projector_c, self.projector_p, GaussianKernel(0.4))

    def init_basis(self, kernel, X):
        projector = FeatureVectorProjection(kernel)
        projector.fit(X)
        return projector

    def init_orientation_basis(self, gamma, grid_dim):
        print("Initializing basis for orientation")
        kernel = GaussianKernelForAngle(1 / np.sqrt(2 * gamma))
        X = np.linspace(-np.pi, np.pi, grid_dim + 1)[:-1][:, np.newaxis]
        return self.init_basis(kernel, X)

    def init_color_basis(self, gamma, grid_dims):
        print("Initializing basis for color")
        kernel = GaussianKernel(1 / np.sqrt(2 * gamma))
        steps = [1.0 / (dim - 1) for dim in grid_dims]
        X = np.mgrid[0:1 + steps[0]:steps[0], 0:1 + steps[1]:steps[1], 0:1 + steps[2]:steps[2]].reshape(3, -1).T
        return self.init_basis(kernel, X)

    def init_binary_pattern_basis(self, gamma):
        print("Initializing basis for binary patterns")
        kernel = GaussianKernel(1 / np.sqrt(2 * gamma))
        X = np.mgrid[[slice(0, 2, 1) for _ in range(8)]].reshape(8, -1).T
        return self.init_basis(kernel, X)

    def init_position_basis(self, gamma, grid_dims):
        print("Initializing basis for positions")
        kernel = GaussianKernel(1 / np.sqrt(2 * gamma))
        steps = [1.0 / (dim - 1) for dim in grid_dims]
        X = np.mgrid[0:1 + steps[0]:steps[0], 0:1 + steps[1]:steps[1]].reshape(2, -1).T
        return self.init_basis(kernel, X)

    def init_kpca(self, projector1, projector2, kernel):
        X1 = projector1.predict(projector1.basis)
        X2 = projector2.predict(projector2.basis)
        kdes_dim = X1.shape[1] * X2.shape[1]
        X_kpca = np.array([np.kron(x, y) for x in X1 for y in X2]).reshape(-1, kdes_dim)
        kpca = KernelPCA(kernel)
        kpca.fit(X_kpca)
        return kpca
    
    def _calc_gradient_match_kernel_for_image(self, I, patch_size, subsample):
        gradients = np.gradient(I, axis=(0, 1))
        Ig_magnitude = np.sqrt(sum(gradients[i] ** 2 for i in range(len(gradients))))
        Ig_angle = np.arctan2(gradients[0], gradients[1])

        X_p = self._generate_patch_positions(patch_size)
        ret = self._compute_feature_vectors(Ig_magnitude, Ig_angle, X_p, patch_size, subsample, self.projector_o)

        return self.kpca_op.predict(ret, components=200).flatten()

    def _calc_color_match_kernel_for_image(self, I, patch_size, subsample):
        X_p = self._generate_patch_positions(patch_size)
        ret = self._compute_feature_vectors_color(I, X_p, patch_size, subsample, self.projector_c)

        return self.kpca_cp.predict(ret, components=200).flatten()

    def _generate_patch_positions(self, patch_size):
        x_step = 1.0 / (patch_size[0] - 1)
        y_step = 1.0 / (patch_size[1] - 1)
        return np.mgrid[0:1 + x_step:x_step, 0:1 + y_step:y_step].reshape(2, -1).T

    def _compute_feature_vectors(self, magnitude, angle, X_p, patch_size, subsample, projector):
        kdes_dims = projector.ndim * self.projector_p.ndim
        ret = []
        for sx in range(0, magnitude.shape[0] - patch_size[0] + 1, subsample[0]):
            for sy in range(0, magnitude.shape[1] - patch_size[1] + 1, subsample[1]):
                patch_mag = magnitude[sx:sx + patch_size[0], sy:sy + patch_size[1]].ravel()
                patch_angle = angle[sx:sx + patch_size[0], sy:sy + patch_size[1]].ravel()[:, np.newaxis]
                norm = np.sqrt(self.epsilon_g + np.sum(patch_mag ** 2))

                X_o = projector.predict(patch_angle)
                aux = np.zeros(kdes_dims)
                for x_o, x_p, mag in zip(X_o, X_p, patch_mag):
                    aux += mag * np.kron(x_o, x_p)
                ret.append(aux / norm)
        return np.array(ret)

    def _compute_feature_vectors_color(self, I, X_p, patch_size, subsample, projector):
        kdes_dims = projector.ndim * self.projector_p.ndim
        ret = []
        for sx in range(0, I.shape[0] - patch_size[0] + 1, subsample[0]):
            for sy in range(0, I.shape[1] - patch_size[1] + 1, subsample[1]):
                patch = I[sx:sx + patch_size[0], sy:sy + patch_size[1]].reshape(-1, I.shape[2])
                X_c = projector.predict(patch)
                aux = np.zeros(kdes_dims)
                for x_c, x_p in zip(X_c, X_p):
                    aux += np.kron(x_c, x_p)
                ret.append(aux)
        return np.array(ret)

    def predict(self, X, patch_size=(16, 16), subsample=(8, 8), match_kernel='gradient'):
        assert X.ndim == 4, "Input must be a 4D array."
        results = []
        for i in tqdm(range(X.shape[0])):
            if match_kernel == 'gradient':
                result = self._calc_gradient_match_kernel_for_image(X[i], patch_size, subsample)
            elif match_kernel == 'color':
                result = self._calc_color_match_kernel_for_image(X[i], patch_size, subsample)
            else:
                raise ValueError("Unknown match kernel: {}".format(match_kernel))
            results.append(result)
        return np.array(results)