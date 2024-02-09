import numpy as np

class HOGFeatureExtractor:
    """
    nbins: number of bins that will be used
    unsigned: if True the sign of the angle is not considered
    """
    def __init__(self, nbins=9, unsigned=True):
        self.nbins = nbins
        self.unsigned = unsigned

    def _calc_gradient_for_channel(self, I):
        dx = np.diff(I, axis=0, append=I[-1:,:])
        dy = np.diff(I, axis=1, append=I[:,-1:])
        
        magnitude = np.sqrt(dx**2 + dy**2)
        if self.unsigned:
            angle = (np.arctan2(dy, dx) + np.pi) / (np.pi / self.nbins)
        else:
            angle = (np.arctan2(dy, dx) + 2 * np.pi) / (2 * np.pi / self.nbins)
        
        bin_pos = np.floor(angle).astype(int) % self.nbins
        r = angle - bin_pos
        histogram = np.zeros((I.shape[0] // 8, I.shape[1] // 8, self.nbins))

        for i in range(self.nbins):
            mask = bin_pos == i
            histogram += np.histogram2d(np.repeat(np.arange(I.shape[0]), I.shape[1])[mask.flatten()],
                                         np.tile(np.arange(I.shape[1]), I.shape[0])[mask.flatten()],
                                         bins=(I.shape[0] // 8, I.shape[1] // 8),
                                         weights=(r * magnitude)[mask])[0]

        ret = np.zeros((3, 3, self.nbins * 4))
        for i in range(3):
            for j in range(3):
                block = histogram[i:i+2, j:j+2, :].flatten()
                ret[i, j, :] = block / np.linalg.norm(block)

        return ret.flatten()

    def _calc_gradient_for_image(self, I):
        return np.concatenate([self._calc_gradient_for_channel(I[:, :, i]) for i in range(I.shape[2])])

    def predict(self, X):
        assert X.ndim == 4, "Input must be a 4D array."
        print("Extracting HOG features")
        return np.array([self._calc_gradient_for_image(X[i]) for i in range(X.shape[0])])