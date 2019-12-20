from glob import glob
from utils import computePSNR, psf2otf
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


class fastMAPDeblurer:
    def __init__(self, rho, sigma, image_shape, kernel_shape, denoiser='./models/map_color/optimizedMAPdenoiser.pb'):
        """
        Initializes the MAP Deblurer.

        Parameters
        ----------
        rho - optimization parameter (optimal 1 / sigma_dae**2), where sigma_dae is the sigma used for training the MAP denoiser
        sigma - standard deviation of the noise used to degrade the image
        image_shape - shape of the image to deblur
        kernel_shape - shape of the blur kernel
        denoiser - path to the frozen denoiser protobuf file
        """
        tf.reset_default_graph()

        # Load denoiser
        with tf.gfile.GFile(denoiser, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.rho = rho
        self.sigma = sigma
        self.image_shape = image_shape

        half = [kernel_shape[0] // 2, kernel_shape[1] // 2]
        self.valid_shape = (image_shape[0] - (kernel_shape[0] - 1), image_shape[1] - (kernel_shape[1] - 1))
        self.kernel_shape = kernel_shape

        self.image_ph = tf.placeholder(tf.float32, (None,) + image_shape, "blurry_image")
        self.kernel_sig_ph = tf.placeholder(tf.complex64, image_shape[:2], "blur_kernel")  # conjugated and flipped version of the kernel that was used to degrade the image

        self.y = tf.get_variable("y", initializer=self.image_ph, validate_shape=False)
        self.x_hat = tf.get_variable("x_hat", initializer=self.image_ph, validate_shape=False)
        self.z_hat = tf.get_variable("z_hat", initializer=self.image_ph, validate_shape=False)
        self.lam = tf.get_variable("lam", initializer=tf.zeros(tf.shape(self.image_ph)), validate_shape=False)

        self.kernel_sig = tf.expand_dims(self.kernel_sig_ph, axis=0)
        self.kernel_sig = tf.expand_dims(self.kernel_sig, axis=0)

        # self.kernel_sig = tf.pad(self.kernel_sig, paddings)

        self.kernel_sig = tf.get_variable("kernel_sig", initializer=self.kernel_sig, validate_shape=False)

        self.rev_kernel_sig = tf.math.conj(tf.reverse(self.kernel_sig, axis=(0, 1)))
        self.rev_kernel_sig = tf.get_variable("rev_kernel_sig", initializer=self.rev_kernel_sig, validate_shape=False)

        self.denominator = tf.cast((tf.abs(self.kernel_sig) ** 2 / self.sigma ** 2) + self.rho, tf.complex64)
        self.denominator = tf.get_variable("denominator", initializer=self.denominator, validate_shape=False)

        # image = tf.cast(tf.transpose(self.image_ph, (0,3,1,2)),tf.complex64)

        # paddings = [[0,0], [0,0], [half[0], half[0]], [half[1], half[1]]]

        # if padding_mode == "WRAP":
        #    image = wrap_pad(image, half)
        # else:
        #    image = tf.pad(image, paddings, mode=padding_mode)

        # self.ul = tf.get_variable("ul", initializer=self.ul, validate_shape=False)

        x_tilde = tf.cast(tf.transpose(self.z_hat - self.lam, (0, 3, 1, 2)), tf.complex64)
        # if padding_mode == "WRAP":
        #    x_tilde = wrap_pad(x_tilde, half)
        # else:
        #    x_tilde = tf.pad(x_tilde, paddings, mode=padding_mode)

        x_hat = tf.cast(tf.transpose(self.x_hat[:, :, :, :], (0, 3, 1, 2)), tf.complex64)
        y_est = tf.real(tf.spectral.ifft2d(tf.spectral.fft2d(x_hat) * self.rev_kernel_sig))
        y = tf.transpose(self.y, (0, 3, 1, 2))

        mask = tf.ones((1, 1) + self.valid_shape, dtype=tf.float32)
        mask = tf.pad(mask, [[0, 0], [0, 0], [half[0], half[0]], [half[1], half[1]]])

        y_est = y * mask + y_est * np.abs(mask - 1.0)
        self.y_est = tf.transpose(y_est, (0, 2, 3, 1))

        self.ul = self.kernel_sig * tf.spectral.fft2d(tf.cast(y_est, tf.complex64)) / self.sigma ** 2

        x_hat = tf.real(tf.spectral.ifft2d((self.ul + self.rho * tf.spectral.fft2d(x_tilde)) / self.denominator))
        # x_hat = x_hat[:,:,half[0]:-half[0],half[1]:-half[1]]
        x_hat = tf.transpose(x_hat, (0, 2, 3, 1))

        self.eq1 = self.x_hat.assign(x_hat)

        # eq3
        self.eq3 = self.lam.assign(self.lam + (self.x_hat - self.z_hat))

        # eq2
        x_tilda = (self.x_hat + self.lam) / 255.0
        self.denoised = tf.import_graph_def(
            graph_def,
            input_map={"clean_image": x_tilda},
            return_elements=["denoised:0"])

        self.eq2 = self.z_hat.assign(self.denoised[0] * 255.0)

        self.init = tf.global_variables_initializer()

    def optimize(self, image, kernel, gt=None, nr_iters=35, test_iter=5, plot=False):
        """
        Optimize an image

        Parameters
        ----------
        image - degraded image (of the same size as the ground truth
        kernel - kernel used to degrade the image
        gt - (optional) ground truth image
        nr_iters - Number of iterations of the ADMM algorithm
        test_iter - Number of iterations between PSNR computations
        plot - if True, result after 5th, 10th, and 35th iteration is plotted

        Returns
        -------
        res - Image estimate
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        kernel_sig = psf2otf(np.flipud(np.fliplr(kernel)), [image.shape[0], image.shape[1]])
        kernel_sig = np.conj(kernel_sig)

        total_time = time.time()
        if plot:
            plot_ind = 1
            plt.rc(('xtick', 'ytick'), color=(1, 1, 1, 0))
            fig = plt.figure(figsize=(12, 6))
            ax = plt.subplot(141)
            pimg = np.squeeze(np.clip(image / 255.0, 0.0, 1.0))
            pimg = pimg[self.kernel_shape[0] // 2:-self.kernel_shape[0] // 2,
                        self.kernel_shape[1] // 2:-self.kernel_shape[1] // 2, :]
            plt.imshow(pimg, cmap="gray")
            plt.title('Blurred')

        pad_y = self.kernel_shape[0] // 2
        pad_x = self.kernel_shape[1] // 2

        image = image[None, ...]

        sess.run(self.init, feed_dict={self.image_ph: image, self.kernel_sig_ph: kernel_sig})

        for i in range(nr_iters):
            t = time.time()
            sess.run(self.eq1)
            sess.run(self.eq3)
            sess.run(self.eq2)

            if (i + 1) % test_iter == 0:
                res = np.squeeze(sess.run(self.x_hat), axis=0)

                if plot and ((i + 1) == 5 or (i + 1) == 10 or (i + 1) == 35):
                    ax = plt.subplot(141 + plot_ind)
                    pimg = np.squeeze(np.clip(res / 255.0, 0.0, 1.0))[
                           self.kernel_shape[0] // 2:-self.kernel_shape[0] // 2,
                           self.kernel_shape[1] // 2:-self.kernel_shape[1] // 2, :]
                    plt.imshow(pimg, cmap="gray")
                    plt.title('Iteration ' + str(i + 1))
                    plot_ind = plot_ind + 1

                if gt is not None:
                    psnr = computePSNR(gt, res, pad_y, pad_x)
                    print ('Iter. ' + str(i + 1) + ': PSNR is:' + str(psnr) + ', iteration finished in ' + str(
                        time.time() - t) + ' seconds')
                else:
                    print('Iter. ' + str(i + 1) + ': iteration finished in ' + str(time.time() - t) + ' seconds')

        print("Optimization finished after: {:0.2f}s".format(time.time() - total_time))
        res = np.squeeze(sess.run(self.x_hat), axis=0)
        # plt.savefig("progress.png", dpi=400, bbox_inches='tight')
        return res
