from utils import computePSNR
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt


class MAPinpainting:
    def __init__(self, rho, sigma, image_shape, denoiser='./models/map_color/optimizedMAPdenoiser.pb'):
        """
        Initializes the MAP Deblurer.

        Parameters
        ----------
        rho - optimization parameter (optimal 1 / sigma_dae**2), where sigma_dae is the sigma used for training the MAP denoiser
        sigma - standard deviation of the noise used to degrade the image
        image_shape - shape of the image to deblur
        denoiser - path to the frozen denoiser protobuf file
        """
        tf.reset_default_graph()

        # Load denoiser
        with tf.gfile.GFile(denoiser, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        self.rho = rho
        self.sigma = sigma

        self.initial_ph = tf.placeholder(tf.float32, (None,) + image_shape, "init_image")
        self.mask_ph = tf.placeholder(tf.float32, (None,) + tuple(image_shape[:2]) + (1,), "mask_ph")

        self.y = tf.get_variable("y", initializer=self.initial_ph, validate_shape=False)
        self.x_hat = tf.get_variable("x_hat", initializer=self.initial_ph, validate_shape=False)
        self.z_hat = tf.get_variable("z_hat", initializer=self.initial_ph, validate_shape=False)
        self.lam = tf.get_variable("lam", initializer=tf.zeros(tf.shape(self.initial_ph)), validate_shape=False)
        self.mask = tf.get_variable("mask", initializer=self.mask_ph, validate_shape=False)

        self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name="global_step")

        self.first_term = self.y - self.x_hat * self.mask
        self.second_term = self.x_hat - self.z_hat + self.lam
        self.data_loss = (tf.nn.l2_loss(self.first_term) / (self.sigma ** 2)) + self.rho * tf.nn.l2_loss(
            self.second_term)

        optimizer = tf.train.AdamOptimizer(learning_rate=(self.sigma / 2.55))
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=(self.sigma/100))

        self.eq1_minimizer = optimizer.minimize(self.data_loss, var_list=[self.x_hat], global_step=self.global_step)

        self.eq1_init = tf.variables_initializer(optimizer.variables())

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

    def optimize(self, initial, mask, gt, nr_iters=35, test_iter=5, plot=False, inner_iterations=200):
        """
        Optimize an image

        Parameters
        ----------
        initial - degraded image (of the same size as the ground truth)
        mask - boolean mask of the missing regions (False for the missing pixel)
        gt - (optional) ground truth image
        nr_iters - Number of iterations of the ADMM algorithm
        test_iter - Number of iterations between PSNR computations
        plot - if True, result after 5th, 50th, and 100th iteration is plotted
        inner_iterations - Number of gradient steps for optimizing the equation 1 of ADMM

        Returns
        -------
        res - Image estimate
        """

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        total_time = time.time()
        if plot:
            plot_ind = 1
            plt.rc(('xtick', 'ytick'), color=(1, 1, 1, 0))
            fig = plt.figure(figsize=(15, 15))
            ax = plt.subplot(141)
            plt.imshow(np.squeeze(np.clip(initial / 255.0, 0, 1)), cmap="gray")
            plt.title('Blurred')

        initial = initial[None, ...]
        mask = mask[None, ...]

        sess.run(self.init, feed_dict={self.initial_ph: initial, self.mask_ph: mask})

        for i in range(nr_iters):
            t = time.time()
            for j in range(inner_iterations):
                _ = sess.run(self.eq1_minimizer)
            sess.run(self.eq3)
            sess.run(self.eq2)

            if (i + 1) % test_iter == 0:
                res = np.squeeze(sess.run(self.x_hat), axis=0)

                if ((i + 1) == 5 or (i + 1) == 50 or (i + 1) == 100) and plot:
                    ax = plt.subplot(141 + plot_ind)
                    plt.imshow(np.squeeze(np.clip(res / 255.0, 0, 1)), cmap="gray")
                    plt.title('Denoise at iteration ' + str(i + 1))
                    plot_ind = plot_ind + 1

                if gt is not None:
                    psnr = computePSNR(gt, res, 0, 0)
                    print ('iteration {:d}, PSNR is: {:0.4f}, iteration finished in {:0.2f} s'.format(i, psnr, time.time() - t))
                else:
                    print('iteration {:d}, iteration finished in {:0.2f} s'.format(i, time.time() - t))

        print("Optimization finished after: {:0.2f}s".format(time.time() - total_time))
        res = np.squeeze(sess.run(self.x_hat), axis=0)
        return res
