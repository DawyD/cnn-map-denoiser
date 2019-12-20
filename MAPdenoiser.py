from glob import glob
from utils import load_data, load_images, load_images_rgb, save_images, tf_psnr, cal_psnr
import tensorflow as tf
from os import makedirs, path
import numpy as np
import time
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib


class MAPdenoiser(object):
    def __init__(self, channels=1, stddev=0.01):
        """
        MAP denoiser trainer

        Parameters
        ----------
        channels - number of channels of the input images (1 - for grayscale images)
        stddev - standard deviation of the noise (assuming pixel values in range 0-1)
        """
        tf.reset_default_graph()
        self.channels = channels
        self.stddev = stddev

        # build model
        self.v_ph = tf.placeholder(tf.float32, [None, None, None, self.channels], name='clean_image')
        self.is_training_ph = tf.placeholder(tf.bool, name='is_training')
        self.lr_ph = tf.placeholder(tf.float32, name='learning_rate')
        self.ro_ph = tf.placeholder(tf.float32, name='ro')

        self.img_shape = tf.shape(self.v_ph)

        self.v = self.v_ph + tf.random_normal(self.img_shape, stddev=0.0)  # no noise added
        self.Dv = MAPdenoiser.dncnn(self.v, is_training=self.is_training_ph, output_channels=self.channels)
        self.Dv = tf.identity(self.Dv, name="denoised")

        self.data_loss = tf.losses.mean_squared_error(self.v, self.Dv)
        self.psnr_loss = tf_psnr(self.Dv, self.v)

        self.vn = self.v + tf.random_normal(self.img_shape, stddev=self.stddev)
        self.Rv = MAPdenoiser.dae(self.vn, is_training=self.is_training_ph, output_channels=self.channels, reuse=False)

        self.Dvn = self.Dv + tf.random_normal(self.img_shape, stddev=self.stddev)
        self.RDv = MAPdenoiser.dae(self.Dvn, is_training=self.is_training_ph, output_channels=self.channels, reuse=True)
        self.RDv = tf.stop_gradient(self.RDv)

        self.dae_loss = tf.losses.mean_squared_error(self.v, self.Rv)
        self.reg_loss = tf.losses.mean_squared_error(self.Dv, self.RDv)

        self.map_loss = self.reg_loss / (self.stddev ** 2) + (self.ro_ph / 2) * self.data_loss

        var_list_D = [i for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')]
        self.train_op_D = tf.train.AdamOptimizer(self.lr_ph).minimize(self.map_loss, var_list=var_list_D)

        var_list_R = [i for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')]
        self.train_op_R = tf.train.AdamOptimizer(self.lr_ph).minimize(self.dae_loss, var_list=var_list_R)

        tf.summary.scalar('train/map_loss', self.map_loss)
        tf.summary.scalar('train/data_loss', self.data_loss)
        tf.summary.scalar('train/reg_loss', self.reg_loss)
        # tf.summary.scalar('denoising_loss', self.denoising_loss)
        tf.summary.scalar('train/lr', self.lr_ph)
        tf.summary.scalar('train/eva_psnr', self.psnr_loss)
        self.summaries = tf.summary.merge_all()

        self.summary_psnr = tf.summary.scalar('test/eva_psnr', self.psnr_loss)

        self.init = tf.global_variables_initializer()

    @staticmethod
    def dncnn(input, is_training=True, output_channels=1):
        """
        DNCNN network (generator)
        Parameters
        ----------
        input - input tensor
        is_training - tensor specifying whether it is launched in training or test mode
        output_channels - number of output channels

        Returns
        -------
        denoised image estimate
        """
        with tf.variable_scope('generator'):
            output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu, name='conv1')
            for layers in range(2, 16 + 1):
                output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
                # output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
                output = tf.nn.relu(output)
            output = tf.layers.conv2d(output, output_channels, 3, padding='same', name='conv17')
            return input + output

    @staticmethod
    def dae(input, is_training=True, output_channels=1, reuse=True):
        """
        Denoising Auto-Encoder network (discriminator)
        Parameters
        ----------
        input - input tensor
        is_training - tensor specifying whether it is launched in training or test mode
        output_channels - number of output channels
        reuse - specifies whether the variables should be created or reused
        Returns
        -------
        denoised image estimate
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu, name='conv1')
            for layers in range(2, 16 + 1):
                output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
                # output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
                output = tf.nn.relu(output)
            output = tf.layers.conv2d(output, output_channels, 3, padding='same', name='conv17')
            return input + output

    def train(self, epoch=50, batch_size=128, lr=0.0001, ckpt_dir='./checkpoint_map', sample_dir='./sample',
              dataset_path='./data/img_clean_pats_rgb.npy', eval_set='Set12', eval_every_epoch=1):

        """
        Train the MAP denoiser
        If the ckpt_dir leads to a valid, model this model is loaded and training continues

        Parameters
        ----------
        epoch - number of training epochs
        batch_size - batch size
        lr - learning rate
        ckpt_dir - directory for saving the model
        sample_dir - directory for saving evaluation images
        dataset_path - path to the numpy file with the training data [size, height, width, channels]
        eval_set - evalutation dataset folder name (found in ./data)
        eval_every_epoch - Number of epochs between the evaluations
        """

        if not path.exists(ckpt_dir):
            makedirs(ckpt_dir)
        if not path.exists(sample_dir):
            makedirs(sample_dir)

        lr = lr * np.ones([epoch])
        lr[30:] = lr[0] / 10.0

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        sess.run(self.init)
        print("[*] Initialize model successfully...")

        # Start network training
        # We train both models at the same time instead of sequentially

        with load_data(filepath=dataset_path) as training_data:
            training_data = training_data.astype(np.float32) / 255.0  # normalize the data to 0-1
            eval_files = glob('./data/{}/*.png'.format(eval_set))
            if self.channels == 3:
                eval_data = load_images_rgb(
                    eval_files)  # list of array of different size, 4-D, pixel value range is 0-255
            else:
                eval_data = load_images(eval_files)  # list of array of different size, 4-D, pixel value range is 0-255

            numBatch = int(training_data.shape[0] / batch_size)

            # load pretrained model
            load_model_status, global_step = self.load(sess, ckpt_dir)
            if load_model_status:
                iter_num = global_step
                start_epoch = global_step // numBatch
                start_step = global_step % numBatch
                print("[*] Model restore success!")
            else:
                iter_num = 0
                start_epoch = 0
                start_step = 0
                print("[*] Not find pretrained model!")

            # make summary
            summary_writer = tf.summary.FileWriter('./logs', sess.graph)
            print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
            start_time = time.time()
            self.evaluate(sess, iter_num, eval_data, sample_dir, summary_writer)

            for epoch in range(start_epoch, epoch):
                np.random.shuffle(training_data)
                for batch_id in range(start_step, numBatch):
                    batch_images = training_data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                    _, dae_loss = sess.run([self.train_op_R, self.dae_loss],
                                           feed_dict={self.v_ph: batch_images,
                                                      self.lr_ph: lr[epoch],
                                                      self.is_training_ph: True})

                    _, map_loss, summary = sess.run([self.train_op_D, self.map_loss, self.summaries],
                                                    feed_dict={self.v_ph: batch_images,
                                                               self.lr_ph: lr[epoch],
                                                               self.ro_ph: 2 / (self.stddev ** 2),
                                                               self.is_training_ph: True})

                    # print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f, den_loss: %.6f"
                    #      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, map_loss, dae_loss))
                    iter_num += 1
                    summary_writer.add_summary(summary, iter_num)
                if np.mod(epoch + 1, eval_every_epoch) == 0:
                    self.evaluate(sess, iter_num, eval_data, sample_dir, summary_writer)
                    self.save(sess, iter_num, ckpt_dir)
            print("[*] Finish training.")

    def evaluate(self, sess, iter_num, test_data, sample_dir, summary_writer):
        """
        Evaluate denoising

        Parameters
        ----------
        sess - Tensorfow session
        iter_num - Iteration number
        test_data - list of array of different size, 4-D, pixel value range is 0-255
        sample_dir - evalutation dataset folder name (found in ./data)
        summary_writer - Tensorflow SummaryWriter

        Returns
        -------

        """
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0
        for idx in range(len(test_data)):
            clean_image = test_data[idx].astype(np.float32) / 255.0
            output_clean_image, noisy_image, psnr_summary = sess.run(
                [self.Dv, self.v, self.summary_psnr],
                feed_dict={self.v_ph: clean_image,
                           self.is_training_ph: False})
            summary_writer.add_summary(psnr_summary, iter_num)
            groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            # print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            save_images(path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        groundtruth, noisyimage, outputimage)
        avg_psnr = psnr_sum / len(test_data)

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

    def denoise_img(self, image, ckpt_dir='./checkpoint_map'):
        """
        Denoise a single image using a pretrained denoiser

        Parameters
        ----------
        image - Degraded image
        ckpt_dir - checkpoint directory containing the pretrained model

        Returns
        -------
        Image estimage

        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.load(sess, ckpt_dir)

        image = np.expand_dims(image, axis=0)
        image = sess.run(self.Dv, feed_dict={self.v_ph: image, self.is_training_ph: False})
        return np.squeeze(image)

    def denoise(self, data, ckpt_dir='./checkpoint_map'):
        """
        Denoise batch of images using a pretrained denoiser
        Parameters
        ----------
        data - batch of input images
        ckpt_dir - checkpoint directory containing the pretrained model

        Returns
        -------
        output_clean_image - batch of image estimates
        noisy_image - batch of input images
        psnr - PSNR (scalar)
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.load(sess, ckpt_dir)

        output_clean_image, noisy_image, psnr = sess.run(
            [self.Dv, self.v, self.psnr_loss], feed_dict={self.v_ph: data, self.is_training_ph: False})
        return output_clean_image, noisy_image, psnr

    def save(self, sess, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        """
        Save the model

        Parameters
        ----------
        sess - Tensorflow session
        iter_num - iteration number
        ckpt_dir - directory into which the model will be saved
        model_name - name of the model
        """
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not path.exists(checkpoint_dir):
            makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(sess,
                   path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, sess, checkpoint_dir):
        """
        Load the model

        Parameters
        ----------
        sess - Tensorflow session
        checkpoint_dir - checkpoint directory containing the pretrained model

        Returns
        -------
        loaded - True if model was loaded
        global_step - Iteration step of the loaded model
        """
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, sess, test_files, ckpt_dir, save_dir):
        """
        Test MAP denoising

        Parameters
        ----------
        sess - Tensorflow session
        test_files - list of filenames of images to test
        ckpt_dir - checkpoint directory containing the pretrained model
        save_dir - directory into which the noisy and estimate images will be saved

        Returns
        -------

        """
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, _ = self.load(sess, ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] " + 'noise variance: ' + str(self.stddev ** 2) + " start testing...")
        for idx in range(len(test_files)):
            if self.channels == 1:
                clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            else:
                clean_image = load_images_rgb(test_files[idx]).astype(np.float32) / 255.0

            output_clean_image, noisy_image = sess.run([self.Dv, self.v],
                                                       feed_dict={self.v_ph: clean_image,
                                                                  self.is_training_ph: False})
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(path.join(save_dir, 'noisy%d.png' % idx), noisyimage)
            save_images(path.join(save_dir, 'denoised%d.png' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)

    def freeze_graph(self, ckpt_dir='./checkpoint_map'):
        """
        Freezes the Graph and optimizes it for inference. Creates frozenMAPdenoiser.pb and optimizedMAPdenoiser.pb
        Parameters
        ----------
        ckpt_dir - checkpoint directory containing the pretrained model
        """
        tf.train.write_graph(tf.get_default_graph().as_graph_def(), ckpt_dir, 'tensorflowModel.pbtxt', as_text=False)

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)

        freeze_graph.freeze_graph(input_graph=path.join(ckpt_dir, 'tensorflowModel.pbtxt'),
                                  input_saver="",
                                  input_binary=True,
                                  input_checkpoint=full_path,
                                  output_node_names="denoised",
                                  restore_op_name=None,
                                  filename_tensor_name=None,
                                  output_graph=path.join(ckpt_dir, 'frozenMAPdenoiser.pb'),
                                  clear_devices=True,
                                  initializer_nodes="")

        inputGraph = tf.GraphDef()
        with tf.gfile.Open(path.join(ckpt_dir, 'frozenMAPdenoiser.pb'), "rb") as f:
            data2read = f.read()
            inputGraph.ParseFromString(data2read)

            outputGraph = optimize_for_inference_lib.optimize_for_inference(
                inputGraph,
                ["clean_image"],  # an array of the input node(s)
                ["denoised"],  # an array of output nodes
                tf.int32.as_datatype_enum)

            # Save the optimized graph'test.pb'
            f = tf.gfile.FastGFile(path.join(ckpt_dir, 'optimizedMAPdenoiser.pb'), "w")
            f.write(outputGraph.SerializeToString())


class frozenMAPdenoiser:
    def __init__(self, channels=1, ckpt_dir='./checkpoint_map', gpu_ratio=0.2):
        """
        Initialize the MAP denoiser with the frozen (optimized) model produced by MAPdenoiser

        Parameters
        ----------
        channels - number of image channels
        ckpt_dir - checkpoint directory containing the pretrained model
        gpu_ratio - GPU ratio for inference
        """
        graph_filename = path.join(ckpt_dir, "optimizedMAPdenoiser.pb")
        with tf.gfile.GFile(graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            print([n.name + '=>' + n.op for n in graph_def.node if n.op in ('Identity', 'Placeholder')])

        # Define new graph
        tf.reset_default_graph()
        self.tf_image = tf.placeholder(tf.float32, [None, None, None, channels], name='clean_image')

        self.tf_denoised = tf.import_graph_def(
            graph_def,
            input_map={"clean_image": self.tf_image},
            return_elements=["denoised:0"])

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_ratio)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def denoise(self, image):
        """
        Denoise an image

        Parameters
        ----------
        image - Noisy image

        Returns
        -------
        Image estimate
        """
        image = np.expand_dims(image, 0)
        res = self.sess.run(self.tf_denoised, {self.tf_image: image})[0]
        return np.squeeze(res, axis=0)
