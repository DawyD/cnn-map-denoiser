import gc
import os
import sys

import numpy as np
import math as m
import tensorflow as tf
from PIL import Image
import scipy.signal as sig
from skimage.measure import compare_ssim as ssim


def data_augmentation(image, mode):
    """
    Augment image based on the mode (0-7)

    Parameters
    ----------
    image : Input image to be augmented
    mode :
        ``0`` -  original
        ``1`` -  flip up and down
        ``2`` -  rotate counter-clockwise 90 degree
        ``3`` -  rotate 90 degree and flip up and down
        ``4`` -  rotate 180 degree
        ``5`` -  rotate 180 degree and flip up and down
        ``6`` -  rotate 270 degree
        ``7`` -  rotate 270 degree and flip up and down

    Returns
    -------
    Augmented image
    """
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counter-clockwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


class train_data:
    def __init__(self, filepath='./data/image_clean_pat.npy'):
        """
        Set the training data path
        Parameters
        ----------
        filepath - training data filepath
        """
        self.filepath = filepath
        assert '.npy' in filepath
        if not os.path.exists(filepath):
            print("[!] Data file not exists")
            sys.exit(1)

    def __enter__(self):
        print("[*] Loading data...")
        self.data = np.load(self.filepath)
        np.random.shuffle(self.data)
        print("[*] Load successfully...")
        return self.data

    def __exit__(self, type, value, trace):
        del self.data
        gc.collect()
        print("In __exit__()")


def load_data(filepath='./data/image_clean_pat.npy'):
    return train_data(filepath=filepath)


def load_images(filelist):
    """
    Load images (as grayscale) given the filename or list of filenames

    Parameters
    ----------
    filelist - filename or list of filenames

    Returns
    -------
    List of images of shape [height, width, 1]
    """
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data


def load_images_rgb(filelist):
    """
    Load images (as RGB) given the filename or list of filenames

    Parameters
    ----------
    filelist - filename or list of filenames

    Returns
    -------
    List of images of shape [height, width, 3]
    """
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist)
        if len(im.size) < 3:
            im = np.expand_dims(im, axis=2)
        if im.shape[2] == 1:
            im = np.concatenate((im, im, im), axis=2)

        return np.array(im).reshape(1, im.size[1], im.size[0], 3)
    data = []
    for file in filelist:
        im = Image.open(file)
        if len(im.size) < 3:
            im = np.expand_dims(im, axis=2)
            im = np.concatenate((im, im, im), axis=2)
            data.append(np.array(im).reshape(1, im.shape[1], im.shape[0], 3))
        else:
            data.append(np.array(im).reshape(1, im.size[1], im.size[0], 3))
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    """
    Save image (concatenated horizontally with ground-truth and noisy image if provided in the order of the parameters)

    Parameters
    ----------
    filepath - filepath of the new image
    ground_truth - ground truth image
    noisy_image - noisy image
    clean_image - image estimate
    """
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'png')


def cal_psnr(im1, im2):
    """
    Compute PSNR in on floating point data assuming range [0-255]

    Parameters
    ----------
    im1 - image 1
    im2 - image 2

    Returns
    -------
    PSNR
    """
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def tf_psnr(im1, im2):
    """
    Compute PSNR between two TensorFlow tensors that are in range 0-255

    Parameters
    ----------
    im1 - Tensor 1
    im2 - Tensor 2

    Returns
    -------
    PSNR between the tensors
    """
    mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
    return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))


def computePSNR(img1, img2, pad_y=0, pad_x=0):
    """ Computes PSNR between two images after clipping to the uint8 format.
    Input:
    img1: First image in range of [0, 255].
    img2: Second image in range of [0, 255].
    pad_y: Scalar radius to exclude boundaries from contributing to PSNR computation in vertical direction.
    pad_x: Scalar radius to exclude boundaries from contributing to PSNR computation in horizontal direction.
    
    Output: PSNR """
    if pad_y != 0 and pad_x != 0:
        img1_u = (np.clip(img1, 0, 255.0)[pad_y:-pad_y, pad_x:-pad_x, ...]).astype(dtype=np.uint8)
        img2_u = (np.clip(img2, 0, 255.0)[pad_y:-pad_y, pad_x:-pad_x, ...]).astype(dtype=np.uint8)
    else:
        img1_u = (np.clip(img1, 0, 255.0)).astype(dtype=np.uint8)
        img2_u = (np.clip(img2, 0, 255.0)).astype(dtype=np.uint8)
    imdiff = (img1_u).astype(dtype=np.float32) - (img2_u).astype(dtype=np.float32)
    rmse = np.sqrt(np.mean(np.power(imdiff[:], 2)))
    return 20.0 * np.log10(255.0 / rmse)


def computeSSIM(img1, img2, pad_y=0, pad_x=0):
    """ Computes peak signal-to-noise ratio between two images. 
    Input:
    img1: First image in range of [0, 255].
    img2: Second image in range of [0, 255].
    pad_y: Scalar radius to exclude boundaries from contributing to PSNR computation in vertical direction.
    pad_x: Scalar radius to exclude boundaries from contributing to PSNR computation in horizontal direction.
    
    Output: PSNR """
    if pad_y != 0 and pad_x != 0:
        img1_u = (np.clip(img1, 0, 255.0)[pad_y:-pad_y, pad_x:-pad_x, ...]).astype(dtype=np.uint8)
        img2_u = (np.clip(img2, 0, 255.0)[pad_y:-pad_y, pad_x:-pad_x, ...]).astype(dtype=np.uint8)
    else:
        img1_u = (np.clip(img1, 0, 255.0)).astype(dtype=np.uint8)
        img2_u = (np.clip(img2, 0, 255.0)).astype(dtype=np.uint8)
    return ssim(img1_u, img2_u)


def filter_image(image, kernel, mode='valid', boundary='symm'):
    """
    Implements color filtering (convolution using a flipped kernel on each channel separately)

    Parameters
    ----------
    image : image [width, height, channels] to be filtered
    kernel : 2D filtering kernel
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:

        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.

    Returns
    -------
    Filtered image
    """
    chs = []
    for d in range(image.shape[2]):
        channel = sig.convolve2d(image[:, :, d], np.flipud(np.fliplr(kernel)), mode=mode, boundary=boundary)
        chs.append(channel)
    return np.stack(chs, axis=2)


def convolve_image(image, kernel, mode='valid', boundary='symm'):
    """
    Implements color convolution (2D convolution on each channel separately)

    Parameters
    ----------
    image : image [width, height, channels] to be convolved with the kernel
    kernel : 2D convolution kernel
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:

        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.

    Returns
    -------
    Convolved image
    """
    chs = []
    for d in range(image.shape[2]):
        channel = sig.convolve2d(image[:, :, d], kernel, mode=mode, boundary=boundary)
        chs.append(channel)
    return np.stack(chs, axis=2)


# The following 2 function are implemented here: https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py
def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    # n_ops = np.sum(psf.size * np.log2(psf.shape))
    # otf = np.real_if_close(otf, tol=n_ops)

    return otf


def edgeTaper(img, kernel):
    """
    Smooth-out the edges of an image in the non-valid region of filtering to have better properties when using
    circular convolutions

    Parameters
    ----------
    img - Filtered image
    kernel - Input kernel used to filter the image

    Returns
    -------
    Image with smoothed edge regions
    """

    blur = convolve_image(img, kernel, mode='same', boundary='fill')
    weight = convolve_image(img * 0 + 1.0, kernel, mode='same', boundary='fill')
    deg_pad = weight * img + (1.0 - weight) * blur / weight
    return deg_pad


def wrap_pad(input, size):
    """
    Pad the input image in the circular fashion (as assumed by the convolution in the frequency domain)
    Parameters
    ----------
    input - 4D array [samples, height, width, channels]
    size -  2D size of the padding on each side [vertical, horizontal]

    Returns
    -------
    Padded image
    """
    M1 = tf.concat([input[:, :, -size[1]:, :], input, input[:, :, 0:size[1], :]], 2)
    M1 = tf.concat([M1[:, -size[0]:, :, :], M1, M1[:, 0:size[0], :, :]], 1)
    return M1


def real_blur_and_noise(image, kernel, sigma_d):
    """
    Blur the image with the kernel and adds the Gaussian noise with standard deviation of sigma_d.
    Blurred image is padded to the original size and the edge regions are handled using edgeTaper.
    :param image: image to be degraded
    :param kernel: blurring kernel
    :param sigma_d: standard deviation of the noise
    :returns initial: degraded image padded to the original size
    """
    degraded = filter_image(image, kernel, mode="valid", boundary="fill")
    noise = np.random.normal(0.0, sigma_d, degraded.shape).astype(np.float32)
    degraded = degraded + noise
    initial = np.pad(degraded, ((kernel.shape[0] // 2, kernel.shape[0] // 2),
                                (kernel.shape[1] // 2, kernel.shape[1] // 2),
                                (0, 0)), mode="edge")
    initial = edgeTaper(initial, kernel)
    return initial


def drop_and_noise(image, sigma_d, percentage=0.8):
    """
    Drops *percentage* of pixels from the image and adds the noise of standard deviation stddev to the rest

    Parameters
    ----------
    image - image to be degraded
    sigma_d - standard deviation of the noise
    percentage - percentage (0-1) of the pixels to be dropped

    Returns
    -------
    y - degraded image
    mask - boolean mask of dropped pixels (False, where it is dropped)
    """
    M, N = image.shape[:2]
    n = N * M
    p = m.floor(percentage * n)
    image = np.cast[np.float32](image)

    missing_pixels_ind = np.random.permutation(n)[:p]

    mask = np.ones((M * N,), dtype=np.bool)
    mask[missing_pixels_ind] = 0
    mask = mask.reshape((M, N, 1))

    maskf = np.cast[np.float32](mask)
    y_clean = image * maskf

    noise = np.random.normal(loc=0, scale=sigma_d, size=image.shape) * maskf
    y = y_clean + noise

    return y, mask


def median_inpainting(y, mask):
    """
    Inpaints an image using repetitive application of a median filter
    Parameters
    ----------
    y - degraded image with missing pixels
    mask - boolean mask of the dropped pixels (False, where it is dropped)

    Returns
    -------
    y0_median - estimate of the image
    """
    # grayscale only so far
    M, N = y.shape[:2]
    y0_median = np.copy(y)
    y0_median[~mask] = np.nan
    win_size = 0
    bitmap_NaN = np.isnan(y0_median)
    while np.count_nonzero(bitmap_NaN) > 0:
        y0_median_prev = np.copy(y0_median)
        win_size = win_size + 1
        rows, cols, _ = np.where(bitmap_NaN)
        for i in range(len(cols)):
            row_start = max([1, rows[i] - win_size])
            row_end = min([M, rows[i] + win_size])
            col_start = max([1, cols[i] - win_size])
            col_end = min([N, cols[i] + win_size])
            median_val = np.nanmedian(y0_median_prev[row_start:row_end, col_start:col_end, :])
            y0_median[rows[i], cols[i]] = median_val
        bitmap_NaN = np.isnan(y0_median)
    return y0_median
