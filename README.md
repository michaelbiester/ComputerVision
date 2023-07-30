# ComputerVision

## Overview 

This repository serves to explore some image processing topics with Ipython notebooks

## Notebooks

Here only the more *important* Jupyter notebooks are summarized. 

(notebooks of lesser importance are those, which have proved useful to study some concepts and programming techniques)


**file**: `corr_and_conv_e.ipynb`

review of the concept of *correlation* applied to image processing. The concept of convolution is only briefly 
touched at the end of the notebook. A PDF version `corr_and_conv_e.pdf` of the notebook is provided for better readability.

**file**: `correlation_convolution_2d_experiments.ipynb`

another more detailed review of *correlation* and *convolution* applied to 2D images. Moreover the interrelation of correlation and convolution is explained. The OpenCV library has only support for correlation but mentions, that by modification of the kernel (filter) convolution can be computed as well. A PDF version `correlation_convolution_2d_experiments.pdf` is provided to get a quick overview without having to run the notebook.

**file**: `Fourier_1D_e.ipynb`

To familiarize myself with the discrete Fourier transform `DFT` a reviewed DFT concepts for the 1D case. A PDF version `Fourier_1D_e.pdf` is also available.

**file**: `Fourier_1D_application_1.ipynb`

Some applications of the 1D DFT. Demonstrates how to apply a time shift in the frequency domain and uses Python libraries `numpy` and `scipy`. A PDF version `Fourier_1D_application_1.pdf` is available.

**file**: `fourier_2d.ipynb`

reviews properties to the 2D discrete Fourier transform. PDf version: `fourier_2d.pdf`

**file**: `frequency_domain_filtering.ipynb`

how to filter an image in the *frequency* domain and demonstrating the effect of restricting the *bandwidth* of the filter. PDF version: `frequency_domain_filtering.pdf`

**file**: `motion_detection.ipynb`

Shows various methods to compare images and proposes a simple method to detect changes in an image (possible application: motion detection). The method is simple but a *production quality* motion detection schema requires a different approach (eg: extracting features of image1 and image2; then comparing features which a common to both images and which are present in one image but missing in the other ... ). PDF version: `motion_detection.pdf`



## Sub-directories

`figures`, `img` : some test images



