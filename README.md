# Open Neural Network Exchange (ONNX), PyTorch, Caffe2 #

The Open Neural Network Exchange (ONNX) is an open source Artificial Intelligence (AI) ecosystem/open standard for Machine Learning (ML) interoperability.

*Source: https://onnx.ai*

PyTorch is an open source ML framework based upon Torch (a scientific computing framework that supports GPU-based ML algorithms). 

*Source: http://torch.ch*

Convolutional Architecture for Fast Feature Embedding (CAFFE OR Caffe) is an open source ML and Deep Learning (DL) framework, which supports various DL architectures. Caffe2 builds upon Caffe (a.k.a. Caffe1).

*Source: https://developer.nvidia.com/blog/caffe2-deep-learning-framework-facebook/*

Depending upon the Operating Systems (OS) desired (e.g., Redhat, Ubuntu, etc.), Caffe/Caffe2 has various prerequisites/dependencies (e.g., https://caffe.berkeleyvision.org/installation.html). For the experimentation herein, Ubuntu (a Linux-based OS based upon the Debian family of Linux) was utilized.
Taking Caffe, its dependencies include: (1) Compute Unified Device Architecture (CUDA), (2) Basic Linear Algebra Subprograms (BLAS), (3) Boost. Other optional dependencies include: (4) OpenCV, (5) IO Libraries, and (6)cuDNN.

* CUDA (a parallel computing platform and programming model for computing on Graphical Processing Units [GPUs]).



* BLAS (a specification, and de facto standard, for a set of low-level routines for performing basic vector and matrix operations [e.g., matrix multiplication]). Please note that the work is "based upon work supported by the National Science Foundation under Grant No. ASC-9313958 and DOE Grant No. DE-FG03-94ER25219." It is available at Netlib (http://www.netlib.org/blas/), which is a scientific computing software repository maintained by AT&T, Bell Laboratories, Oak Ridge National Laboratory, and University of Tennessee. The current standard BLAS library and LAPACK (a library of Fortran subroutines for solving the most commonly occurring problems in numerical linear algebra) development version is v3.9.0. The current release version is v3.8.0.

  * *Source: http://www.netlib.org/blas/*<br/>
  * *Source: http://www.netlib.org/blas/#_git_access*<br/>
  * *Source: https://github.com/Reference-LAPACK*<br/>
  * *Source: https://github.com/Reference-LAPACK/lapack*<br/>
  * *Source: https://github.com/Reference-LAPACK/lapack-release*<br/>
  
    * **Automatically Tuned Linear Algebra Software (ATLAS)** is an open source implementation of BLAS and LAPACK Application Programming Interfaces (APIs) for C and Fortran77. It tunes itself to the involved machine when compiled and automatically generates an optimized BLAS library. 

    * The Lab of Parallel Software and Computational Science of the Institute of Software, Chinese Academy of Sciences (ISCAS) developed **OpenBLAS**, which is an open source implementation of BLAS and LAPACK APIs with various optimizations for specific processor types. 

    * Intel Corporation developed **Math Kernel Library (MKL)**, which is a commercial implementation of, among other functions (e.g., sparse solvers, Fast Fourier Transforms), BLAS and LAPACK. 

    * When ATLAS, OpenBLAS, and MKL were tested on Revolution R Open (RRO), which is the enhanced open source distribution of R from Revolution Analytics that is geared for statistical analysis and data science, it was found that there is approximately a 60% improvement over the standard BLAS library and LAPACK when using ATLAS or OpenBLAS and a 70% improvement over the standard BLAS library and LAPACK when using MKL.

    * *Source: https://mran.microsoft.com/archives/rro-8.0.1*<br/>
    * *Source: http://blog.nguyenvq.com/blog/2014/11/10/optimized-r-and-python-standard-blas-vs-atlas-vs-openblas-vs-mkl/*<br/>

* Boost (pre-bundled libraries that work well with the C++ Standard Library). The current version is v1.75.0.

  * *Source: https://www.boost.org*

* OpenCV (an ML and computer vision library of programming functions). The current version is v4.5.1.

  * *Source: https://opencv.org/about/*<br/>
  * *Source: https://opencv.org/releases/*<br/>

* Input/Output (IO) Libraries include LevelDB (key-value store storage library) and Lightning Memory-Mapped Database (LMDB)(key-value storage library).

* NVIDIA CUDA Deep Neural Network (cuDNN) library (a GPU-accelerated library for deep neural networks).

## ONNX ##

ONNX was designed to be an open standard for ML interoperability. Among other examples, PyTorch and Caffe1/Caffe2 could be made interoperable, as there were numerous compatibility issues. However, this is not straightforward, as conversion from, let us say, PyTorch to ONNX may be confronted by issue (e.g., value mismatch).

*Source: https://github.com/pytorch/pytorch/issues/34731*

## PyTorch ##




## Caffe1/Caffe2 ##

Caffe requires, among other items: (1) Nvidia CUDA Compiler (NVCC), and (2) CUDA driver.





*Source: https://leimao.github.io/blog/CUDA-Driver-VS-CUDA-Runtime/*

Caffe can be accelerated by a drop-in integration of NVIDIA cuDNN (i.e., cuDNN Caffe). This acceleration will result in faster operations for the involved Caffe models.


```javascript
* install cuDNN 
* uncomment the USE_CUDNN := 1 flag in Makefile.config when installing Caffe. 
```
The current tested version is cuDNN v6. It has been reported that Caffe does not well support cuDNN v8. The workaround for certain versions of CUDA (e.g., CUDA v10.2) with cuDNN v8.0 is to bypass Caffe and use Openpose.

*Source: https://forums.developer.nvidia.com/t/cudnn-found-but-version-cant-be-deduced/145551/3*

For further reference please see: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

Within Openpose, find:

```javascript
file(READ {CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS) 
```
and change it to:

```javascript
file(READ {CUDNN_INCLUDE}/cudnn_version.h CUDNN_VERSION_FILE_CONTENTS)
```

*Source: https://forums.developer.nvidia.com/t/cudnn-found-but-version-cant-be-deduced/145551*
