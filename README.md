# Open Neural Network Exchange (ONNX), PyTorch, Caffe2 #

The Open Neural Network Exchange (ONNX) is an open source Artificial Intelligence (AI) ecosystem/open standard for Machine Learning (ML) interoperability.

*Source: https://onnx.ai*

PyTorch is an open source ML framework based upon Torch (a scientific computing framework that supports GPU-based ML algorithms). 

*Source: http://torch.ch*

Convolutional Architecture for Fast Feature Embedding (CAFFE OR Caffe) is an open source ML and Deep Learning (DL) framework, which supports various DL architectures. Caffe2 builds upon Caffe (a.k.a. Caffe1).

*Source: https://developer.nvidia.com/blog/caffe2-deep-learning-framework-facebook/*

Depending upon the Operating Systems (OS) desired (e.g., Redhat, Ubuntu, etc.), Caffe/Caffe2 has various prerequisites/dependencies (e.g., https://caffe.berkeleyvision.org/installation.html). For the experimentation herein, Ubuntu (a Linux-based OS based upon the Debian family of Linux) was utilized.
Taking Caffe, its dependencies include: (1) Compute Unified Device Architecture (CUDA), (2) Basic Linear Algebra Subprograms (BLAS), (3) Boost. Other optional dependencies include: (4) OpenCV, (5) IO Libraries, and (6)cuDNN.

* CUDA



* BLAS (a specification, and de facto standard, for a set of low-level routines for performing basic vector and matrix operations [e.g., matrix multiplication]). Please note that the work is "based upon work supported by the National Science Foundation under Grant No. ASC-9313958 and DOE Grant No. DE-FG03-94ER25219." It is available at Netlib (http://www.netlib.org/blas/), which is a scientific computing software repository maintained by AT&T, Bell Laboratories, Oak Ridge National Laboratory, and University of Tennessee. The current standard BLAS library and LAPACK (a library of Fortran subroutines for solving the most commonly occurring problems in numerical linear algebra) development version is v3.9.0. The current release version is v3.8.0.

*Source: http://www.netlib.org/blas/*<br/>
*Source: http://www.netlib.org/blas/#_git_access*<br/>
*Source: https://github.com/Reference-LAPACK*<br/>
*Source: https://github.com/Reference-LAPACK/lapack*<br/>
*Source: https://github.com/Reference-LAPACK/lapack-release*<br/>

* Boost (pre-bundled libraries that work well with the C++ Standard Library). The current version is v1.75.0.

*Source: https://www.boost.org*

* OpenCV (an ML and computer vision library of programming functions). The current version is v4.5.1.

*Source: https://opencv.org/about/*
*Source: https://opencv.org/releases/*

* Input/Output (IO) Libraries include LevelDB (key-value store storage library) and Lightning Memory-Mapped Database (LMDB)(key-value storage library).

* NVIDIA CUDA Deep Neural Network (cuDNN) library (a GPU-accelerated library for deep neural networks).

**Automatically Tuned Linear Algebra Software (ATLAS)** is an open source implementation of BLAS and LAPACK Application Programming Interfaces (APIs) for C and Fortran77. It tunes itself to the involved machine when compiled and automatically generates an optimized BLAS library. 

The Lab of Parallel Software and Computational Science of the Institute of Software, Chinese Academy of Sciences (ISCAS) developed **OpenBLAS**, which is an open source implementation of BLAS and LAPACK APIs with various optimizations for specific processor types. 

Intel Corporation developed **Math Kernel Library (MKL)**, which is a commercial implementation of, among other functions (e.g., sparse solvers, Fast Fourier Transforms), BLAS and LAPACK. 

When ATLAS, OpenBLAS, and MKL were tested on Revolution R Open (RRO), which is the enhanced open source distribution of R from Revolution Analytics that is geared for statistical analysis and data science, it was found that there is approximately a 60% improvement over the standard BLAS library and LAPACK when using ATLAS or OpenBLAS and a 70% improvement over the standard BLAS library and LAPACK when using MKL.

*Source: https://mran.microsoft.com/archives/rro-8.0.1*<br/>
*Source: http://blog.nguyenvq.com/blog/2014/11/10/optimized-r-and-python-standard-blas-vs-atlas-vs-openblas-vs-mkl/*br/>






