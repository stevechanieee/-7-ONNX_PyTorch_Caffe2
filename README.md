# Open Neural Network Exchange, PyTorch, Caffe2 #

The Open Neural Network Exchange (ONNX) is an open source Artificial Intelligence (AI) ecosystem/open standard for Machine Learning (ML) interoperability.

*Source: https://onnx.ai*

PyTorch is an open source ML framework based upon Torch (a scientific computing framework that supports GPU-based ML algorithms). 

*Source: http://torch.ch*

Convolutional Architecture for Fast Feature Embedding (CAFFE OR Caffe) is an open source ML and Deep Learning (DL) framework, which supports various DL architectures. Caffe2 builds upon Caffe (a.k.a. Caffe1).

*Source: https://developer.nvidia.com/blog/caffe2-deep-learning-framework-facebook/*

Depending upon the Operating Systems (OS) desired (e.g., Redhat, Ubuntu, etc.), Caffe/Caffe2 has various prerequisites/dependencies (e.g., https://caffe.berkeleyvision.org/installation.html). For the experimentation herein, Ubuntu (a Linux-based OS based upon the Debian family of Linux) was utilized.
Taking Caffe, its dependencies include: (1) Compute Unified Device Architecture (CUDA), (2) Basic Linear Algebra Subprograms (BLAS), (3) Boost. Other optional dependencies include: (4) OpenCV, (5)

* CUDA



* BLAS (a specification, and de facto standard, for a set of low-level routines for performing basic vector and matrix operations [e.g., matrix multiplication]. etc.].The current LAPACK (a library of Fortran subroutines for solving the most commonly occurring problems in numerical linear algebra) version is v3.9.0.




Please note that the work is "based upon work supported by the National Science Foundation under Grant No. ASC-9313958 and DOE Grant No. DE-FG03-94ER25219." 

*Source: http://www.netlib.org/blas/*

* Boost (pre-bundled libraries that work well with the C++ Standard Library). The current version is v1.75.0.

*Source: https://www.boost.org*

* OpenCV (an ML and computer vision library of programming functions). The current version is v4.5.1.

*Source: https://opencv.org/about/*
*Source: https://opencv.org/releases/*










Basic Linear Algebra Subprograms (BLAS) is a specification that prescribes a set of low-level routines for performing common linear algebra operations such as vector addition, scalar multiplication, dot products, linear combinations, and matrix multiplication. They are the de facto standard low-level routines for linear algebra libraries; the routines have bindings for both C ("CBLAS interface") and Fortran ("BLAS interface"). Although the BLAS specification is general, BLAS implementations are often optimized for speed on a particular machine, so using them can bring substantial performance benefits. BLAS implementations will take advantage of special floating point hardware such as vector registers or SIMD instructions.









Value mismatch after convert models from PyTorch to ONNX

*Source: https://github.com/pytorch/pytorch/issues/34731*




When updating Caffe, it’s best to make clean before re-compiling.

Caffe has several dependencies:

CUDA is required for GPU mode. BLAS via ATLAS, MKL, or OpenBLAS; Caffe requires BLAS as the backend of its matrix and vector computations. There are several implementations of this library. The choice is yours: Boost >= 1.55 protobuf, glog, gflags, hdf5

Other dependencies:

OpenCV >= 2.4 including 3.0 IO libraries: lmdb, leveldb (note: leveldb requires snappy) cuDNN for GPU acceleration (v6) Pycaffe and Matcaffe interfaces have their own natural needs.

For Python Caffe: Python 2.7 or Python 3.3+, numpy (>= 1.7), boost-provided boost.python For MATLAB Caffe: MATLAB with the mex compiler.

cuDNN Caffe: for fastest operation Caffe is accelerated by drop-in integration of NVIDIA cuDNN. To speed up your Caffe models, install cuDNN then uncomment the USE_CUDNN := 1 flag in Makefile.config when installing Caffe. Acceleration is automatic. The current version is cuDNN v6;

Caffe requires the CUDA nvcc compiler to compile its GPU code and CUDA driver for GPU operation.

Deep learning is a subset of AI and machine learning that uses multi-layered artificial neural networks to deliver state-of-the-art accuracy in tasks such as object detection, recognition

automatically learn representations from data such as images, video or text, without introducing hand-coded rules or human domain knowledge.

can learn directly from raw data and can increase their predictive accuracy when provided with more data.

accelerated deep learning frameworks, researchers and data scientists can significantly speed up deep learning training, that could otherwise take days and weeks to just hours and days.

Developing AI applications start with training deep neural networks with large datasets. GPU-accelerated deep learning frameworks offer flexibility to design and train custom deep neural networks and provide interfaces to commonly-used programming languages such as Python and C/C++. Every major deep learning framework such as TensorFlow, PyTorch, and others, are already GPU-accelerated, so data scientists and researchers can get productive in minutes without any GPU programming.

Every major deep learning framework such as Caffe2, Chainer, Microsoft Cognitive Toolkit, MxNet, PaddlePaddle, Pytorch and TensorFlow rely on Deep Learning SDK libraries to deliver high-performance multi-GPU accelerated training.

DLE requires a CUDA

You probably don’t need to downgrade the CUDA 11 installed in your system. As explained here 615, conda install pytorch torchvision cudatoolkit=10.2 -c pytorch will install CUDA 10.2 and cudnn binaries within the Conda environment, so the system-installed CUDA 11 will not be used at all.

I recently installed ubuntu 20.04 and Nvidia driver 450. It took me a while to realize that I didn’t have to build pytorch from source just because I have CUDA 11 in my system.

Also, if you do actually want to try CUDA 11, easiest way is to make sure you have a sufficiently new driver and run the PyTorch NGC docker container. The latest 20.06 container has PyTorch 1.6, CUDA 11, and cuDNN 8, unfortunately cuDNN is an release candidate with some fairly significant performance regressions right now, not always the best idea to be bleeding edge

There is unfortunately no workaround for this, as compute capability 3.0 and 3.2 were dropped in CUDA11 and 3.5, 3.7, and 5.0 were deprecated

The conda binaries and pip wheels ship with their CUDA (cudnn, NCCL, etc.) runtime, so you don’t need a local CUDA installation to use native PyTorch operations. However, you would have to install a matching CUDA version, if you want to build PyTorch from source or build custom CUDA extensions.

Caffe2 improves Caffe 1.0 in a series of directions: first-class support for large-scale distributed training mobile deployment new hardware support (in addition to CPU and CUDA) flexibility for future directions such as quantized computation stress tested by the vast scale of Facebook applications

Caffe

https://docs.nvidia.com/deeplearning/frameworks/caffe-release-notes/rel_19-04.html

Converting Models from Caffe to Caffe2

Getting Caffe1 Models for Translation to Caffe2

Converting from Torch

https://caffe2.ai/docs/caffe-migration.html https://github.com/facebookarchive/fb-caffe-exts#torch2caffe We have provided a command line python script tailor made for this purpos caffe_translator.py - This script has built-in translators for common layers. The tutorial mentioned above implements this same script, so it may be helpful to review the tutorial to see how the script can be utilized. You can also call the script directly from command line.

caffe_translator_test.py - This a large test that goes through the translation of the BVLC caffenet model, runs an example through the whole model, and verifies numerically that all the results look right. In default, it is disabled unless you explicitly want to run it.
