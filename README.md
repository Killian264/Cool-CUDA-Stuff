# Cool-CUDA-Stuff
CSC 410 Programming Challenge using CUDA

# How To RUN

* Download CUDA Here  
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal

* Download Visual Studio Here  
https://visualstudio.microsoft.com/downloads/


# VS 2017 or Later 
CUDA is not fully supported past 2017, switch from Build + IntelliSense to Build Only

# C_Compute launch failed: invalid configuration argument
* Lower number of threads, my GPU has 2,176 Cores but I can only run with around 1000.
* Use Threads in multiples of 32.
