# MatMul
Matrix multiplication CPU + GPU compare
## Prerequisites
1. Microsoft visual studio 19
2. Nvidia GPU (cuda SUPPORT)
3. CUDA Toolkit 11
## Build and Run
1. Make new CUDA-project.
2. Include in the project "matM.cu".
## System configuration
| Name  | Values  |
|-------|---------|
| CPU  | Intel® Pentium® G3430 |
| RAM  | 4 GB DDR3 |
| GPU  | GeForce GTX 750 Ti 2GB |
| OS   | Windows 10 64-bit  |
## Results

Average results after 100 times of runs. Matrix elements type is float.

|    Size     |          CPU        |         GPU       | Acceleration |
|-------------|---------------------|-------------------|--------------|
| 64 х 64   | 1 ms               | 0.16 ms            |    6.25      |
| 128 х 128   | 17 ms               | 1.02 ms            |    16.6      |
| 256 х 256   | 113 ms               | 6.92 ms            |    16.33      |
| 512 х 512   | 884 ms              | 65.96 ms             |    13.4      |
| 1024 х 1024 | 23109 ms   | 388.36 ms            |    59.5      |

