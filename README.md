
# Optimizing Resource Efficiency in Gallager B LDPC Decoders

## Overview
This repository contains the implementation of various optimization strategies for Gallager B LDPC (Low-Density Parity-Check) decoders on GPU platforms, aiming to enhance throughput and efficiency. The project explores the use of massively parallel processing capabilities of modern GPUs to improve the performance of LDPC decoders, which are widely used in error-correction coding for digital communications.

## Abstract
This project leverages the GPU's parallel processing capabilities to optimize the Gallager B algorithm for LDPC decoding, addressing computational challenges and limitations of traditional CPU-based decoding methods. The implemented solution achieves a 12.16x speedup in decoding throughput and a significant reduction in memory latency by utilizing CUDA kernels, constant memory, and shared memory techniques.

## Methodology
The project includes the following key versions and optimization techniques:

- **Version-0: Baseline Implementation in C**
  - A basic implementation of the Gallager-B LDPC decoder using a hard decision bit-flipping algorithm.
  
- **Version-1: Enhanced LDPC Decoding on GPU**
  - Utilizes GPU parallel processing to handle variable-to-check and check-to-variable message updates, significantly improving throughput.

- **Version-2: Optimization with Constant Memory on GPU**
  - Improves scalability by storing frequently accessed, read-only data in constant memory to minimize access latency.

- **Version-3: Advanced GPU Acceleration**
  - Introduces shared memory, local variables, and asynchronous streaming to further enhance throughput and reduce memory latency.

- **Change Index Order Optimization**
  - Optimizes memory access patterns through matrix transposition techniques to achieve better performance.

## Experimental Setup
The optimizations were tested using a Tesla P100-PCIE GPU with 16 GB of global memory and CUDA Toolkit version 11.0. The performance metrics used include throughput (Mbps) and decoding latency (ms), with data generated using the Monte Carlo method.

## Results
The optimized LDPC decoders achieved:
- **12.16x speedup in decoding throughput**.
- **48% reduction in memory access latency** compared to traditional methods.
- Significant improvements in both throughput and latency, demonstrating the effectiveness of GPU acceleration techniques.

## Future Work
Potential future enhancements include:
- Integrating multi-codeword decoding algorithms to further increase throughput.
- Developing adaptive decoding algorithms that adjust based on real-time feedback from communication channels.

## Getting Started

### Prerequisites
To run the code, ensure you have the following prerequisites installed:
- NVIDIA GPU (Tesla P100 or equivalent)
- CUDA Toolkit (version 11.0 or later)
- C/C++ Compiler (GCC recommended)
- NVIDIA CUDA-enabled GPU drivers

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/SaiDhanyaa/Optimizing-LDPC-Decoders.git
   cd Optimizing-LDPC-Decoders
  ```
2. Build the project:
   ```bash
   make all
   ```
### Running the Experiments
Run the compiled executable for each version:

```bash
./run_version0
./run_version1
./run_version2
./run_version3
```
### Benchmarking
Use the provided benchmarking script to compare the performance of each version:

```bash
python benchmark.py
```
## Directory Structure
```
Optimizing-LDPC-Decoders/
│
├── src/                          # Source files for all versions
│   ├── version0.c                # Baseline C implementation
│   ├── version1_cuda.cu          # GPU version using global memory
│   ├── version2_cuda.cu          # GPU version using constant memory
│   ├── version3_cuda.cu          # GPU version with shared memory optimization
│   └── utils.cu                  # Utility functions
│
├── benchmarks/                   # Benchmarking scripts and results
│   ├── benchmark.py              # Python script to run benchmarks
│   └── results.csv               # Results from benchmarking different versions
│
├── docs/                         # Documentation files
│   ├── report.pdf                # Full project report
│   └── methodology.md            # Detailed methodology description
│
├── LICENSE                       # License file
└── README.md                     # This README file
```
## Contributing
Contributions are welcome! Please read the contributing guidelines for more details.

## References
[1] B. Unal, A. Akoglu, F. Ghaffari, and B. Vasić, “Hardware Implementation and Performance Analysis of Resource Efficient Probabilistic Hard Decision LDPC Decoders,” IEEE Transactions on Circuits and Systems I: Regular Papers, vol. 65, no. 9, pp. 3074-3084, Sept. 2018.

[2] G. Wang, M. Wu, Y. Sun and J. R. Cavallaro, “A massively parallel implementation of QC-LDPC decoder on GPU,” IEEE 9th Symposium on Application Specific Processors (SASP), 2011.

[3] K. K. Abburi, “A Scalable LDPC Decoder on GPU,” 24th International Conference on VLSI Design, 2011.

[4] S. Keskin and T. Kocak, “GPU-Based Gigabit LDPC Decoder,” IEEE Communications Letters, vol. 21, no. 8, pp. 1703-1706, Aug. 2017.

[5] R. Amiri and H. Mehrpouyan, “Multi-Stream LDPC Decoder on GPU of Mobile Devices,” IEEE 9th Annual Computing and Communication Workshop and Conference (CCWC), 2019.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any inquiries, please contact Dhanyapriya Somasundaram at dhanyapriyas@arizona.edu.
