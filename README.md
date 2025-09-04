# microtron

Inspired by [picotron](https://github.com/huggingface/picotron) and [nanoGPT](https://github.com/karpathy/nanoGPT), and while taking the course [Scratch to Scale: Large-Scale Training in the Modern World](https://maven.com/walk-with-code/scratch-to-scale), my plan is to build a clean, minimal, and education-oriented 3D parallel implementation (Data Parallel, Tensor Parallel, Pipeline Parallel) for pretraining a simple dense model like GPT-2 from scratch.

## Roadmap

* [ ] Implement the GPT-2 model
* [ ] Implement the single GPU training workflow for pretraining, including the dataloader, monitoring (throughput, loss, val\_loss, MFU), optimizer, etc.
* [ ] Extend to pipeline parallelism on 2 GPUs (pp = 2)
* [ ] Extend to data parallelism on 4 GPUs (pp = 2, dp = 2)
* [ ] Extend to tensor parallelism on 8 GPUs (pp = 2, dp = 2, tp = 2)
* [ ] Benchmark, profile, and optimize code performance
* [ ] Train a GPT-2 model from scratch using 3D parallelism with 8 GPUs
* [ ] Compile the results into a technical blog using [Quarto](https://quarto.org/)

## Acknowledgements

1. [picotron](https://github.com/huggingface/picotron)
2. [nanoGPT](https://github.com/karpathy/nanoGPT)
