# microtron

Inspired by [picotron](https://github.com/huggingface/picotron) and [nanoGPT](https://github.com/karpathy/nanoGPT), and while taking the course [Scratch to Scale: Large-Scale Training in the Modern World](https://maven.com/walk-with-code/scratch-to-scale), my plan is to build a clean, minimal, and education-oriented 3D parallel implementation (Data Parallel, Tensor Parallel, Pipeline Parallel) for pretraining a modern sparse model like gpt-oss from scratch (but with fewer parameters).

## Roadmap

* [ ] Implement a gpt-oss-like model architecture with smaller parameter settings (fewer layers, experts, and dimensions)
* [ ] Implement the single GPU training workflow for pretraining, including the dataloader, monitoring (throughput, loss, val\_loss, MFU), optimizer, etc.
* [ ] Extend to pipeline parallelism on 2 GPUs (pp = 2)
* [ ] Extend to data parallelism on 4 GPUs (pp = 2, dp = 2)
* [ ] Extend to tensor parallelism on 8 GPUs (pp = 2, dp = 2, tp = 2)
* [ ] Benchmark, profile, and optimize code performance
* [ ] Train a small gpt-oss like model from scratch using 3D parallelism with 8 GPUs
* [ ] Compile the results into a technical blog using [Quarto](https://quarto.org/)

## Notes

1. Decided to use uv for local dependency management and modal for remote GPU resources
2. Defined a smaller GPT-OSS setting with layer = 12, expert = 8, active experts = 4, hidden dims = 1440, FFN dims = 2880

## Acknowledgements

1. Thanks to [picotron](https://github.com/huggingface/picotron) for the minimal, easily digestible implementation of 3D parallelism that helped me learn.
2. Thanks to [nanoGPT](https://github.com/karpathy/nanoGPT) for the ideas and reproducible pretraining pipeline.
3. Thanks to [modal](https://modal.com/) for the $1000 in compute credits.
4. Thanks to [gpt-oss](https://arxiv.org/abs/2508.10925) for the model architecture.
5. Thanks to [Scratch to Scale](https://maven.com/walk-with-code/scratch-to-scale) for inspiring the whole project.