# microtron

Inspired by [picotron](https://github.com/huggingface/picotron) and [nanoGPT](https://github.com/karpathy/nanoGPT), and while taking the course [Scratch to Scale: Large-Scale Training in the Modern World](https://maven.com/walk-with-code/scratch-to-scale), my plan is to build a clean, minimal, and education-oriented 3D parallel implementation (Data Parallel, Tensor Parallel, Pipeline Parallel) for pretraining a modern sparse model like gpt-oss from scratch (but with fewer parameters).

## Motivation

In the [Reproduce GPT-2 project](https://www.youtube.com/watch?v=l8pRSuU81PU), we have already seen that, by using modern compute, it is possible to reproduce the GPT-2 model from scratch at a lower cost compared to the original OpenAI implementation. However, here is the new challenge. With the introduction of various architectural changes recently, the gpt-oss models now incorporate many modern techniques that improve efficiency, such as RMSNorm, sliding window attention, GQA, and MoE. This raises the question: Is it possible to achieve the same performance level as the GPT-2 model series, or even surpass it, at an even lower cost?
This project aims to explore that possibility. I plan to initialize a smaller gpt-oss model with the following configuration: 20 layers, 4 experts, 1 active experts, hidden size of 512, and intermediate size of 512. The model size is 363M parameters, which is close to GPT-2 Medium (380M).
By the end of the project, my goal is to outperform the GPT-2 Medium model in terms of training cost and efficiency, while training the new model from scratch.

## Roadmap

* [ ] Follow the spirit of nanoGPT to implement a clean and minimal GPT-OSS model architecture from scratch, referencing the official implementations from Hugging Face and OpenAI.
* [ ] Implement a single-GPU training workflow for pretraining, including the dataloader, monitoring (throughput, loss, val\_loss, MFU), optimizer, and more.
* [ ] Extend to distributed data parallelism on 1 to 8 GPUs and try to reproduce GPT-2 level results.
* [ ] Extend to pipeline parallelism on 2 GPUs (pp = 2)
* [ ] Extend to data parallelism on 4 GPUs (pp = 2, dp = 2)
* [ ] Extend to tensor parallelism on 8 GPUs (pp = 2, dp = 2, tp = 2)
* [ ] Benchmark, profile, and optimize code performance like using huggingface kernels, liger kernels
* [ ] Train a small gpt-oss like model from scratch using 3D parallelism with 8 GPUs
* [ ] Compile the results into a technical blog using [Quarto](https://quarto.org/)

## Future Directions

1. Try to follow the llm.c solution using pure C and CUDA with the modern FineWeb dataset from Hugging Face. It was able to reproduce GPT-2 (124M) in only 90 minutes for \$20 [Reproducing GPT-2 (124M) in llm.c in 90 minutes for \$20](https://github.com/karpathy/llm.c/discussions/481). I believe that by following this pure C solution, we might be able to achieve the same result at a lower cost, or possibly achieve higher performance at the same cost due to architectural advantages.
2. Try to integrate context parallelism for long context training.

## Acknowledgements

1. Thanks to [picotron](https://github.com/huggingface/picotron) for the minimal, easily digestible implementation of 3D parallelism that helped me learn.
2. Thanks to [nanoGPT](https://github.com/karpathy/nanoGPT) for the ideas and reproducible pretraining pipeline.
3. Thanks to [modal](https://modal.com/) for the $1000 in compute credits.
4. Thanks to [gpt-oss](https://arxiv.org/abs/2508.10925) for the model architecture.
5. Thanks to [Scratch to Scale](https://maven.com/walk-with-code/scratch-to-scale) for inspiring the whole project.

## Reference

1. [nanoGPT](https://github.com/karpathy/nanoGPT)
2. [build nanoGPT](https://github.com/karpathy/build-nanogpt)
3. [picotron](https://github.com/huggingface/picotron)
4. [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
5. [From GPT-2 to gpt-oss: Analyzing the Architectural Advances](https://sebastianraschka.com/blog/2025/from-gpt-2-to-gpt-oss.html)
6. [How LLM architecture has evolved from GPT-2 to gpt-oss](https://modal.com/blog/gpt-oss-arch)
7. [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)
8. [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925)
9. [official openai gpt-oss implementation](https://github.com/openai/gpt-oss)
10. [huggingface-transformers-gpt-oss-implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py)
11. https://huggingface.co/unsloth/gpt-oss-20b-BF16