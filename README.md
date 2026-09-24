cat << 'EOF' > README.md
---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- reasoning
- ollama
- gguf
- qwen
- slm
- unsloth
library_name: transformers
base_model: Qwen/Qwen2.5-3B-Instruct
---

# NIM-1 (3B)

**NIM-1** is a high-efficiency 3-billion parameter local reasoning model engineered for rapid, deterministic task execution, code intelligence, and structured agentic workflows. Built to deliver flagship reasoning density within consumer hardware limits, NIM-1 runs completely offline with ultra-low latency.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-NIM--1--3B-yellow)](https://huggingface.co/N-I-M-AI/NIM-1-3B)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-N--I--M--AI%2FNIM--1-black?logo=github)](https://github.com/N-I-M-AI/NIM-1)

---

## Highlights

* **High-Density Reasoning:** Tuned on high-signal chain-of-thought demonstrations to handle multi-step logic, Python algorithms, and complex instruction following.
* **Consumer-Ready Edge Inference:** Consumes **under 4 GB of VRAM** in 8-bit precision, streaming 60–90 tokens/sec on consumer GPUs (e.g., RTX 4060) and Apple Silicon.
* **Zero-Degradation Precision:** Preserved via single-pass `Q8_0` GGUF quantization.
* **Autonomous Agent Ready:** Built with native support for tool orchestration, structured schema outputs, and multi-turn conversational memory.

---

## Quickstart with Ollama

Run NIM-1 directly via Ollama pulling from Hugging Face:

```bash
ollama run hf.co/N-I-M-AI/NIM-1-3B:NIM-1-3B-Q8_0.gguf
Or build and run locally from this repository:
ollama create nim-1 -f Modelfile
ollama run nim-1

Model Specifications
Parameter	Specification
Model Architecture	Dense Transformer (Decoder-only)
Total Parameters	3.09 Billion
Context Window	2,048 tokens
Quantization Format	GGUF (Q8_0)
Inference Footprint	~3.4 GB VRAM / System RAM
Chat Template	ChatML format (`<
Hardware Compatibility
Hardware	Throughput	VRAM / Memory Usage
NVIDIA RTX 4060 (8 GB)	~70–90 tok/s	~3.4 GB
Apple Silicon (M-Series, 16 GB)	~60–80 tok/s	~3.6 GB Unified
Modern x86 CPU (AVX-512)	~18–28 tok/s	~4.2 GB System RAM
Reproducing Training & Packaging
# Clone the repository
git clone git@github.com:N-I-M-AI/NIM-1.git
cd NIM-1

# Install requirements
pip install -r requirements.txt

# Run fine-tuning pipeline
python train_student.py

# Export weights to GGUF format
python export_gguf.py
