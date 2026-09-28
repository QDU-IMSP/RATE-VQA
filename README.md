# RATE-VQA: Reliability-Aware Temporal Evidence Learning for AIGC Video Quality Assessment

## English

This repository provides the official implementation of **RATE-VQA**, a reliability-aware framework for **AI-generated video quality assessment (AIGC-VQA)**.

RATE-VQA enhances vision-language models with explicit temporal evidence extracted from motion consistency analysis. By introducing correspondence-aware reliability modeling, the framework distinguishes between actual temporal artifacts and unreliable measurements caused by invalid correspondences, enabling more robust video quality assessment.

### Overview

AI-generated videos may contain temporal issues such as:
- Temporal structural drift
- Motion discontinuity
- High-frequency flickering

while maintaining visually plausible individual frames. RATE-VQA addresses these challenges by incorporating motion-based temporal evidence and reliability-aware evidence calibration into a vision-language quality assessment framework.

### Features

- Temporal evidence extraction based on optical flow consistency analysis
- Motion-compensated residual modeling for temporal artifacts
- Correspondence-aware reliability estimation
- Reliability-guided adaptive evidence calibration
- Vision-language model based quality prediction

### Installation

```bash
git clone https://github.com/your_username/RATE-VQA.git
cd RATE-VQA
pip install -r requirements.txt
