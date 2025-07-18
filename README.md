# Medical Text Simplification: From Jargon Detection to Automated Simplification

## Overview
This repository contains the code for my MSc thesis on medical text simplification, focusing on jargon detection and automated simplification using large language models. The thesis investigates whether explicitly identifying and incorporating medical jargon into LLM prompts can enhance the quality of medical text simplification.

## Project Structure
The codebase is organized into three main parts, corresponding to the three chapters of the thesis:

### Part 1: Biomedical Jargon Detection
`src/part1/` - Reproducibility study of MedReadMe and evaluation across datasets
- `medreadme.py` - Implementation of the MedReadMe jargon detection pipeline
- `plaba.py` - Implementation for the PLABA dataset jargon detection
- `transfer_learning.py` - Cross-dataset transfer learning experiments
- `error_analysis.py` - Analysis of model errors across jargon categories
- `visualize_jargons.py` - Visualization tools for jargon detection results
- `labeling_tool.py` - Custom annotation tool for MedReadMe scheme
- `data_statistic.py` - Dataset statistics and analysis

### Part 2: Jargon-Aware Prompting for Text Simplification
`src/part2/` - Evaluation of jargon-aware prompting strategies
- `simplify.py` - Main simplification pipeline
- `models.py` - Model loading and configuration
- `prompts.py` - Different prompting strategies implementation
- `finetune.py` - Model fine-tuning utilities
- `test_llama.py` & `test_medicine_llama.py` - Evaluation scripts for different models
- `test_metrics.py` - Metrics computation and analysis
- `analyze_dataset.py` - PLABA dataset analysis

### Part 3: CLEF 2025 SimpleText Track Participation
`src/part3/` - Implementation for the CLEF 2025 SimpleText track (Task 1)
- `llama_jargon_simplify.py` - Jargon-aware simplification for CLEF
- `clef2025_batch.py` - Batch processing for CLEF dataset
- `clef2025_split_data.py` - Data preparation utilities
- `clef2025_combine_results.py` - Results aggregation

### Job Files
`src/job_files/` - Slurm job submission scripts for running experiments on HPC
- Various `.job` files for different experimental configurations

## Setting up the environment

You can set up the environment using the following commands:

```bash
virtualenv medplaba
source medplaba/bin/activate
pip install -r requirements.txt
```

## Running Experiments

### Jargon Detection (Part 1)
```bash
# MedReadMe jargon detection
python src/part1/medreadme.py --model_name roberta

# PLABA jargon detection
python src/part1/plaba.py --experiment_type 1a

# Transfer learning experiments
python src/part1/transfer_learning.py
```

### Text Simplification with Jargon-Aware Prompts (Part 2)
```bash
# Run simplification with Llama model
python src/part2/test_llama.py

# Run simplification with Medicine-Llama model
python src/part2/test_medicine_llama.py

# Analyze simplification metrics
python src/part2/test_metrics.py
```

### CLEF 2025 SimpleText Track (Part 3)
```bash
# Run CLEF 2025 simplification
python src/part3/clef2025_batch.py

# Combine results
python src/part3/clef2025_combine_results.py
```

## Analysis Tools

### Data Statistics
Generate dataset statistics and visualizations:
```bash
python src/part1/data_statistic.py
```

### Error Analysis
Analyze model predictions and error patterns:
```bash
python src/part1/error_analysis.py --model_name roberta --cls_type 3-cls
```

### Jargon Visualization
Visualize jargon detection results:
```bash
python src/part1/visualize_jargons.py --model_name roberta --cls_type 3-cls --num_samples 5
```

## Citation
If you use this code in your research, please cite:
```
@mastersthesis{papandreou2025medical,
  title={Medical Text Simplification: From Jargon Detection to Automated Simplification},
  author={Papandreou-Lazos, Panagiotis Taiki},
  year={2025},
  school={University of Amsterdam}
}
```
