# Pseudo Attack Adversarial Suffix Generation

[![Paper](https://img.shields.io/badge/Paper-ACL%20Anthology-blue)](https://aclanthology.org/2024.woah-1.12/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

> ⚠️ **WARNING**: This repository contains code for generating adversarial examples that may include harmful or inappropriate content. The code is intended for research purposes only to improve AI safety mechanisms.

This repository contains the implementation of the Pseudo Attack method described in our paper ["Robust Safety Classifier Against Jailbreaking Attacks: Adversarial Prompt Shield"](https://aclanthology.org/2024.woah-1.12/) (WOAH 2024). This is part of our work on developing robust safety classifiers against jailbreaking attacks.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{kim2024robust,
  title={Robust Safety Classifier Against Jailbreaking Attacks: Adversarial Prompt Shield},
  author={Kim, Jinhwa and Derakhshan, Ali and Harris, Ian},
  booktitle={Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024)},
  pages={159--170},
  year={2024}
}
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/aliderakhsh/pseudo-attack-suffix-generation.git
cd pseudo-attack-suffix-generation
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script accepts several arguments to customize the generation process:

```bash
python main.py \
    --model_path /path/to/model \
    --template_name vicuna-7b \
    --input_file ./data/advbench/harmful_behaviors_index_1.csv \
    --output_dir ./output/ \
    --num_steps 50 \
    --batch_size 256 \
    --topk 32 \
    --device cuda
```

### Arguments

- `--model_path`: Path to the language model (default: vicuna-7b-v1.3)
- `--template_name`: Conversation template name (options: falcon-7b-instruct, guanaco-7b, guanaco-7B-HF, llama-2)
- `--input_file`: Path to input CSV file containing test cases
- `--output_dir`: Directory for saving output files
- `--num_steps`: Number of optimization steps (default: 50)
- `--batch_size`: Batch size for processing (default: 256)
- `--topk`: Top k tokens to consider (default: 32)
- `--allow_non_ascii`: Flag to allow non-ASCII tokens
- `--device`: Device to run the model on (cuda/cpu)

## Method Description

The Pseudo Attack method implements an efficient approach to generating adversarial suffixes for training safety classifiers. Key features include:

1. Parallel token optimization instead of sequential modification
2. Simultaneous gradient evaluation across the token space
3. Multiple candidate generation based on loss metrics
4. Reduced computational overhead compared to traditional GCG approaches

This implementation is specifically designed to work with the APS (Adversarial Prompt Shield) safety classifier system.

## Acknowledgments

Some parts of our implementation build upon the work presented in ["Universal and Transferable Adversarial Attacks on Aligned Language Models"](https://arxiv.org/abs/2307.15043) by Zou et al.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about the code or paper, please open an issue in this repository or contact the authors directly.