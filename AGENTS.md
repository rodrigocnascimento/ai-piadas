# AGENTS.md — AI Piada

Fine-tuned joke generator using Mistral-7B with QLoRA (4-bit quantization).

## Entry Points

| Script | Purpose |
|--------|---------|
| `generate_jokes.py` | Interactive joke generator (uses trained model) |
| `finetune_jokes.py` | Fine-tune Mistral on joke dataset |
| `prepare_data.py` | Split JSONL dataset into train/validation sets |

## Environment

- **Python**: 3.x with existing virtual environment (venv at repo root)
- **GPU Required**: NVIDIA with CUDA support (tested on RTX 3060)
- **Activate**: `source bin/activate` or `source venv/bin/activate`

## Key Dependencies

- `torch==2.5.1+cu121` — PyTorch with CUDA 12.1
- `transformers==4.52.4` — Hugging Face models
- `peft==0.15.2` — LoRA adapters
- `trl==0.18.2` — SFT trainer
- `bitsandbytes==0.46.0` — 4-bit quantization
- `datasets==3.6.0` — Dataset handling

## Data Format

Dataset files are JSON Lines with this schema:
```json
{"instruction": "Gere uma piada sobre", "input": "futebol", "output": "Piada gerada..."}
```

Files:
- `data/jokes_bigger_dataset.jsonl` — Full dataset (~6500 examples)
- `data/jokes_dataset.jsonl` — Smaller dataset
- `data/jokes_train_dataset/` — HuggingFace format (auto-generated)
- `data/jokes_val_dataset/` — HuggingFace format (auto-generated)

## Model Configuration

- **Base Model**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Adapter Rank**: 16, Alpha: 32
- **Target Modules**: q_proj, v_proj
- **Training**: 3 epochs, batch size 4, lr 2e-4

## Output Structure

```
data/results_jokes_bigger_model/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # Trained adapter weights
├── checkpoint-*/            # Intermediate checkpoints (2000, 4000, 6000 steps)
└── tokenizer.json           # Saved tokenizer
```

## Commands

```bash
# Activate environment
source bin/activate

# Generate jokes interactively
python generate_jokes.py

# Fine-tune (requires ~8GB VRAM, takes several hours)
python finetune_jokes.py

# Prepare data splits
python prepare_data.py

# Test PyTorch/CUDA
python py-torch-test.py
```

## Memory Requirements

- **Inference**: ~6GB VRAM (4-bit quantized)
- **Training**: ~8-10GB VRAM (QLoRA with batch size 4)

## Important Notes

- Prompt format: `<s>[INST] {instruction}: {input} [/INST] {output}</s>`
- Do not commit model weights or large datasets (see `.gitignore`)
- Generated adapters are small (~100MB); base model is downloaded from HuggingFace on first run
- Checkpoints save per epoch; best model loaded automatically at end
