# DAC-SE1: HIGH-FIDELITY SPEECH ENHANCEMENT VIA DISCRETE AUDIO TOKENS
Recent autoregressive transformer-based speech enhancement (SE) methods have shown promising results by leveraging advanced semantic understanding and contextual modeling of speech. However, these approaches often rely on complex multi-stage pipelines and low sampling rate codecs, limiting them to narrow and task-specific speech enhancement. In this work, we introduce DAC-SE1, a simplified language model-based SE framework leveraging discrete high-resolution audio representations; DAC-SE1 preserves fine-grained acoustic details while maintaining semantic coherence. Our experiments show that DAC-SE1 surpasses state-of-the-art autoregressive SE methods on both objective perceptual metrics and in a MUSHRA human evaluation. We release our codebase and model checkpoints to support further research in scalable, unified, and high-quality speech enhancement

## Installation

Requires Python 3.10+

```bash
pip install -r requirements.txt
```

## Usage

### Inference

#### Python Snippet

You can use the denoiser directly after cloning the repo:

```python
import torch
from transformers import LlamaForCausalLM, LogitsProcessorList
from inference import DACTools, DACTokenizer, DACConstrainedLogitsProcessor
import re

# Initialize DAC tools for audio encoding/decoding
dac_tools = DACTools()
tokenizer = DACTokenizer(num_layers=9, codebook_size=1024)

# Load denoiser model
model_path = "disco-eth/DAC-SE1 "
model = LlamaForCausalLM.from_pretrained(model_path)
model = model.to('cuda')
model.eval()

# Load noisy audio and convert to tokens
noisy_tokens = dac_tools.audio_to_tokens('input.wav')

# Prepare input for model
token_ids = tokenizer.encode(noisy_tokens, add_special_tokens=False)
input_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.start_clean_token_id]
input_tensor = torch.tensor([input_ids]).cuda()

# Generate clean tokens
num_tokens = len(re.findall(r'<\|s\d+_c\d\|>', noisy_tokens))
logits_processor = LogitsProcessorList([
    DACConstrainedLogitsProcessor(tokenizer=tokenizer, min_tokens=num_tokens)
])

with torch.no_grad():
    outputs = model.generate(
        input_tensor,
        max_new_tokens=num_tokens,
        min_new_tokens=num_tokens,
        logits_processor=logits_processor,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

# Extract generated tokens
generated_ids = outputs[0, len(input_ids):].tolist()
generated_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

# Convert tokens back to audio
valid_tokens = re.findall(r'<\|s\d+_c\d\|>', generated_output)
if valid_tokens:
    remainder = len(valid_tokens) % 9
    if remainder != 0:
        valid_tokens = valid_tokens[:len(valid_tokens) - remainder]
    denoised_tokens = "".join(valid_tokens)
    tokens = dac_tools.string_to_tokens(denoised_tokens)
    clean_audio = dac_tools.tokens_to_audio(tokens)

# Save denoised audio
import soundfile as sf
sf.write('output.wav', clean_audio, dac_tools.sample_rate)
```

Denoise audio files using a pretrained model through the shell script:

```bash
accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    inference.py \
    --model_path "disco-eth/DAC-SE1" \
    --input_dir "./noisy_audio" \
    --output_dir "./denoised_audio" \
    --chunk_duration 4.0 \
    --overlap_ratio 0.05
```



**Arguments:**
| Argument | Description |
|----------|-------------|
| `--model_path` | Path to pretrained model (local or HuggingFace hub) |
| `--input_dir` | Directory containing noisy WAV files |
| `--output_dir` | Directory for denoised output files |
| `--chunk_duration` | Duration of audio chunks in seconds (default: 4.0) |
| `--overlap_ratio` | Overlap ratio between chunks for crossfading (default: 0.05) |

### Training

Train a new model or fine-tuned an existing one on paired clean/noisy audio:

```bash
accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    train.py \
    --model_size 1B \
    --train_dir "./train_data" \
    --eval_dir "./eval_data" \
    --output_dir "./checkpoints" \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_steps 10000 \
    --warmup_steps 200 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --seed 42 \
    --bf16 \
    --gradient_checkpointing \
    --max_duration 4
```

**Data format:** Place paired audio files in the training directory with naming convention:
- Noisy: `*noisy*.wav` or `*dirty*.wav`
- Clean: `*clean*.wav`

Example: `001_noisy.wav` / `001_clean.wav`

**Model sizes:**
| Size | Hidden | Layers |
|------|--------|--------|
| ~250M | 1024 | 18 |
| ~1B | 1536 | 24 |
| ~2B | 2048 | 29 |
| ~4B | 2816 | 32 |

### Dataset Creation
Refer to the script ```create_data_SE.py```. For more detailed information refer to the following [HackMD article](https://hackmd.io/@AdonisAsonitis/B1j6wVJlbe).

### Checkpoints
We release a pre-trained checkpoint in the following huggingface repository: https://huggingface.co/disco-eth/DAC-SE1 

## Citation

```bibtex
@misc{lanzendörfer2025highfidelityspeechenhancementdiscrete,
      title={High-Fidelity Speech Enhancement via Discrete Audio Tokens}, 
      author={Luca A. Lanzendörfer and Frédéric Berdoz and Antonis Asonitis and Roger Wattenhofer},
      year={2025},
      eprint={2510.02187},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2510.02187}, 
}
```

## License
MIT License
