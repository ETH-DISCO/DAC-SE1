import re
import os
import glob
import argparse
import random
import torch
import numpy as np
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DacModel,
    AutoProcessor,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train DAC Denoiser")
    parser.add_argument("--model_size", type=str, choices=["250M", "1B", "2B", "4B"], required=True, help="Model size")
    parser.add_argument("--train_dir", type=str, required=True, help="Directory with training audio pairs")
    parser.add_argument("--eval_dir", type=str, help="Directory with evaluation audio pairs")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to resume from")
    
    parser.add_argument("--per_device_batch_size", type=int, required=True, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, required=True, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, required=True, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, required=True, help="Max gradient norm")
    parser.add_argument("--max_steps", type=int, required=True, help="Max training steps")
    parser.add_argument("--warmup_steps", type=int, required=True, help="Warmup steps")
    parser.add_argument("--eval_steps", type=int, required=True, help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, required=True, help="Checkpoint save frequency")
    parser.add_argument("--save_total_limit", type=int, required=True, help="Maximum number of checkpoints to keep")
    parser.add_argument("--logging_steps", type=int, required=True, help="Logging frequency")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Pin memory in dataloader")
    parser.add_argument("--dataloader_drop_last", action="store_true", help="Drop last incomplete batch")
    parser.add_argument("--max_duration", type=float, help="Maximum duration in seconds (skip pairs exceeding this)")
    
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6, help="RMS norm epsilon")
    parser.add_argument("--rope_theta", type=float, default=100000.0, help="RoPE theta parameter")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Attention dropout")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="Hidden dropout probability")
    parser.add_argument("--initializer_range", type=float, default=0.005, help="Initializer range")
    parser.add_argument("--hidden_act", type=str, default="silu", help="Hidden activation function")
    parser.add_argument("--attention_bias", action="store_true", help="Use attention bias")
    parser.add_argument("--tie_word_embeddings", action="store_true", default=True, help="Tie word embeddings")
    
    return parser.parse_args()


class DACTools:
    def __init__(self):
        self.model = DacModel.from_pretrained("descript/dac_44khz")
        self.processor = AutoProcessor.from_pretrained("descript/dac_44khz")
        self.num_codebooks = 9
        self.sample_rate = 44100
        self.timesteps_per_second = 86.13
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def audio_to_tokens(self, audio_path):
        audio, sr = sf.read(audio_path)
        
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        inputs = self.processor(raw_audio=audio, sampling_rate=self.sample_rate, return_tensors="pt")
        
        with torch.no_grad():
            encoder_outputs = self.model.encode(inputs["input_values"].to(self.device))
            codes = encoder_outputs.audio_codes
        
        flattened_tokens = codes[0].T.flatten().tolist()
        return self.tokens_to_string(flattened_tokens)
    
    def get_duration(self, token_string):
        dac_tokens = re.findall(r'<\|s\d+_c\d\|>', token_string)
        num_timesteps = len(dac_tokens) // self.num_codebooks
        return num_timesteps / self.timesteps_per_second
    
    def tokens_to_string(self, tokens):
        out = []
        for i in range(0, len(tokens), self.num_codebooks):
            group = tokens[i:i+self.num_codebooks]
            for codebook_idx, token_id in enumerate(group, start=1):
                out.append(f"<|s{token_id}_c{codebook_idx}|>")
        return "".join(out)


class DACTokenizer:
    def __init__(self, num_layers=9, codebook_size=1024):
        self.bos_token = "<|bos|>"
        self.pad_token = "<|pad|>"
        self.eos_token = "<|eos|>"
        self.start_clean_token = "<|start_clean|>"
        self.unk_token = "<|unk|>"

        self.special_tokens = [
            self.bos_token,
            self.pad_token,
            self.eos_token,
            self.start_clean_token,
            self.unk_token,
        ]
        
        self.codebook_tokens = [
            f"<|s{idx}_c{layer}|>"
            for layer in range(1, num_layers + 1)
            for idx in range(codebook_size)
        ]
        self.vocab_tokens = self.special_tokens + self.codebook_tokens

        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab_tokens)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        self.bos_token_id = self.token_to_id[self.bos_token]
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.start_clean_token_id = self.token_to_id[self.start_clean_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

        self.token_regex = re.compile(r"<\|s\d+_c\d\|>")

    def encode(self, text, add_special_tokens=True):
        current_pos = 0
        tokens = []
        
        while current_pos < len(text):
            matched = False
            for special_token in self.special_tokens:
                if text[current_pos:].startswith(special_token):
                    tokens.append(special_token)
                    current_pos += len(special_token)
                    matched = True
                    break
            
            if not matched:
                match = self.token_regex.search(text[current_pos:])
                if match:
                    token = match.group(0)
                    tokens.append(token)
                    current_pos += len(token)
                else:
                    current_pos += 1
        
        token_ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [self.id_to_token.get(i, self.unk_token) for i in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return "".join(tokens)
    
    @property
    def vocab_size(self):
        return len(self.vocab_tokens)
    
    def save_pretrained(self, save_directory, **kwargs):
        import json
        os.makedirs(save_directory, exist_ok=True)
        
        config = {
            "num_layers": 9,
            "codebook_size": 1024,
            "vocab_size": self.vocab_size,
            "bos_token": self.bos_token,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
            "start_clean_token": self.start_clean_token,
            "unk_token": self.unk_token,
            "tokenizer_type": "DACTokenizer"
        }
        
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        with open(os.path.join(save_directory, "vocab.json"), "w") as f:
            json.dump(self.token_to_id, f, indent=2)
        

class DACDataset(Dataset):
    def __init__(self, sequences, tokenizer):
        self.sequences = sequences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        token_ids = self.tokenizer.encode(seq)
        return torch.tensor(token_ids, dtype=torch.long)


def collate_fn(batch, pad_token_id, start_clean_token_id):
    if not batch:
        return {
            "input_ids": torch.tensor([]),
            "labels": torch.tensor([]),
            "attention_mask": torch.tensor([])
        }
    
    input_sequences, label_sequences = [], []
    
    for sequence in batch:
        sequence = torch.as_tensor(sequence, dtype=torch.long)

        try:
            start_clean_pos = (sequence == start_clean_token_id).nonzero(as_tuple=True)[0].item()
        except (IndexError, RuntimeError):
            start_clean_pos = None

        if start_clean_pos is not None:
            input_sequences.append(sequence)
            labels = sequence.clone()
            labels[:start_clean_pos + 1] = -100
            label_sequences.append(labels)
        else:
            input_sequences.append(sequence)
            label_sequences.append(sequence.clone())

    max_len = max(len(x) for x in input_sequences)
    batch_size = len(input_sequences)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_sequences, label_sequences)):
        input_ids[i, :len(inp)] = inp
        labels[i, :len(lab)] = lab
        attention_mask[i, :len(inp)] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def find_audio_pairs(directory):
    dirty_files = []
    clean_files = []
    
    for ext in ['*.wav', '*.WAV', '*.wave', '*.WAVE']:
        dirty_files.extend(glob.glob(os.path.join(directory, f"*dirty*{ext}")))
        dirty_files.extend(glob.glob(os.path.join(directory, f"*noisy*{ext}")))
        clean_files.extend(glob.glob(os.path.join(directory, f"*clean*{ext}")))
    
    pairs = []
    for dirty_file in dirty_files:
        base_name = os.path.basename(dirty_file)
        clean_name = base_name.replace("dirty", "clean").replace("noisy", "clean")
        clean_file = os.path.join(directory, clean_name)
        
        if not os.path.exists(clean_file):
            for clean_file_candidate in clean_files:
                if os.path.basename(clean_file_candidate).replace("clean", "") == base_name.replace("dirty", "").replace("noisy", ""):
                    clean_file = clean_file_candidate
                    break
        
        if os.path.exists(clean_file):
            pairs.append((dirty_file, clean_file))
    
    return pairs


def get_model_config(model_size, vocab_size, args):
    configs = {
        "250M": {
            "hidden_size": 1024,
            "intermediate_size": 3000,
            "num_hidden_layers": 18,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 8192,
        },
        "1B": {
            "hidden_size": 1536,
            "intermediate_size": 6144,
            "num_hidden_layers": 24,
            "num_attention_heads": 24,
            "num_key_value_heads": 24,
            "max_position_embeddings": 8192,
        },
        "2B": {
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "num_hidden_layers": 29,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "max_position_embeddings": 8192,
        },
        "4B": {
            "hidden_size": 2816,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 44,
            "num_key_value_heads": 44,
            "max_position_embeddings": 8192,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    config_params = configs[model_size]
    
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
    
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=config_params["hidden_size"],
        intermediate_size=config_params["intermediate_size"],
        num_hidden_layers=config_params["num_hidden_layers"],
        num_attention_heads=config_params["num_attention_heads"],
        num_key_value_heads=config_params["num_key_value_heads"],
        max_position_embeddings=config_params["max_position_embeddings"],
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=args.rope_theta,
        attention_bias=args.attention_bias,
        attention_dropout=args.attention_dropout,
        hidden_act=args.hidden_act,
        hidden_dropout_prob=args.hidden_dropout_prob,
        initializer_range=args.initializer_range,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=args.tie_word_embeddings,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )

    return config


def main():
    args = parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading DAC model and tokenizer...")
    dac_tools = DACTools()
    tokenizer = DACTokenizer(num_layers=9, codebook_size=1024)
    
    print(f"Loading training audio pairs from: {args.train_dir}...")
    train_pairs = find_audio_pairs(args.train_dir)
    if not train_pairs:
        print("No training audio pairs found!")
        return
    
    print(f"Found {len(train_pairs)} training pairs")
    print("Converting training audio to sequences...")
    train_sequences = []
    skipped = 0
    for dirty_path, clean_path in tqdm(train_pairs, desc="Converting training"):
        try:
            dirty_tokens = dac_tools.audio_to_tokens(dirty_path)
            clean_tokens = dac_tools.audio_to_tokens(clean_path)
            
            if args.max_duration:
                dirty_duration = dac_tools.get_duration(dirty_tokens)
                clean_duration = dac_tools.get_duration(clean_tokens)
                if dirty_duration > args.max_duration or clean_duration > args.max_duration:
                    skipped += 1
                    continue
            
            sequence = dirty_tokens + "<|start_clean|>" + clean_tokens
            train_sequences.append(sequence)
        except Exception as e:
            print(f"Error converting pair: {e}")
            continue
    
    if skipped > 0:
        print(f"Skipped {skipped} pairs exceeding max_duration ({args.max_duration}s)")
    
    if not train_sequences:
        print("No valid training sequences!")
        return
    
    eval_sequences = []
    if args.eval_dir:
        print(f"Loading evaluation audio pairs from: {args.eval_dir}...")
        eval_pairs = find_audio_pairs(args.eval_dir)
        if eval_pairs:
            print(f"Found {len(eval_pairs)} evaluation pairs")
            print("Converting evaluation audio to sequences...")
            eval_skipped = 0
            for dirty_path, clean_path in tqdm(eval_pairs, desc="Converting eval"):
                try:
                    dirty_tokens = dac_tools.audio_to_tokens(dirty_path)
                    clean_tokens = dac_tools.audio_to_tokens(clean_path)
                    
                    if args.max_duration:
                        dirty_duration = dac_tools.get_duration(dirty_tokens)
                        clean_duration = dac_tools.get_duration(clean_tokens)
                        if dirty_duration > args.max_duration or clean_duration > args.max_duration:
                            eval_skipped += 1
                            continue
                    
                    sequence = dirty_tokens + "<|start_clean|>" + clean_tokens
                    eval_sequences.append(sequence)
                except Exception as e:
                    print(f"Error converting pair: {e}")
                    continue
            
            if eval_skipped > 0:
                print(f"Skipped {eval_skipped} eval pairs exceeding max_duration ({args.max_duration}s)")
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Evaluation sequences: {len(eval_sequences)}")
    
    train_dataset = DACDataset(train_sequences, tokenizer)
    eval_dataset = DACDataset(eval_sequences, tokenizer) if eval_sequences else None
    
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        config = LlamaConfig.from_pretrained(args.checkpoint)
        model = LlamaForCausalLM.from_pretrained(args.checkpoint, config=config)
    else:
        print(f"Initializing new {args.model_size} model...")
        config = get_model_config(args.model_size, tokenizer.vocab_size, args)
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        model = LlamaForCausalLM(config)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        bf16=args.bf16,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        logging_steps=args.logging_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        dataloader_pin_memory=args.dataloader_pin_memory,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_drop_last=args.dataloader_drop_last,
        seed=args.seed,
        data_seed=args.seed,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=lambda x: collate_fn(
            x,
            pad_token_id=tokenizer.pad_token_id,
            start_clean_token_id=tokenizer.start_clean_token_id
        ),
    )
    
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.checkpoint if args.checkpoint and os.path.isdir(args.checkpoint) else None)
    
    print("Training completed!")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
