import re
import os
import glob
import argparse
import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    DacModel,
    AutoProcessor,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DAC Denoiser Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to WAV file or directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for denoised WAV files")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to process")
    parser.add_argument("--chunk_duration", type=float, default=4.0, help="Chunk duration in seconds")
    parser.add_argument("--overlap_ratio", type=float, default=0.5, help="Overlap ratio between chunks")
    parser.add_argument("--min_chunk_duration", type=float, default=1.0, help="Minimum duration for last chunk")
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
    
    def tokens_to_string(self, tokens):
        out = []
        for i in range(0, len(tokens), self.num_codebooks):
            group = tokens[i:i+self.num_codebooks]
            for codebook_idx, token_id in enumerate(group, start=1):
                out.append(f"<|s{token_id}_c{codebook_idx}|>")
        return "".join(out)
    
    def string_to_tokens(self, token_string):
        pattern = r'<\|s(\d+)_c(\d+)\|>'
        matches = re.findall(pattern, token_string)
        return [int(token_id) for token_id, _ in matches] if matches else []
    
    def tokens_to_audio(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        num_steps = len(tokens) // self.num_codebooks
        codes = tokens.view(num_steps, self.num_codebooks).T.unsqueeze(0)
        
        with torch.no_grad():
            audio_values = self.model.decode(audio_codes=codes.to(self.device)).audio_values
            return audio_values[0].cpu().detach().numpy()
    
    def get_duration(self, token_string):
        dac_tokens = re.findall(r'<\|s\d+_c\d\|>', token_string)
        num_timesteps = len(dac_tokens) // self.num_codebooks
        return num_timesteps / self.timesteps_per_second


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
        self.vocab_size = len(self.vocab_tokens)
    
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


class DACConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, num_codebooks=9, codebook_size=1024, min_tokens=0):
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.min_tokens = min_tokens
        self.call_count = 0
        
        self.codebook_ranges = {}
        for i in range(1, num_codebooks + 1):
            start_idx = 5 + (i - 1) * codebook_size
            end_idx = start_idx + codebook_size
            self.codebook_ranges[i] = (start_idx, end_idx)
        
        self.start_clean_token_id = tokenizer.start_clean_token_id
        self.eos_token_id = tokenizer.eos_token_id
    
    def __call__(self, input_ids, scores):
        self.call_count += 1
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            prev_token_id = input_ids[batch_idx, -1].item()
            prev_token_str = self.tokenizer.id_to_token.get(prev_token_id, "")
            
            device = scores.device
            allowed_mask = torch.zeros(self.tokenizer.vocab_size, dtype=torch.bool, device=device)
            
            if prev_token_id == self.start_clean_token_id:
                start_idx, end_idx = self.codebook_ranges[1]
                allowed_mask[start_idx:end_idx] = True
            elif "_c" in prev_token_str:
                match = re.search(r'_c(\d+)', prev_token_str)
                if match:
                    current_layer = int(match.group(1))
                    if current_layer < 9:
                        next_layer = current_layer + 1
                        start_idx, end_idx = self.codebook_ranges[next_layer]
                        allowed_mask[start_idx:end_idx] = True
                    else:
                        start_idx, end_idx = self.codebook_ranges[1]
                        allowed_mask[start_idx:end_idx] = True
                        if self.call_count >= self.min_tokens:
                            allowed_mask[self.eos_token_id] = True
                else:
                    for i in range(1, 10):
                        start_idx, end_idx = self.codebook_ranges[i]
                        allowed_mask[start_idx:end_idx] = True
            else:
                start_idx, end_idx = self.codebook_ranges[1]
                allowed_mask[start_idx:end_idx] = True
            
            scores[batch_idx, :] = scores[batch_idx, :].masked_fill(~allowed_mask, float('-inf'))
        
        return scores


def split_into_chunks(token_string, chunk_duration, overlap_ratio, min_chunk_duration, dac_tools):
    dac_tokens = re.findall(r'<\|s\d+_c\d\|>', token_string)
    num_timesteps = len(dac_tokens) // dac_tools.num_codebooks
    total_duration = num_timesteps / dac_tools.timesteps_per_second
    
    if total_duration <= chunk_duration:
        return [{
            'tokens': token_string,
            'start_timestep': 0,
            'end_timestep': num_timesteps,
        }]
    
    chunk_timesteps = int(chunk_duration * dac_tools.timesteps_per_second)
    overlap_timesteps = int(chunk_timesteps * overlap_ratio)
    stride_timesteps = chunk_timesteps - overlap_timesteps
    min_chunk_timesteps = int(min_chunk_duration * dac_tools.timesteps_per_second)
    
    chunks = []
    start_timestep = 0
    chunk_idx = 0
    
    while start_timestep < num_timesteps:
        end_timestep = min(start_timestep + chunk_timesteps, num_timesteps)
        
        if end_timestep == num_timesteps:
            chunk_len = end_timestep - start_timestep
            if chunk_len < min_chunk_timesteps and chunk_idx > 0:
                chunks[-1]['end_timestep'] = end_timestep
                start_token_idx = chunks[-1]['start_timestep'] * dac_tools.num_codebooks
                end_token_idx = end_timestep * dac_tools.num_codebooks
                chunks[-1]['tokens'] = "".join(dac_tokens[start_token_idx:end_token_idx])
                break
        
        start_token_idx = start_timestep * dac_tools.num_codebooks
        end_token_idx = end_timestep * dac_tools.num_codebooks
        chunk_tokens = "".join(dac_tokens[start_token_idx:end_token_idx])
        
        chunks.append({
            'tokens': chunk_tokens,
            'start_timestep': start_timestep,
            'end_timestep': end_timestep,
        })
        
        if end_timestep == num_timesteps:
            break
        
        start_timestep += stride_timesteps
        chunk_idx += 1
    
    return chunks


def merge_chunks_with_crossfade(chunk_results, overlap_ratio, dac_tools):
    if not chunk_results:
        return "", None
    
    if len(chunk_results) == 1:
        seq = chunk_results[0]['generated_tokens']
        tokens = dac_tools.string_to_tokens(seq)
        if tokens:
            remainder = len(tokens) % dac_tools.num_codebooks
            if remainder != 0:
                tokens = tokens[:len(tokens) - remainder]
            if tokens:
                return seq, dac_tools.tokens_to_audio(tokens)
        return seq, None
    
    chunk_audios = []
    chunk_sequences = []
    
    for chunk in chunk_results:
        seq = chunk['generated_tokens']
        chunk_sequences.append(seq)
        
        tokens = dac_tools.string_to_tokens(seq)
        if not tokens:
            chunk_audios.append(np.zeros(1000, dtype=np.float32))
            continue
        
        remainder = len(tokens) % dac_tools.num_codebooks
        if remainder != 0:
            tokens = tokens[:len(tokens) - remainder]
        
        if tokens:
            chunk_audios.append(dac_tools.tokens_to_audio(tokens).flatten())
        else:
            chunk_audios.append(np.zeros(1000, dtype=np.float32))
    
    if chunk_results:
        chunk_timesteps = chunk_results[0]['end_timestep'] - chunk_results[0]['start_timestep']
        overlap_timesteps = int(chunk_timesteps * overlap_ratio)
        overlap_samples = int(overlap_timesteps * (dac_tools.sample_rate / dac_tools.timesteps_per_second))
    else:
        overlap_samples = 0
    
    merged_audio = chunk_audios[0].copy()
    
    for i in range(1, len(chunk_audios)):
        current_chunk = chunk_audios[i]
        actual_overlap = min(overlap_samples, len(merged_audio), len(current_chunk))
        
        if actual_overlap > 10:
            fade_out = np.linspace(1.0, 0.0, actual_overlap)
            fade_in = np.linspace(0.0, 1.0, actual_overlap)
            
            overlap_start = len(merged_audio) - actual_overlap
            merged_overlap = merged_audio[overlap_start:].copy()
            current_overlap = current_chunk[:actual_overlap].copy()
            
            crossfaded = merged_overlap * fade_out + current_overlap * fade_in
            
            merged_audio = np.concatenate([
                merged_audio[:overlap_start],
                crossfaded,
                current_chunk[actual_overlap:],
            ])
        else:
            merged_audio = np.concatenate([merged_audio, current_chunk])
    
    return "".join(chunk_sequences), merged_audio


def find_wav_files(input_path):
    wav_files = []
    
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.wav', '.wave')):
            wav_files.append((input_path, os.path.basename(input_path)))
    elif os.path.isdir(input_path):
        for ext in ['*.wav', '*.WAV', '*.wave', '*.WAVE']:
            found = glob.glob(os.path.join(input_path, ext))
            wav_files.extend((f, os.path.basename(f)) for f in found)
            
            found = glob.glob(os.path.join(input_path, '**', ext), recursive=True)
            wav_files.extend((f, os.path.relpath(f, input_path)) for f in found)
        
        wav_files = list(set(wav_files))
    
    return wav_files


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading DAC model and tokenizer...")
    dac_tools = DACTools()
    tokenizer = DACTokenizer(num_layers=9, codebook_size=1024)
    
    print(f"Loading model from {args.model_path}...")
    model = LlamaForCausalLM.from_pretrained(args.model_path)
    model = model.cuda()
    model.eval()
    
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    model.config.use_cache = True
    
    print(f"Loading WAV files from: {args.input_dir}...")
    wav_files = find_wav_files(args.input_dir)
    if not wav_files:
        print("No WAV files found!")
        return
    
    if args.max_samples:
        wav_files = wav_files[:args.max_samples]
    
    print("Converting audio to tokens...")
    samples = []
    for wav_path, filename in tqdm(wav_files, desc="Converting"):
        try:
            noisy_tokens = dac_tools.audio_to_tokens(wav_path)
            samples.append((noisy_tokens, filename))
        except Exception as e:
            print(f"Error converting {filename}: {e}")
            continue
    
    if not samples:
        print("No valid audio files processed!")
        return
    
    print(f"Running inference on {len(samples)} samples...")
    errors = 0
    
    for idx, (noisy_tokens, filename) in enumerate(tqdm(samples, desc="Processing")):
        try:
            chunks = split_into_chunks(
                noisy_tokens,
                args.chunk_duration,
                args.overlap_ratio,
                args.min_chunk_duration,
                dac_tools,
            )
            
            chunk_results = []
            for chunk in chunks:
                chunk_tokens = chunk['tokens']
                
                token_ids = tokenizer.encode(chunk_tokens, add_special_tokens=False)
                input_ids = [tokenizer.bos_token_id] + token_ids + [tokenizer.start_clean_token_id]
                input_tensor = torch.tensor([input_ids]).cuda()
                
                num_tokens = len(re.findall(r'<\|s\d+_c\d\|>', chunk_tokens))
                max_new_tokens = min(num_tokens + 100, 4098)
                min_new_tokens = num_tokens
                
                logits_processor = LogitsProcessorList([
                    DACConstrainedLogitsProcessor(
                        tokenizer=tokenizer,
                        min_tokens=min_new_tokens,
                    )
                ])
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_tensor,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        logits_processor=logits_processor,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        num_beams=1,
                        early_stopping=False,
                        use_cache=True,
                    )
                
                generated_ids = outputs[0, len(input_ids):].tolist()
                generated_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                valid_tokens = re.findall(r'<\|s\d+_c\d\|>', generated_output)
                if valid_tokens:
                    remainder = len(valid_tokens) % 9
                    if remainder != 0:
                        valid_tokens = valid_tokens[:len(valid_tokens) - remainder]
                    denoised_tokens = "".join(valid_tokens)
                else:
                    denoised_tokens = ""
                
                chunk_results.append({
                    'generated_tokens': denoised_tokens,
                    'start_timestep': chunk['start_timestep'],
                    'end_timestep': chunk['end_timestep'],
                })
            
            _, merged_audio = merge_chunks_with_crossfade(chunk_results, args.overlap_ratio, dac_tools)
            
            if merged_audio is not None:
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}_denoised.wav"
                output_path = os.path.join(args.output_dir, output_filename)
                try:
                    sf.write(output_path, merged_audio, dac_tools.sample_rate)
                except Exception as e:
                    print(f"Error saving {output_filename}: {e}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            errors += 1
            continue
    
    print(f"Inference complete! Processed {len(samples) - errors} samples (errors: {errors})")
    print(f"Denoised WAV files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
