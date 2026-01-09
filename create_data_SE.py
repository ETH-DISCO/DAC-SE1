import os
import random
import numpy as np
import soundfile as sf
from datasets import Dataset, Features, Value, Audio
from huggingface_hub import login
import glob
from scipy import signal
import librosa
import gc
import psutil
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

print("Imported")

# Clean speech directories (absolute paths)
CLEAN_SPEECH_DIRS = [
    "/path/to/...",
    # Add more clean speech directories here
]

# Noise directories (absolute paths)
NOISE_DIRS = [
    "/path/to/..."
    # Add more noise directories here
]

# RIR (Room Impulse Response) directories (absolute paths)
RIR_DIRS = [
    "/path/to/...",
    # Add more RIR directories here
]

# Configuration
BATCH_SIZE = 10000  # save every x
TOTAL_ROWS = 1000000
RANDOM_SEED = 42

# Dataset distribution
DATASET_DISTRIBUTION_general = {
    'mixture_to_clean': 1000000
}

# Custom distribution - modify this to customize your dataset
DATASET_DISTRIBUTION_custom = {
    'clean_to_clean': 2000,
    'white_noise_to_clean': 1000,
    'packet_loss_to_clean': 1000,
    'white_noise_packet_loss_to_clean': 1000,
    'reverberation_to_clean': 2000,
    'downsampled_to_clean': 2000,
    'natural_noise_to_clean': 3000,
    'mixture_to_clean': 10000
}

# =============================================================================
# MIXTURE_TO_CLEAN CONFIGURATION
# =============================================================================
# Parameters for mixture_to_clean noise type generation

# Probabilities for each noise type (0.0 to 1.0)
MIXTURE_PROB_DOWNSAMPLING = 0.25      # Probability of applying downsampling
MIXTURE_PROB_WHITE_NOISE = 0.2       # Probability of applying white noise
MIXTURE_PROB_PACKET_LOSS = 0.05      # Probability of applying packet loss
MIXTURE_PROB_NATURAL_NOISE = 0.7     # Probability of applying natural noise
MIXTURE_PROB_REVERBERATION = 0.05    # Probability of applying reverberation
MIXTURE_PROB_FALLBACK_NOISE = 0.9    # Probability of adding natural noise if no noise was applied

# SNR ranges (in dB) for each noise type
MIXTURE_SNR_WHITE_NOISE_MIN = -7      # Minimum SNR for white noise
MIXTURE_SNR_WHITE_NOISE_MAX = 5       # Maximum SNR for white noise

MIXTURE_SNR_NATURAL_NOISE_MIN = -8    # Minimum SNR for natural noise (main)
MIXTURE_SNR_NATURAL_NOISE_MAX = 15    # Maximum SNR for natural noise (main)

MIXTURE_SNR_REVERBERATION_MIN = -2    # Minimum SNR for reverberation
MIXTURE_SNR_REVERBERATION_MAX = 15    # Maximum SNR for reverberation

MIXTURE_SNR_FALLBACK_NOISE_MIN = -5   # Minimum SNR for fallback natural noise
MIXTURE_SNR_FALLBACK_NOISE_MAX = 15   # Maximum SNR for fallback natural noise

# Packet loss parameters
MIXTURE_PACKET_LOSS_DROP_PROB_MIN = 0.02   # Minimum drop probability
MIXTURE_PACKET_LOSS_DROP_PROB_MAX = 0.2    # Maximum drop probability
MIXTURE_PACKET_LOSS_DURATION_MS_MIN = 10    # Minimum packet duration (ms)
MIXTURE_PACKET_LOSS_DURATION_MS_MAX = 200  # Maximum packet duration (ms)

# RMS threshold for noise signal validation
MIXTURE_RMS_THRESHOLD = 0.001         # Minimum RMS value to consider noise as valid signal

# =============================================================================
# END MIXTURE_TO_CLEAN CONFIGURATION
# =============================================================================

def get_memory_usage():
    """Get current memory usage information"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        return f"RAM: {memory_mb:.1f}MB"
    except ImportError:
        return "Memory monitoring not available"

def load_single_file(file_path):
    """Load a single audio file with detailed error handling"""
    try:
        if not os.path.exists(file_path):
            print(f"ERROR: File does not exist: {file_path}")
            return None
        
        audio_data, sample_rate = sf.read(file_path)
        
        if len(audio_data) == 0:
            print(f"ERROR: Empty audio file: {file_path}")
            return None
            
        if sample_rate <= 0:
            print(f"ERROR: Invalid sample rate {sample_rate} in file: {file_path}")
            return None
            
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"ERROR loading {file_path}: {e}")
        return None

def load_file_batch_parallel(file_paths, max_workers=8):
    """Load multiple files in parallel using thread pool"""
    loaded_files = {}
    
    print(f"Loading {len(file_paths)} files in parallel with {max_workers} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(load_single_file, path): path 
            for path in file_paths
        }
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result()
                if result is not None:
                    audio_data, sample_rate = result
                    loaded_files[path] = (audio_data, sample_rate)
            except Exception as e:
                print(f"ERROR loading {path}: {e}")
    
    load_time = time.time() - start_time
    print(f"Loaded {len(loaded_files)}/{len(file_paths)} files in {load_time:.1f}s")
    
    return loaded_files

def check_noise_has_signal(noise_audio, rms_threshold=0.01):
    """Check if noise audio has sufficient signal (not silence)"""
    rms = np.sqrt(np.mean(noise_audio**2))
    return rms > rms_threshold

class NoiseGenerator:
    def __init__(self, noise_dirs, rir_dirs):
        """Initialize noise generator with multiple noise and RIR directories"""
        self.noise_dirs = noise_dirs
        self.rir_dirs = rir_dirs
        
        # Load available noise files from all directories
        self.demand_files = []
        for noise_dir in noise_dirs:
            if os.path.exists(noise_dir):
                files = glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
                self.demand_files.extend(files)
                print(f"Found {len(files)} noise files in {noise_dir}")
            else:
                print(f"Warning: Noise directory {noise_dir} does not exist")
        
        # Load available RIR files from all directories
        self.rirs_files = []
        for rir_dir in rir_dirs:
            if os.path.exists(rir_dir):
                files = glob.glob(os.path.join(rir_dir, "**/*.wav"), recursive=True)
                self.rirs_files.extend(files)
                print(f"Found {len(files)} RIR files in {rir_dir}")
            else:
                print(f"Warning: RIR directory {rir_dir} does not exist")
        
        print(f"Total: {len(self.demand_files)} noise files, {len(self.rirs_files)} RIR files")
    
    def add_white_noise(self, audio, snr_db=20):
        """Add white noise to audio at specified SNR"""
        signal_power = np.mean(audio**2)
        noise_power = signal_power / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        return audio + noise
    
    def add_packet_loss(self, audio, sample_rate, drop_prob=0.2, packet_duration_ms=30):
        """Add packet loss simulation to audio"""
        audio_copy = audio.copy()
        packet_size = int(sample_rate * packet_duration_ms / 1000)
        num_packets = len(audio) // packet_size
        
        for i in range(num_packets):
            if random.random() < drop_prob:
                start = i * packet_size
                end = start + packet_size
                audio_copy[start:end] = 0
        
        return audio_copy
    
    def add_reverberation_with_loaded_rir(self, audio, rir_audio, rir_sr, target_sr=44100, snr_db=25):
        """Add reverberation using pre-loaded RIR audio data"""
        try:
            if audio.ndim > 1:
                audio = audio.flatten()
            if rir_audio.ndim > 1:
                rir_audio = rir_audio.flatten()
            
            if rir_sr != target_sr:
                rir = librosa.resample(rir_audio, orig_sr=rir_sr, target_sr=target_sr)
            else:
                rir = rir_audio.copy()
            
            if len(rir) > len(audio):
                start_pos = random.randint(0, len(rir) - len(audio))
                rir = rir[start_pos:start_pos + len(audio)]
            
            reverb_audio = signal.convolve(audio, rir, mode='same')
            
            signal_power = np.mean(audio**2)
            reverb_power = np.mean(reverb_audio**2)
            
            if reverb_power > 0:
                snr_linear = 10 ** (snr_db / 10)
                scaling_factor = np.sqrt(signal_power / (snr_linear * reverb_power))
                reverb_scaled = reverb_audio * scaling_factor
                
                result = audio + reverb_scaled
                
                if np.max(np.abs(result)) > 1.0:
                    result = result / np.max(np.abs(result))
                
                return result
            else:
                return audio
        except Exception as e:
            print(f"Error adding reverberation with loaded RIR: {e}")
            return audio
    
    def downsample_upsample(self, audio, target_sr=44100):
        """Downsample then upsample audio to simulate quality loss"""
        try:
            downsampling_sr = random.choice([3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000])
            downsampled = librosa.resample(audio, orig_sr=target_sr, target_sr=downsampling_sr)
            upsampled = librosa.resample(downsampled, orig_sr=downsampling_sr, target_sr=target_sr)
            return upsampled
        except Exception as e:
            print(f"Error in downsample_upsample: {e}")
            return audio
    
    def add_natural_noise_with_loaded_noise(self, audio, noise_audio, noise_sr, target_sr=44100, snr_db=25):
        """Add natural noise using pre-loaded noise audio data"""
        try:
            if audio.ndim > 1:
                audio = audio.flatten()
            if noise_audio.ndim > 1:
                noise_audio = noise_audio.flatten()
            
            if noise_sr != target_sr:
                noise = librosa.resample(noise_audio, orig_sr=noise_sr, target_sr=target_sr)
            else:
                noise = noise_audio.copy()
            
            if len(noise) >= len(audio):
                start_pos = random.randint(0, len(noise) - len(audio))
                noise = noise[start_pos:start_pos + len(audio)]
            else:
                num_repeats = (len(audio) // len(noise)) + 1
                noise_tiled = np.tile(noise, num_repeats)
                noise = noise_tiled[:len(audio)]
            
            signal_power = np.mean(audio**2)
            noise_power = np.mean(noise**2)
            
            if noise_power > 0:
                snr_linear = 10 ** (snr_db / 10)
                scaling_factor = np.sqrt(signal_power / (snr_linear * noise_power))
                noise_scaled = noise * scaling_factor
                
                result = audio + noise_scaled
                if np.max(np.abs(result)) > 1.0:
                    result = result / np.max(np.abs(result))
                
                return result
            else:
                return audio
        except Exception as e:
            print(f"Error adding natural noise with loaded noise: {e}")
            return audio

def create_dataset_row_from_loaded_audio(noise_generator, clean_path, clean_audio, clean_sr, 
                                       noise_path, noise_audio, noise_sr, rir_path, rir_audio, rir_sr,
                                       noise_type, sample_id, output_dir=None, save_local=False):
    """Create dataset row using pre-loaded audio data and save waveforms"""
    try:
        target_sr = 44100
        original_duration = len(clean_audio) / clean_sr
        
        # Skip clips less than 1 second
        if original_duration < 1.0:
            return None
        
        # For clips > 5 seconds: cut at EXACT 4-second intervals
        if original_duration > 5.0:
            total_samples = len(clean_audio)
            samples_per_4s = int(4.0 * clean_sr)
            samples_per_1s = int(1.0 * clean_sr)
            
            num_full_chunks = int(total_samples // samples_per_4s)
            remainder_samples = total_samples - (num_full_chunks * samples_per_4s)
            
            if remainder_samples < samples_per_1s and num_full_chunks > 0:
                num_chunks = num_full_chunks
            else:
                num_chunks = num_full_chunks + (1 if remainder_samples >= samples_per_1s else 0)
            
            if num_chunks > 0:
                chunk_idx = random.randint(0, num_chunks - 1)
                start_sample = int(chunk_idx * 4.0 * clean_sr)
                
                if chunk_idx == num_chunks - 1 and remainder_samples < samples_per_1s:
                    end_sample = total_samples
                else:
                    end_sample = min(start_sample + samples_per_4s, total_samples)
                
                clean_audio = clean_audio[start_sample:end_sample].copy()
        
        # Resample clean audio to target sample rate
        if clean_sr != target_sr:
            clean_audio_44k = librosa.resample(clean_audio, orig_sr=clean_sr, target_sr=target_sr)
        else:
            clean_audio_44k = clean_audio.copy()
        
        # Apply noise based on type
        if noise_type == 'clean_to_clean':
            noisy_audio = clean_audio.copy()
            noise_description = "clean"
        
        elif noise_type == 'white_noise_to_clean':
            noisy_audio = noise_generator.add_white_noise(clean_audio, snr_db=random.uniform(5, 25))
            noise_description = "white_noise"
        
        elif noise_type == 'packet_loss_to_clean':
            noisy_audio = noise_generator.add_packet_loss(clean_audio, clean_sr,
                                                        drop_prob=random.uniform(0.1, 0.3),
                                                        packet_duration_ms=random.randint(100, 200))
            noise_description = "packet_loss"
        
        elif noise_type == 'white_noise_packet_loss_to_clean':
            noisy_audio = noise_generator.add_white_noise(clean_audio, snr_db=random.uniform(5, 25))
            noisy_audio = noise_generator.add_packet_loss(noisy_audio, clean_sr,
                                                        drop_prob=random.uniform(0.02, 0.2),
                                                        packet_duration_ms=random.randint(50, 200))
            noise_description = "white_noise_packet_loss"
        
        elif noise_type == 'reverberation_to_clean':
            snr_db = random.uniform(0, 15)
            noisy_audio = noise_generator.add_reverberation_with_loaded_rir(clean_audio, rir_audio, rir_sr, target_sr, snr_db)
            noise_description = "reverberation"
        
        elif noise_type == 'downsampled_to_clean':
            noisy_audio = noise_generator.downsample_upsample(clean_audio, target_sr)
            noise_description = "downsampled_upsampled"
        
        elif noise_type == 'natural_noise_to_clean':
            snr_db = random.uniform(-5, 15)
            noisy_audio = noise_generator.add_natural_noise_with_loaded_noise(clean_audio, noise_audio, noise_sr, target_sr, snr_db)
            
            noise_component = noisy_audio - clean_audio
            if not check_noise_has_signal(noise_component, rms_threshold=0.01):
                return None
            
            if random.random() < 0.1:
                reverb_snr = random.uniform(0, 15)
                noisy_audio = noise_generator.add_reverberation_with_loaded_rir(noisy_audio, rir_audio, rir_sr, target_sr, reverb_snr)
                noise_description = "natural_noise_reverberation"
            else:
                noise_description = "natural_noise"
        
        elif noise_type == 'mixture_to_clean':
            use_reverb = True
            noisy_audio = clean_audio.copy()
            noise_types = []
            
            if random.random() < MIXTURE_PROB_DOWNSAMPLING:
                noisy_audio = noise_generator.downsample_upsample(noisy_audio, target_sr)
                noise_types.append("downsampled_upsampled")
            
            if random.random() < MIXTURE_PROB_WHITE_NOISE:
                noisy_audio = noise_generator.add_white_noise(noisy_audio, snr_db=random.uniform(MIXTURE_SNR_WHITE_NOISE_MIN, MIXTURE_SNR_WHITE_NOISE_MAX))
                noise_types.append("white_noise")
            
            if random.random() < MIXTURE_PROB_PACKET_LOSS:
                noisy_audio = noise_generator.add_packet_loss(noisy_audio, clean_sr,
                                                        drop_prob=random.uniform(MIXTURE_PACKET_LOSS_DROP_PROB_MIN, MIXTURE_PACKET_LOSS_DROP_PROB_MAX),
                                                        packet_duration_ms=random.randint(MIXTURE_PACKET_LOSS_DURATION_MS_MIN, MIXTURE_PACKET_LOSS_DURATION_MS_MAX))
                noise_types.append("packet_loss")
            
            if random.random() < MIXTURE_PROB_NATURAL_NOISE:
                snr_db = random.uniform(MIXTURE_SNR_NATURAL_NOISE_MIN, MIXTURE_SNR_NATURAL_NOISE_MAX)
                temp_noisy = noise_generator.add_natural_noise_with_loaded_noise(noisy_audio, noise_audio, noise_sr, target_sr, snr_db)
                noise_component = temp_noisy - noisy_audio
                if check_noise_has_signal(noise_component, rms_threshold=MIXTURE_RMS_THRESHOLD):
                    noisy_audio = temp_noisy
                    noise_types.append("natural_noise")
            
            if random.random() < MIXTURE_PROB_REVERBERATION and use_reverb:
                snr_db = random.uniform(MIXTURE_SNR_REVERBERATION_MIN, MIXTURE_SNR_REVERBERATION_MAX)
                noisy_audio = noise_generator.add_reverberation_with_loaded_rir(noisy_audio, rir_audio, rir_sr, target_sr, snr_db)
                noise_types.append("reverberation")
            
            if noise_types == []:
                if random.random() < MIXTURE_PROB_FALLBACK_NOISE:
                    snr_db = random.uniform(MIXTURE_SNR_FALLBACK_NOISE_MIN, MIXTURE_SNR_FALLBACK_NOISE_MAX)
                    temp_noisy = noise_generator.add_natural_noise_with_loaded_noise(noisy_audio, noise_audio, noise_sr, target_sr, snr_db)
                    noisy_audio = temp_noisy
                    noise_types.append("natural_noise")
                else:
                    noise_types.append("clean")
            
            noise_description = "_".join(noise_types) if noise_types else "clean"
        
        else:
            print(f"Unknown noise type: {noise_type}")
            return None
        
        # Resample noisy audio to target sample rate
        if clean_sr != target_sr:
            noisy_audio_44k = librosa.resample(noisy_audio, orig_sr=clean_sr, target_sr=target_sr)
        else:
            noisy_audio_44k = noisy_audio.copy()
        
        # Save waveforms if saving locally
        if save_local and output_dir:
            clean_dir = os.path.join(output_dir, "clean")
            noisy_dir = os.path.join(output_dir, "noisy")
            os.makedirs(clean_dir, exist_ok=True)
            os.makedirs(noisy_dir, exist_ok=True)
            
            clean_filename = f"clean_{sample_id}.wav"
            noisy_filename = f"dirty_{sample_id}.wav"
            
            clean_path = os.path.join(clean_dir, clean_filename)
            noisy_path = os.path.join(noisy_dir, noisy_filename)
            
            sf.write(clean_path, clean_audio_44k, target_sr)
            sf.write(noisy_path, noisy_audio_44k, target_sr)
        
        # Create dataset row with audio arrays (for HuggingFace) or file paths (for local)
        if save_local:
            dataset_row = {
                'sample_id': sample_id,
                'noise_type': noise_description,
                'clean_audio_path': os.path.join("clean", f"clean_{sample_id}.wav"),
                'noisy_audio_path': os.path.join("noisy", f"dirty_{sample_id}.wav"),
                'sample_rate': target_sr,
                'duration': len(clean_audio_44k) / target_sr
            }
        else:
            dataset_row = {
                'sample_id': sample_id,
                'noise_type': noise_description,
                'clean_audio': {'array': clean_audio_44k, 'sampling_rate': target_sr},
                'noisy_audio': {'array': noisy_audio_44k, 'sampling_rate': target_sr},
                'sample_rate': target_sr,
                'duration': len(clean_audio_44k) / target_sr
            }
        
        return dataset_row
    
    except Exception as e:
        print(f"ERROR creating dataset row from loaded audio: {e}", flush=True)
        return None

def process_batch_combinations(clean_loaded, noise_loaded, rir_loaded, noise_generator, noise_type, 
                              target_count, used_speech_files=None, sample_id_start=0, output_dir=None, save_local=False):
    """Process combinations of loaded files efficiently"""
    combinations_processed = 0
    batch_data = []
    
    if used_speech_files is None:
        used_speech_files = set()
    
    max_samples_per_batch = 10000
    
    clean_items = [(path, data) for path, data in clean_loaded.items() if data is not None]
    noise_items = [(path, data) for path, data in noise_loaded.items() if data is not None]
    rir_items = [(path, data) for path, data in rir_loaded.items() if data is not None]
    
    random.shuffle(noise_items)
    random.shuffle(rir_items)
    
    batch_start_time = time.time()
    
    if not clean_items or not noise_items or not rir_items:
        print(f"ERROR: Missing files - clean: {len(clean_items)}, noise: {len(noise_items)}, rir: {len(rir_items)}")
        return batch_data
    
    print(f"  Using {len(clean_items)} clean, {len(noise_items)} noise, {len(rir_items)} RIR files")
    
    for clean_path, (clean_audio, clean_sr) in clean_items:
        try:
            if clean_path in used_speech_files:
                continue
            
            if combinations_processed >= max_samples_per_batch:
                break
            
            noise_choice = random.choice(noise_items)
            rir_choice = random.choice(rir_items)
            
            if noise_choice is None or len(noise_choice) != 2:
                continue
            if rir_choice is None or len(rir_choice) != 2:
                continue
            
            noise_path, (noise_audio, noise_sr) = noise_choice
            rir_path, (rir_audio, rir_sr) = rir_choice
            
            sample_id = f"{sample_id_start + combinations_processed:08d}"
            
            result = create_dataset_row_from_loaded_audio(
                noise_generator, clean_path, clean_audio, clean_sr,
                noise_path, noise_audio, noise_sr, rir_path, rir_audio, rir_sr,
                noise_type, sample_id, output_dir, save_local
            )
            
            if result is not None:
                batch_data.append(result)
                combinations_processed += 1
                used_speech_files.add(clean_path)
                
                if combinations_processed % 100 == 0:
                    elapsed = time.time() - batch_start_time
                    print(f"  Processed {combinations_processed}/{min(len(clean_items), max_samples_per_batch)} rows in {elapsed:.1f}s", flush=True)
                
                if combinations_processed % 50 == 0:
                    gc.collect()
        
        except Exception as e:
            print(f"ERROR in combination loop: {e}", flush=True)
            continue
    
    print(f"Processed {combinations_processed} combinations")
    return batch_data

def pre_scan_and_cache_file_lists(clean_speech_dirs, noise_dirs, rir_dirs, speaker_ids=None):
    """Pre-scan directories and cache file lists with metadata"""
    print("Pre-scanning directories and caching file lists...")
    
    clean_files = []
    print("📁 Scanning clean speech directories...")
    for clean_dir in clean_speech_dirs:
        if os.path.exists(clean_dir):
            flac_files = glob.glob(os.path.join(clean_dir, "**/*.flac"), recursive=True)
            wav_files = glob.glob(os.path.join(clean_dir, "**/*.wav"), recursive=True)
            all_files = flac_files + wav_files
            
            if speaker_ids is not None:
                filtered_files = []
                for file_path in all_files:
                    path_parts = file_path.split(os.sep)
                    for part in path_parts:
                        try:
                            file_speaker_id = int(part)
                            if file_speaker_id in speaker_ids:
                                filtered_files.append(file_path)
                                break
                        except ValueError:
                            continue
                all_files = filtered_files
            
            clean_files.extend(all_files)
            print(f"  ✓ Found {len(all_files)} clean files in {clean_dir}")
        else:
            print(f"  ⚠️  Warning: Clean speech directory {clean_dir} does not exist")
    
    noise_files = []
    print("📁 Scanning noise directories...")
    for noise_dir in noise_dirs:
        if os.path.exists(noise_dir):
            files = glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
            noise_files.extend(files)
            print(f"  ✓ Found {len(files)} noise files in {noise_dir}")
        else:
            print(f"  ⚠️  Warning: Noise directory {noise_dir} does not exist")
    
    rir_files = []
    print("📁 Scanning RIR directories...")
    for rir_dir in rir_dirs:
        if os.path.exists(rir_dir):
            files = glob.glob(os.path.join(rir_dir, "**/*.wav"), recursive=True)
            rir_files.extend(files)
            print(f"  ✓ Found {len(files)} RIR files in {rir_dir}")
        else:
            print(f"  ⚠️  Warning: RIR directory {rir_dir} does not exist")
    
    print(f"\n📊 Total cached: {len(clean_files)} clean, {len(noise_files)} noise, {len(rir_files)} RIR files")
    
    return clean_files, noise_files, rir_files

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create clean/noisy audio waveform dataset')
    parser.add_argument('--identifier', type=str, required=True, 
                       help='Dataset number/identifier for Hugging Face dataset name')
    parser.add_argument('--distribution', type=str, required=True,
                       choices=['general', 'custom'],
                       help='Dataset distribution type: general (balanced mix) or custom (modify DATASET_DISTRIBUTION_custom in code)')
    parser.add_argument('--split', type=str, required=True,
                       choices=['train', 'grpo', 'test'],
                       help='Which split to generate: train, grpo, or test')
    parser.add_argument('--save_mode', type=str, required=True,
                       choices=['hf', 'local'],
                       help='Save mode: hf (HuggingFace Hub) or local (local folders)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for local save mode (required if save_mode=local)')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace API token (default: from HF_TOKEN environment variable)')
    return parser.parse_args()

def main():
    """Main function to create the dataset"""
    args = parse_arguments()
    hf_number = args.identifier
    distribution_type = args.distribution
    selected_split = args.split
    save_mode = args.save_mode
    output_dir = args.output_dir
    
    # Get HuggingFace token from argument or environment variable
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    if not hf_token and save_mode == 'hf':
        print("ERROR: HuggingFace token required for HF save mode. Provide --hf_token or set HF_TOKEN environment variable")
        return
    
    # Set dataset distribution
    distribution_map = {
        'general': DATASET_DISTRIBUTION_general,
        'custom': DATASET_DISTRIBUTION_custom
    }
    
    DATASET_DISTRIBUTION = distribution_map[distribution_type]
    
    # Define target sample counts for each split
    split_targets = {
        'train': 1000000,
        'grpo': 100000,
        'test': 10000
    }
    
    target_samples = split_targets[selected_split]
    distribution_sum = sum(DATASET_DISTRIBUTION.values())
    DATASET_DISTRIBUTION_SCALED = {
        k: int(v * target_samples / distribution_sum) 
        for k, v in DATASET_DISTRIBUTION.items()
    }
    
    # Validate save mode and output directory
    save_local = (save_mode == 'local')
    if save_local:
        if output_dir is None:
            print("ERROR: --output_dir is required when --save_mode=local")
            return
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 SAVE MODE: Local (to {output_dir})")
    else:
        login(token=hf_token)
        HF_DATASET_NAME = f"AdoCleanCode/clean_dirty_waveform_{hf_number}_{selected_split}"
        print(f"☁️  SAVE MODE: HuggingFace Hub ({HF_DATASET_NAME})")
    
    print("=== Clean/Noisy Waveform Dataset Creator ===")
    print(f"Dataset identifier: {hf_number}")
    print(f"Distribution type: {distribution_type}")
    print(f"Selected split: {selected_split.upper()}")
    print(f"Target samples: {target_samples:,}")
    print(f"Batch size: {BATCH_SIZE:,}")
    print(f"Distribution: {DATASET_DISTRIBUTION_SCALED}")
    
    # Check if directories exist
    clean_dirs_exist = any(os.path.exists(d) for d in CLEAN_SPEECH_DIRS)
    noise_dirs_exist = any(os.path.exists(d) for d in NOISE_DIRS)
    rir_dirs_exist = any(os.path.exists(d) for d in RIR_DIRS)
    
    if not clean_dirs_exist:
        print(f"ERROR: No clean speech directories found! Checked: {CLEAN_SPEECH_DIRS}")
        return
    
    if not noise_dirs_exist:
        print(f"ERROR: No noise directories found! Checked: {NOISE_DIRS}")
        return
    
    if not rir_dirs_exist:
        print(f"ERROR: No RIR directories found! Checked: {RIR_DIRS}")
        return
    
    print(f"✓ Found clean speech directories: {[d for d in CLEAN_SPEECH_DIRS if os.path.exists(d)]}")
    print(f"✓ Found noise directories: {[d for d in NOISE_DIRS if os.path.exists(d)]}")
    print(f"✓ Found RIR directories: {[d for d in RIR_DIRS if os.path.exists(d)]}")
    
    print("Initializing noise generator...")
    noise_generator = NoiseGenerator(NOISE_DIRS, RIR_DIRS)
    
    initial_memory = get_memory_usage()
    print(f"Initial memory status: {initial_memory}")
    
    # Define dataset features
    if save_local:
        features = Features({
            'sample_id': Value('string'),
            'noise_type': Value('string'),
            'clean_audio_path': Value('string'),
            'noisy_audio_path': Value('string'),
            'sample_rate': Value('int64'),
            'duration': Value('float64')
        })
    else:
        features = Features({
            'sample_id': Value('string'),
            'noise_type': Value('string'),
            'clean_audio': Audio(sampling_rate=44100),
            'noisy_audio': Audio(sampling_rate=44100),
            'sample_rate': Value('int64'),
            'duration': Value('float64')
        })
    
    # Pre-scan and cache file lists
    print(f"\n📁 Caching files for {selected_split.upper()} split...")
    split_clean_files, split_noise_files, split_rir_files = pre_scan_and_cache_file_lists(
        CLEAN_SPEECH_DIRS, NOISE_DIRS, RIR_DIRS
    )
    
    if not split_clean_files:
        print("Error: Not enough clean files found in directories")
        return
    
    # Process each noise type
    total_processed = 0
    batch_num = 0
    start_time = time.time()
    used_speech_files = set()
    sample_id_counter = 0
    
    total_target_samples = sum(DATASET_DISTRIBUTION_SCALED.values())
    
    for noise_type, target_count in DATASET_DISTRIBUTION_SCALED.items():
        print(f"\n=== Processing {noise_type}: {target_count:,} samples ===")
        processed_count = 0
        accumulated_data = []
        
        clean_batch_size = 500
        noise_batch_size = 500
        rir_batch_size = 25
        
        while processed_count < target_count:
            available_clean_files = [f for f in split_clean_files if f not in used_speech_files]
            
            if not available_clean_files:
                print(f"⚠️  WARNING: All clean speech files exhausted! Processed {processed_count}/{target_count} samples")
                break
            
            clean_batch_files = random.sample(available_clean_files, min(clean_batch_size, len(available_clean_files)))
            noise_batch_files = random.sample(split_noise_files, min(noise_batch_size, len(split_noise_files)))
            rir_batch_files = random.sample(split_rir_files, min(rir_batch_size, len(split_rir_files)))
            
            print(f"Loading batch: {len(clean_batch_files)} clean, {len(noise_batch_files)} noise, {len(rir_batch_files)} RIR files", flush=True)
            
            try:
                clean_loaded = load_file_batch_parallel(clean_batch_files, max_workers=8)
                noise_loaded = load_file_batch_parallel(noise_batch_files, max_workers=8)
                rir_loaded = load_file_batch_parallel(rir_batch_files, max_workers=8)
                
                if not clean_loaded or not noise_loaded or not rir_loaded:
                    print("ERROR: Failed to load files", flush=True)
                    break
                
                remaining_count = target_count - processed_count
                batch_data = process_batch_combinations(
                    clean_loaded, noise_loaded, rir_loaded, noise_generator,
                    noise_type, remaining_count, used_speech_files=used_speech_files,
                    sample_id_start=sample_id_counter, output_dir=output_dir, save_local=save_local
                )
                
                if not batch_data:
                    print("ERROR: No data generated from batch combinations")
                    break
                
                accumulated_data.extend(batch_data)
                processed_count += len(batch_data)
                total_processed += len(batch_data)
                sample_id_counter += len(batch_data)
                
                print(f"  📊 Accumulated: {len(accumulated_data):,} samples (BATCH_SIZE: {BATCH_SIZE:,})", flush=True)
                
                # Save/upload when batch is full
                if len(accumulated_data) >= BATCH_SIZE or processed_count >= target_count:
                    save_data = accumulated_data[:BATCH_SIZE] if len(accumulated_data) > BATCH_SIZE else accumulated_data
                    save_data = [row for row in save_data if row is not None]
                    
                    if not save_data:
                        accumulated_data = accumulated_data[BATCH_SIZE:] if len(accumulated_data) > BATCH_SIZE else []
                        continue
                    
                    if save_local:
                        # For local save, data is already saved to files, just create metadata JSON
                        metadata_file = os.path.join(output_dir, f"metadata_batch_{batch_num:04d}.jsonl")
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            for sample in save_data:
                                json.dump(sample, f, ensure_ascii=False)
                                f.write('\n')
                        print(f"✓ Batch {batch_num:04d} metadata saved to {metadata_file}")
                    else:
                        print(f"\n[•] Uploading batch {batch_num:04d} with {len(save_data)} samples to HuggingFace...", flush=True)
                        try:
                            dataset = Dataset.from_list(save_data, features=features)
                            dataset.push_to_hub(
                                HF_DATASET_NAME,
                                split=f"batch_{batch_num:04d}",
                                token=hf_token,
                                private=False
                            )
                            print(f"✓ SUCCESS: Batch {batch_num:04d} uploaded successfully")
                        except Exception as e:
                            print(f"✗ ERROR: Failed to upload batch {batch_num}: {e}")
                    
                    batch_num += 1
                    accumulated_data = accumulated_data[BATCH_SIZE:] if len(accumulated_data) > BATCH_SIZE else []
                    
                    del save_data
                    gc.collect()
                
                del clean_loaded, noise_loaded, rir_loaded, batch_data
                gc.collect()
                
                memory_status = get_memory_usage()
                elapsed_time = time.time() - start_time
                samples_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
                print(f"📊 Speedometer: {samples_per_second:.1f} samples/sec | Memory: {memory_status}")
                print(f"   Processed: {total_processed:,}/{total_target_samples:,} | Batches: {batch_num}")
                
            except Exception as e:
                print(f"✗ CRITICAL ERROR in batch processing: {e}")
                gc.collect()
                break
        
        print(f"Completed processing {noise_type}: {processed_count:,} samples")
    
    # Final statistics
    total_elapsed_time = time.time() - start_time
    avg_samples_per_second = total_processed / total_elapsed_time if total_elapsed_time > 0 else 0
    
    print(f"\n🎉 === {selected_split.upper()} SPLIT COMPLETE ===")
    print(f"📊 Final Statistics:")
    print(f"   Total samples processed: {total_processed:,}")
    print(f"   Total batches created: {batch_num}")
    print(f"   Total time: {total_elapsed_time/3600:.2f} hours")
    print(f"   Average speed: {avg_samples_per_second:.1f} samples/sec")
    if save_local:
        print(f"   💾 Dataset saved to: {output_dir}")
        print(f"      - Clean audio: {os.path.join(output_dir, 'clean')}")
        print(f"      - Noisy audio: {os.path.join(output_dir, 'noisy')}")
    else:
        print(f"   ☁️  Dataset uploaded to: {HF_DATASET_NAME}")
    print(f"   Unique speech files used: {len(used_speech_files)}")
    
    final_memory = get_memory_usage()
    print(f"   Final memory status: {final_memory}")

if __name__ == "__main__":
    main()
