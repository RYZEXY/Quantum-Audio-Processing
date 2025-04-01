from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
import numpy as np
from scipy.io.wavfile import read, write
import matplotlib.pyplot as plt
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator, Aer, aer_simulator

def load_audio(filename):
    """Load an audio file and return the sample rate and audio data."""
    sample_rate, audio = read(filename)
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    return sample_rate, audio

def process_audio_chunk(audio_chunk, n_qubits=13, threshold=0.05):
    """Process a chunk of audio data using quantum circuit."""
    # Normalize the audio for quantum state preparation
    norm_factor = np.linalg.norm(audio_chunk)
    
    # Handle the case where the chunk might be all zeros
    if norm_factor < 1e-10:  # Effectively zero
        print("Warning: Audio chunk contains all zeros or very small values")
        # Return zeros for this chunk
        return np.zeros(len(audio_chunk))
        
    normalized_audio = audio_chunk / norm_factor
    
    # Prepare quantum state
    qc = prepare_quantum_state(normalized_audio, n_qubits)
    
    # Apply QFT
    qc.append(QFT(n_qubits, do_swaps=True), range(n_qubits))
    
    # Get state after QFT
    state = Statevector(qc)
    
    # Apply oracle to denoise the signal
    qc = qc.compose((denoise_oracle(state, n_qubits, threshold)))
    
    # Inverse QFT to return to time domain
    qc = qc.compose(QFT(n_qubits, do_swaps=True, inverse=True), range(n_qubits))
    
    # "Measure" the statevector
    backend = Aer.get_backend('statevector_simulator')
    qc = transpile(qc, backend)
    state = Statevector(qc)
    
    # Denormalize the processed audio
    processed_chunk = state.data.real * norm_factor
    
    return processed_chunk

def prepare_quantum_state(normalized_audio, n_qubits):
    """Prepare a quantum state from normalized audio data."""
    qc = QuantumCircuit(n_qubits)
    # Initialize the quantum state with the normalized audio data
    qc.initialize(normalized_audio, range(n_qubits))
    return qc

def denoise_oracle(statevector, n_qubits, threshold=0.05):
    """Apply denoising to the quantum state."""
    # Get data from statevector (we can't modify it directly)
    sv_data = statevector.data.copy()  # Make a copy we can modify
    probabilities = np.abs(sv_data) ** 2
    
    # Apply thresholding to our copy
    for i in range(0, 2**n_qubits):
        if probabilities[i] < threshold:
            sv_data[i] = 0
    
    # Check if we removed all values or if norm is too small
    norm_factor2 = np.linalg.norm(sv_data)
    
    if norm_factor2 < 1e-10:  # Effectively zero
        # If all values were removed or norm is too small, keep the strongest component
        max_idx = np.argmax(probabilities)
        sv_data = np.zeros_like(sv_data)
        sv_data[max_idx] = 1.0
        normalized_denoise = sv_data
    else:
        normalized_denoise = sv_data / norm_factor2
    
    denoised_state = prepare_quantum_state(normalized_denoise, n_qubits)
    return denoised_state

def process_full_audio(full_audio, sample_rate, threshold=0.05, overlap=0.1):
    """Process audio data in chunks of 2^13 samples with overlap."""
    # Define chunk size (2^13)
    chunk_size = 2**13
    n_qubits = 13
    
    # Calculate overlap in samples
    overlap_samples = int(chunk_size * overlap)
    
    # Calculate number of chunks with overlap
    total_samples = len(full_audio)
    # Adjust calculation for overlapping chunks
    num_chunks = (total_samples - overlap_samples + chunk_size - 1) // (chunk_size - overlap_samples)
    
    print(f"Processing {total_samples} samples in {num_chunks} chunks of {chunk_size} samples each (with {overlap_samples} sample overlap)")
    
    # Initialize array for processed audio and overlap weights
    processed_audio = np.zeros(total_samples, dtype=np.float64)  # Use float64 for accumulation
    overlap_weights = np.zeros(total_samples, dtype=np.float64)
    
    # Process each chunk
    for i in range(num_chunks):
        # Calculate start and end indices with overlap
        start_idx = i * (chunk_size - overlap_samples)
        end_idx = min(start_idx + chunk_size, total_samples)
        
        # Get current chunk
        current_chunk = full_audio[start_idx:end_idx]
        
        # If the last chunk is smaller than chunk_size, pad with zeros
        if len(current_chunk) < chunk_size:
            padded_chunk = np.zeros(chunk_size, dtype=full_audio.dtype)
            padded_chunk[:len(current_chunk)] = current_chunk
            current_chunk = padded_chunk
            
        print(f"Processing chunk {i+1}/{num_chunks}: samples {start_idx} to {end_idx-1}")
        
        try:
            # Process the chunk
            processed_chunk = process_audio_chunk(current_chunk, n_qubits, threshold)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {str(e)}")
            # If processing fails, use the original chunk
            processed_chunk = current_chunk
        
        # If it was a padded chunk, only take the valid part
        valid_samples = min(chunk_size, end_idx - start_idx)
        
        # Create a crossfade weight window (triangular for simplicity)
        window = np.hanning(valid_samples)
        if i > 0 and overlap_samples > 0:  # Apply fade-in for non-first chunks
            fade_in = np.linspace(0, 1, min(overlap_samples, valid_samples))
            window[:len(fade_in)] = fade_in
        
        if i < num_chunks - 1 and end_idx < total_samples:  # Apply fade-out for non-last chunks
            fade_out = np.linspace(1, 0, min(overlap_samples, valid_samples - max(0, valid_samples - overlap_samples)))
            window[-len(fade_out):] = fade_out
        
        # Apply window and accumulate
        processed_audio[start_idx:start_idx + valid_samples] += processed_chunk[:valid_samples].real * window
        overlap_weights[start_idx:start_idx + valid_samples] += window
    
    # Normalize by the overlap weights to get the final audio
    # Avoid division by zero
    mask = overlap_weights > 0
    processed_audio[mask] /= overlap_weights[mask]
    
    return processed_audio

# Main execution
if __name__ == "__main__":
    audio_file = "noisy_audio.wav"
    threshold = 0.05  # Adjust as needed
    overlap = 0.50  # 50% overlap between chunks
    
    # Load the original audio directly in the main function
    sample_rate, original_audio = load_audio(audio_file)
    
    # Process the audio file
    processed_audio = process_full_audio(original_audio, sample_rate, threshold, overlap)
    
    # Save the processed audio - ensure we're using the original sample rate and data type
    original_dtype = original_audio.dtype
    if np.issubdtype(original_dtype, np.integer):
        # For integer types, scale appropriately to avoid clipping
        processed_audio = np.clip(processed_audio, np.iinfo(original_dtype).min, np.iinfo(original_dtype).max)
    
    write("processed_audio.wav", sample_rate, processed_audio.astype(original_dtype))
    
    # Plot original and processed audio (optional)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.title("Original Audio")
    plt.plot(original_audio[:min(10000, len(original_audio))])
    
    plt.subplot(2, 1, 2)
    plt.title("Processed Audio")
    plt.plot(processed_audio[:min(10000, len(processed_audio))])
    
    plt.tight_layout()
    plt.savefig("audio_comparison.png")
    plt.show()