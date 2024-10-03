import numpy as np
import librosa
import librosa.effects
import math
import soundfile as sf
import sounddevice as sd

audio = "audio_prueba.wav"

# Notas afinadas
frecuencias = {
    'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63,
    'F': 349.23, 'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00,
    'A#': 466.16, 'B': 493.88
}

A4 = 440  # Frecuencia de La4
C0 = A4 * 2**(-4.75)  # Calcular frecuencia de C0

def loadAudio(archivo):
    data, samplerate = librosa.load(archivo, sr = None, mono = True)
    return data, samplerate

def playAudio(data, sr):
    sd.play(data, sr)
    sd.wait()

def definir_nota(freq):
    closest_note = min(frecuencias.keys(), key = lambda note: abs(frecuencias[note] - freq))
    return frecuencias[closest_note]

def frecuencia_nota(freq):
    if freq == 0:
        return None
    h = round(12 * math.log2(freq / C0))
    octave = h // 12
    n = h % 12
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[n] + str(octave)

def calculate_semitone_shift(fundamental_freq, adjusted_freq):
    return 12 * np.log2(adjusted_freq / fundamental_freq)

def tune_audio(audio_path, afinado):
    # Cargar el audio
    y, sr = loadAudio(audio_path)

    # Aplicar FFT
    fft_result = np.fft.fft(y)
    fft_freqs = np.fft.fftfreq(len(fft_result), 1/sr)

    # Obtener la frecuencia fundamental
    frequencies = np.abs(fft_result)
    max_index = np.argmax(frequencies)
    fundamental_freq = abs(fft_freqs[max_index])

    print(f"Frecuencia Fundamental Detectada: {fundamental_freq} Hz")

    # Convertir a nota musical
    note = frecuencia_nota(fundamental_freq)
    print(f"Nota Detectada: {note}")

    # Encontrar la frecuencia de la nota afinada más cercana
    adjusted_freq = definir_nota(fundamental_freq)
    print(f"Frecuencia Ajustada a: {adjusted_freq} Hz")

    # Calcular el cambio en semitonos
    semitone_shift = calculate_semitone_shift(fundamental_freq, adjusted_freq)

    # Ajustar el pitch del audio
    y_tuned = librosa.effects.pitch_shift(y, sr = sr, n_steps = semitone_shift)

    # Guardar el archivo ajustado
    sf.write(afinado, y_tuned, sr)
    print(f"Archivo afinado guardado en: {afinado}")

def main():
    tune_audio(audio, "audio_prueba_tune.wav")

if __name__ == "__main__":
    main()