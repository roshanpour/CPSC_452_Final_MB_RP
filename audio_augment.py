import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# make the directories
INPUT_DIR = 'Data/genres_original'
OUTPUT_DIR = 'Data/genres_augmented'
AUG_PER_FILE = 5

# make the funciton that randomly applies the augmentations
def augment_audio(y, sr):
    if np.random.rand() < 0.5:
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 0.95))
    if np.random.rand() < 0.5:
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=np.random.randint(-2, 3))
    if np.random.rand() < 0.5:
        y += np.random.normal(0, 0.005, y.shape)
    if np.random.rand() < 0.5:
        y *= np.random.uniform(0.8, 1.2)
    return y

# loop through the original data and make a directory that includes the augmented files
for genre in os.listdir(INPUT_DIR):
    genre_path = os.path.join(INPUT_DIR, genre)
    if not os.path.isdir(genre_path):
        continue

    output_genre_dir = os.path.join(OUTPUT_DIR, genre)
    os.makedirs(output_genre_dir, exist_ok=True)

    for filename in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
        if not filename.lower().endswith('.wav'):
            continue

        input_path = os.path.join(genre_path, filename)
        try:
            y, sr = librosa.load(input_path, sr=None)
        except Exception as e:
            print(f"Skipping {input_path}: {e}")
            continue

        sf.write(os.path.join(output_genre_dir, filename), y, sr)

        base, ext = os.path.splitext(filename)
        for i in range(AUG_PER_FILE):
            y_aug = augment_audio(y.copy(), sr)
            new_filename = f"{base}_aug{i+1}{ext}"
            output_path = os.path.join(output_genre_dir, new_filename)
            y_aug = augment_audio(y.copy(), sr)

            if y_aug is None or len(y_aug) == 0:
                continue  

            y_aug = np.ascontiguousarray(y_aug).astype(np.float32)
            y_aug = np.clip(y_aug, -1.0, 1.0)

            new_filename = f"{base}_aug{i+1}{ext}"
            output_path = os.path.join(output_genre_dir, new_filename)
            sf.write(output_path, y_aug, sr)

