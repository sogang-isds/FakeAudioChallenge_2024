import glob
import os
import random

import librosa
import numpy as np

import pandas as pd

from common import APP_ROOT
from utils.utils import save_to_json


def get_audio_level(file_path):
    # 오디오 파일 로드
    y, sr = librosa.load(file_path, sr=None)

    # RMS 에너지 계산
    rms = librosa.feature.rms(y=y)

    # RMS를 dB로 변환
    db = librosa.amplitude_to_db(rms, ref=np.max)[0]

    # 평균 dB 계산
    avg_db = np.mean(db)

    return avg_db



noise_dir = "data/marblenet_n"
train_dir = "data/train"
noise_paths = glob.glob(os.path.join(APP_ROOT, noise_dir, "**/*.wav"), recursive=True)
train_paths = glob.glob(os.path.join(APP_ROOT, train_dir, "**/*.wav"), recursive=True)

# noise file

# for audio_path in noise_paths:
#     print(audio_path)

outputs = []
for audio_path in train_paths:
    print(audio_path)
    level = get_audio_level(audio_path)

    filename, file_ext = os.path.splitext(os.path.basename(audio_path))

    random_noise_idx = np.random.randint(0, len(noise_paths))
    noise_path = noise_paths[random_noise_idx]
    noise_level = get_audio_level(noise_path)

    # audio length
    y, sr = librosa.load(audio_path, sr=None)
    audio_length = librosa.get_duration(y=y, sr=sr)

    start = 0
    if len(y) < 5 * 8000:
        idx = 5 * 8000 - len(y)
        start = random.randint(0, idx) / 8000

    mixture_dict = {
        'mixture_name': filename,
        's1': [
            {
                'file': audio_path,
                'lvl': float(level),
                'start': start,
            }
        ],
        'noise': [
            {
                'file': noise_path,
                'lvl': float(noise_level),
                'start': 0
            }
        ]
    }

    outputs.append(mixture_dict)

# save to json
save_to_json(outputs, 'metadata.json')

print("Finish")