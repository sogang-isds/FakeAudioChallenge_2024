import os
import shutil

import librosa



data_dir = "data/02_noise"


def detect_no_noise(file_path):
    # 오디오 파일 로드
    y, sr = librosa.load(file_path, sr=None)

    # 프레임 단위로 RMS 에너지 계산
    frame_length = 1024
    hop_length = 512
    rms_list = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    rms_list = list(rms_list[0])
    # count 0 elems in rms_list
    zero_count = 0
    for rms in rms_list:
        if rms == 0.0:
            zero_count += 1

    zero_ratio = zero_count / len(rms_list)

    if zero_ratio >= 0.01:
        return zero_ratio

    return False


import glob
output_dir = 'data/no_noise'
os.makedirs(output_dir, exist_ok=True)

audio_paths = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)

silence_count = 0

for audio_path in audio_paths:
    print(f"Processing {audio_path}")
    result = detect_no_noise(audio_path)

    if result:
        dir_num = audio_path.split('/')[-2]
        filename, file_ext = os.path.splitext(os.path.basename(audio_path))
        file = f'{filename}_{dir_num}_{result:.2f}{file_ext}'
        output_s_path = os.path.join(output_dir, file)
        shutil.copyfile(audio_path, output_s_path)

        # remove file
        # os.remove(audio_path)

        silence_count += 1

print(f"Total silence files: {silence_count}")
