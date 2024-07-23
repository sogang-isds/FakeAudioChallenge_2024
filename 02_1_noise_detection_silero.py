import os
from tqdm import tqdm
import glob
import shutil
import torch

USE_ONNX = True

# model = load_silero_vad()
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                force_reload=True,
                                onnx=USE_ONNX)

(get_speech_timestamps,
  save_audio,
  read_audio,
  VADIterator,
  collect_chunks) = utils
 
data_dir = "data/unlabeled_data"

audio_paths = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)

noise_count = 0

for audio_path in tqdm(audio_paths):
    # print(f"Processing {audio_path}")

    wav = read_audio(audio_path, sampling_rate=8000)  # backend (sox, soundfile, or ffmpeg) required!
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=8000)

    # copy file to tmp/vad
    if len(speech_timestamps) == 0:
        # copy wav using shutil
        out_path = audio_path.replace('unlabeled_data', 'noise')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        shutil.copyfile(audio_path, out_path)

        noise_count += 1

print(f"Total noise files: {noise_count}")
