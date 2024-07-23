import glob
import os

from common import APP_ROOT

data_dir = os.path.join(APP_ROOT, "data/sample_data/unlabeled_data")
out_file = os.path.join(APP_ROOT, 'data/wav.scp')

audio_paths = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)

with open(out_file, 'w') as f:
    for audio_path in audio_paths:
        filename, file_extension = os.path.splitext(os.path.basename(audio_path))
        print(audio_path)
        
        f.write(f'{filename} {audio_path}\n')
        
print('Finish')