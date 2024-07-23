import glob
import os
import shutil

import numpy as np
import librosa
import matplotlib.pyplot as plt

from common import APP_ROOT
import nemo.collections.asr as nemo_asr

# sample rate, Hz
# SAMPLE_RATE = 16000
SAMPLE_RATE = 8000

vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained('vad_marblenet')

from omegaconf import OmegaConf
import copy

# Preserve a copy of the full config
cfg = copy.deepcopy(vad_model._cfg)
print(OmegaConf.to_yaml(cfg))

vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)
# Set model to inference mode
vad_model.eval()
vad_model = vad_model.to(vad_model.device)

import wave

from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader


# simple data layer to pass audio signal
class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)

    def set_signal(self, signal):
        self.signal = signal.astype(np.float32) / 32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1


data_layer = AudioDataLayer(sample_rate=cfg.train_ds.sample_rate)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

# inference method for audio signal (single instance)
def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(vad_model.device), audio_signal_len.to(vad_model.device)
    logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
    return logits


# class for streaming frame-based VAD
# 1) use reset() method to reset FrameVAD's state
# 2) call transcribe(frame) to do VAD on
#    contiguous signal's frames
# To simplify the flow, we use single threshold to binarize predictions.
class FrameVAD:

    def __init__(self, model_definition,
                 threshold=0.5,
                 frame_len=2, frame_overlap=2.5,
                 offset=10):
        '''
        Args:
          threshold: If prob of speech is larger than threshold, classify the segment to be speech.
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')

        self.sr = model_definition['sample_rate']
        self.threshold = threshold
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMFCCPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.buffer = np.zeros(shape=2 * self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()

    def _decode(self, frame, offset=0):
        assert len(frame) == self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = infer_signal(vad_model, self.buffer).cpu().numpy()[0]
        decoded = self._greedy_decoder(
            self.threshold,
            logits,
            self.vocab
        )
        return decoded

    @torch.no_grad()
    def transcribe(self, frame=None):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        return unmerged

    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer = np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(threshold, logits, vocab):
        s = []
        if logits.shape[0]:
            probs = torch.softmax(torch.as_tensor(logits), dim=-1)
            probas, _ = torch.max(probs, dim=-1)
            probas_s = probs[1].item()
            preds = 1 if probas_s >= threshold else 0
            s = [preds, str(vocab[preds]), probs[0].item(), probs[1].item(), str(logits)]
        return s


def offline_inference(wave_file, STEP=0.025, WINDOW_SIZE=0.5, threshold=0.5):
    FRAME_LEN = STEP  # infer every STEP seconds
    CHANNELS = 1  # number of audio channels (expect mono signal)
    RATE = 16000  # sample rate, Hz

    CHUNK_SIZE = int(FRAME_LEN * SAMPLE_RATE)

    vad = FrameVAD(model_definition={
        'sample_rate': SAMPLE_RATE,
        'AudioToMFCCPreprocessor': cfg.preprocessor,
        'JasperEncoder': cfg.encoder,
        'labels': cfg.labels
    },
        threshold=threshold,
        frame_len=FRAME_LEN, frame_overlap=(WINDOW_SIZE - FRAME_LEN) / 2,
        offset=0)

    wf = wave.open(wave_file, 'rb')

    empty_counter = 0

    preds = []
    proba_b = []
    proba_s = []

    data = wf.readframes(CHUNK_SIZE)

    while len(data) > 0:

        data = wf.readframes(CHUNK_SIZE)
        signal = np.frombuffer(data, dtype=np.int16)
        result = vad.transcribe(signal)

        preds.append(result[0])
        proba_b.append(result[2])
        proba_s.append(result[3])

        if len(result):
            # print(result, end='\n')
            empty_counter = 3
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                print(' ', end='')

    # p.terminate()
    vad.reset()

    return preds, proba_b, proba_s


data_dir = os.path.join(APP_ROOT, 'data/02_noise')
audio_paths = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)

print(f'len(audio_paths): {len(audio_paths)}')

output_s_dir = os.path.join(APP_ROOT, 'data/marblenet_s')
output_n_dir = os.path.join(APP_ROOT, 'data/marblenet_n')

os.makedirs(output_s_dir, exist_ok=True)
os.makedirs(output_n_dir, exist_ok=True)

for audio_path in audio_paths:
    wave_file = audio_path

    CHANNELS = 1
    # RATE = 16000
    RATE = 8000
    audio, sample_rate = librosa.load(wave_file, sr=RATE)
    dur = librosa.get_duration(audio, sr=sample_rate)
    print(dur)

    threshold=0.5

    results = []

    STEP_LIST = [0.05]
    WINDOW_SIZE_LIST = [0.25]

    for STEP, WINDOW_SIZE in zip(STEP_LIST, WINDOW_SIZE_LIST, ):
        print(f'====== STEP is {STEP}s, WINDOW_SIZE is {WINDOW_SIZE}s ====== ')
        preds, proba_b, proba_s = offline_inference(wave_file, STEP, WINDOW_SIZE, threshold)
        results.append([STEP, WINDOW_SIZE, preds, proba_b, proba_s])

    # exit()
    num = len(results)
    speech_ranges = []

    for i in range(num):
        step = STEP_LIST[i]
        win_size = WINDOW_SIZE_LIST[i]
        len_pred = len(results[i][2])

        pred = results[i][2]

        last_label = 0
        start_pos = -0.1
        end_pos = -0.1

        count = 0
        for j, label in enumerate(pred):
            if label == 1 and last_label == 0:
                start_pos = j
                last_label = 1

            if label == 0 and last_label == 1:
                end_pos = j
                last_label = 0

            if start_pos >= 0 and end_pos > 0:
                count += 1
                print(f'({count}) speech : {start_pos * step:.2f} to {end_pos * step:.2f}')
                speech_ranges.append((start_pos * step, end_pos * step))
                start_pos = -0.1
                end_pos = -0.1

    speech_sum = 0
    for speech_range in speech_ranges:
        speech_sum += speech_range[1] - speech_range[0]

    speech_ratio = speech_sum / dur
    print(f"speech duration: {speech_sum:.2f}")
    print(f"speech ratio: {speech_ratio:.2f}")

    dir_num = audio_path.split('/')[-2]
    filename, file_ext = os.path.splitext(os.path.basename(audio_path))

    if speech_ratio >= 0.2:
        # copy file
        file = f'{filename}_{dir_num}_{speech_ratio:.2f}{file_ext}'
        output_s_path = os.path.join(output_s_dir, file)
        shutil.copyfile(audio_path, output_s_path)

    else:
        file = f'{filename}_{dir_num}_{speech_ratio:.2f}{file_ext}'
        output_n_path = os.path.join(output_n_dir, file)
        shutil.copyfile(audio_path, output_n_path)

