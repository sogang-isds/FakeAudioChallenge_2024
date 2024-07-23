import json
from tqdm import tqdm
import soundfile as sf
import numpy as np
import os
from common import APP_ROOT
import pyloudnorm
from scipy.signal import resample_poly


json_file = os.path.join(APP_ROOT, 'metadata.json')
out_dir = os.path.join(APP_ROOT, 'data/mixture')
sample_rate = 8000

with open(json_file, "r") as f:
    total_meta = json.load(f)


def resample_and_norm(signal, orig, target, lvl):
    if orig != target:
        signal = resample_poly(signal, target, orig)

    # fx = (AudioEffectsChain().custom("norm {}".format(lvl)))
    # signal = fx(signal)

    meter = pyloudnorm.Meter(target, block_size=0.1)
    loudness = meter.integrated_loudness(signal)
    signal = pyloudnorm.normalize.loudness(signal, loudness, lvl)

    return signal


for mix in tqdm(total_meta):
    filename = mix["mixture_name"]
    sources_list = [x for x in mix.keys() if x != "mixture_name"]

    sources = {}
    maxlength = 0
    for source in sources_list:
        # read file optional resample it
        source_utts = []
        for utt in mix[source]:
            utt_fs = sf.SoundFile(utt["file"]).samplerate
            audio, fs = sf.read(utt["file"], start=int(0 * utt_fs),
                                stop=int(5 * utt_fs))

            # assert len(audio.shape) == 1, "we currently not support multichannel"
            if len(audio.shape) > 1:
                audio = audio[:, utt["channel"]]  # TODO
            audio = audio - np.mean(audio)  # zero mean cos librispeech is messed up sometimes
            # if source != "noise":
            # audio = resample_and_norm(audio, fs, sample_rate, utt["lvl"])
            audio = np.pad(audio, (int(utt["start"] * sample_rate), 0), "constant")  # pad the beginning
            source_utts.append(audio)
            maxlength = max(len(audio), maxlength)

        sources[source] = source_utts

    # pad everything to same length
    for s in sources.keys():
        for i in range(len(sources[s])):
            tmp = sources[s][i]
            sources[s][i] = np.pad(tmp, (0, maxlength - len(tmp)), 'constant')

    # mix n sum
    tot_mixture = None
    for indx, s in enumerate(sources.keys()):
        if s == "noise":
            continue
        source_mix = np.sum(sources[s], 0)
        
        if indx == 0:
            tot_mixture = source_mix
        else:
            tot_mixture += source_mix
   
    s = "noise"
    source_mix = np.sum(sources[s], 0)
    
    tot_mixture += source_mix
    os.makedirs(out_dir, exist_ok=True)
    sf.write(os.path.join(out_dir, filename + ".wav"), tot_mixture, sample_rate)
