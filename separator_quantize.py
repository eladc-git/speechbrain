from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import numpy as np
import torch
import model_compression_toolkit as mct


model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')


# for custom file, change path
audio_path = "speechbrain/sepformer-wsj02mix/test_mixture.wav"
#audio_path = "61-70968-0000_8455-210777-0012_noisy.wav"
est_sources = model.separate_file(path=audio_path, chunk_size=-1)
torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu()/2, 8000, bits_per_sample=32)
torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu()/2, 8000, bits_per_sample=32)
print("done!")
