from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import numpy as np
import torch
import model_compression_toolkit as mct
from model_compression_toolkit.core.pytorch.utils import to_torch_tensor, get_working_device

# ------------------------------- #
# Configuration
# ------------------------------- #
QUANTIZATION = False
CHUNK_SIZE = 9810
DEVICE = get_working_device()

# ------------------------------- #
# Model
# ------------------------------- #
model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='pretrained_models/sepformer-wham16k-enhancement')

# Run sanity
x = np.random.random((1, CHUNK_SIZE))
x_torch = torch.from_numpy(x).float()
y = model(x_torch)

# ------------------------------- #
# Quantization
# ------------------------------- #
if QUANTIZATION:
    def representative_data_gen() -> list:
        return [np.random.random((1,CHUNK_SIZE))]

    target_platform_cap = mct.get_target_platform_capabilities('pytorch', 'default')
    quantization_config = mct.DEFAULTCONFIG
    quantized_model, quantization_info = mct.pytorch_post_training_quantization(model,
                                                                                representative_data_gen,
                                                                                target_platform_capabilities=target_platform_cap,
                                                                                n_iter=20)


# ------------------------------- #
# Speech Separation
# ------------------------------- #
audio_path = "61-70968-0000_8455-210777-0012_noisy.wav"
est_sources = model.separate_file(path=audio_path, chunk_size=CHUNK_SIZE)
torchaudio.save("61-70968-0000_8455-210777-0012_enhanced.wav", est_sources[:, :, 0].detach().cpu()/2, sample_rate=16000, bits_per_sample=16)
print("done!")
