import librosa
import random
import numpy as np
import os
import tqdm
import re

import torch
import torchaudio
from transformers import Wav2Vec2Model, AutoModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
# from transformers import AutoProcessor, Wav2Vec2BertModel

from pathlib import Path
import argparse


def load_audio(audio_path: str, target_sr: int) -> np.ndarray:
    samples, rate = librosa.load(audio_path) # by default rate <= 22050kHz
    samples = librosa.resample(y=samples, orig_sr=rate, target_sr=target_sr, res_type='scipy')
    return samples


def repetitive_crop(samples: np.ndarray, length: int) -> np.ndarray:
    new_samples = samples
    while len(new_samples) < length:
        new_samples = np.concatenate((new_samples, samples), axis=0)
    if len(new_samples) > length:
        rand_start = random.randrange(0, len(new_samples)-length)
        new_samples = new_samples[rand_start:length+rand_start]
    return new_samples


def process_audio(audio_path: str, target_sr: int = 16_000, duration: float = 10.0) -> torch.Tensor:
	samples = load_audio(audio_path, target_sr)
	samples = repetitive_crop(samples, int(duration*target_sr))
	samples = torch.Tensor(samples)
	assert len(samples) == duration*target_sr
	return torch.unsqueeze(samples, 0)

def extract_epoch(input_string, dataset):
    pattern = rf'w2v2_base_{dataset}_finetuned_(\d+)epoch'
    match = re.search(pattern, input_string)
    if match:
        return int(match.group(1))
    else:
        return None
    

class ModelWithBuffer():
    def __init__(self, model):
        self.model = model
        self.feature_buffer = []
        for layer in self.model._modules['encoder']._modules['layers']:
            layer.register_forward_hook(self.layer_hook)

    def layer_hook(self, module, input, output):
        emb = output[0].detach()#.cpu().numpy().squeeze().T
        self.feature_buffer.append(emb)

    def extract_features(self, samples: torch.Tensor) -> list[torch.Tensor]:
        self.feature_buffer.clear()

        _ = self.model(samples)
        return self.feature_buffer.copy(), None


def get_ssl_model(model_name: str):
    model_sr = 16_000
    match model_name:
        #===========================Not Fine-tuned============================
        # models pre-trained on 960hr LibriSpeech
        case 'w2v2_base':
            model = torchaudio.pipelines.WAV2VEC2_BASE.get_model() # 12 layers
        case 'w2v2_large':
            model = torchaudio.pipelines.WAV2VEC2_LARGE.get_model()
        case 'hubert_base':
            model = torchaudio.pipelines.HUBERT_BASE.get_model() # 12 layers
        case 'wavlm_base':
            model = torchaudio.pipelines.WAVLM_BASE.get_model() # 12 layers
        # models pre-trained on 60khr Libri-Light
        case 'w2v2_large_lv60k':
            model = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K.get_model()
        case 'hubert_large':
            model = torchaudio.pipelines.HUBERT_LARGE.get_model()
        case 'hubert_xlarge':
            model = torchaudio.pipelines.HUBERT_XLARGE.get_model() # 1280-dim feature
        case 'wavlm_base_plus':
            model = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model()
        case 'wavlm_large':
            model = torchaudio.pipelines.WAVLM_LARGE.get_model() # 1024-dim feature
        # models pre-trained on 436khr multi-datasets
        case 'w2v2_xlsr_300m':
            model = torchaudio.pipelines.WAV2VEC2_XLSR_300M.get_model() # 1024-dim feature, 24 layers
        case 'w2v2_xlsr_1b':
            model = torchaudio.pipelines.WAV2VEC2_XLSR_1B.get_model() # 1280-dim feature, 48 layers
        case 'w2v2_xlsr_2b':
            model = torchaudio.pipelines.WAV2VEC2_XLSR_2B.get_model() # 1920-dim feature, 48 layers
        #===========================ASR Fine-tuned============================
        case 'w2v2_large_xlsr53':
            model = AutoModel.from_pretrained('facebook/wav2vec2-large-xlsr-53')
            model = ModelWithBuffer(model)
        # https://huggingface.co/jonatasgrosman
        case 'w2v2_large_asr_en':
            model = AutoModel.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english')
            model = ModelWithBuffer(model)
        case 'w2v2_large_asr_de':
            model = AutoModel.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-german')
            model = ModelWithBuffer(model)
        case 'w2v2_large_asr_zh':
            model = AutoModel.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
            model = ModelWithBuffer(model)
        #============================Custom Data Fine-tuned====================
        case _ if re.fullmatch(r'w2v2_base_bvcc_finetuned_\d+epoch', model_name):
            ckpt_num = 39*extract_epoch(model_name, 'bvcc')
            model = AutoModel.from_pretrained(f'/proj/berzelius-2023-179/users/projects/finetune_ssl/bvcc_finetuned_10epoch/checkpoint-{ckpt_num}')
            model = ModelWithBuffer(model)
        case _ if re.fullmatch(r'w2v2_base_tencent_finetuned_\d+epoch', model_name):
            ckpt_nums = [62, 125, 187, 250, 312, 375, 437, 500, 562, 620]
            ckpt_num = ckpt_nums[extract_epoch(model_name, 'tencent')-1]
            model = AutoModel.from_pretrained(f'/proj/berzelius-2023-179/users/projects/finetune_ssl/tencent_finetuned_10epoch/checkpoint-{ckpt_num}')
            model = ModelWithBuffer(model)
        # case 'w2v2_base_bvcc_finetuned_1epoch':
        #     model = AutoModel.from_pretrained('/proj/berzelius-2023-179/users/projects/finetune_ssl/bvcc_finetuned_3epoch/checkpoint-39')
        #     model = ModelWithBuffer(model)
        # case 'w2v2_base_bvcc_finetuned_2epoch':
        #     model = AutoModel.from_pretrained('/proj/berzelius-2023-179/users/projects/finetune_ssl/bvcc_finetuned_3epoch/checkpoint-78')
        #     model = ModelWithBuffer(model)
        # case 'w2v2_base_bvcc_finetuned_3epoch':
        #     model = AutoModel.from_pretrained('/proj/berzelius-2023-179/users/projects/finetune_ssl/bvcc_finetuned_3epoch/checkpoint-117')
        #     model = ModelWithBuffer(model)
        # case 'w2v2_base_tencent_finetuned_1epoch':
        #     model = AutoModel.from_pretrained('/proj/berzelius-2023-179/users/projects/finetune_ssl/tencent_finetuned_3epoch/checkpoint-62')
        #     model = ModelWithBuffer(model)
        # case 'w2v2_base_tencent_finetuned_2epoch':
        #     model = AutoModel.from_pretrained('/proj/berzelius-2023-179/users/projects/finetune_ssl/tencent_finetuned_3epoch/checkpoint-125')
        #     model = ModelWithBuffer(model)
        # case 'w2v2_base_tencent_finetuned_3epoch':
        #     model = AutoModel.from_pretrained('/proj/berzelius-2023-179/users/projects/finetune_ssl/tencent_finetuned_3epoch/checkpoint-186')
        #     model = ModelWithBuffer(model)
        # TODO: Fix bugs.
        # case 'mms-300m':
        #     model = Wav2Vec2Model.from_pretrained("facebook/mms-300m") # 24 layer, 1024-dim feature
        #     model = ModelWithBuffer(model)
        # case 'mms-1b':
        #     model = Wav2Vec2Model.from_pretrained("facebook/mms-1b") # 48 layer, 1280-dim feature
        #     model = ModelWithBuffer(model)
        # case 'w2v2_base':
        #     model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h") # 12 layer, 768-dim feature
        #     model = ModelWithBuffer(model)
        case _:
            raise ValueError('')
    return model, model_sr


def write_output(output_path: str, model_name: str, audio_path: str, features: dict[str, np.ndarray]):
    # file_name = audio_path.split("\\")[-1][:-4]
    file_name = Path(audio_path).stem
    for layer_name, layer_feature in features.items():
        # folder_path = os.path.join(output_path, model_name, layer_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        full_output_path = os.path.join(output_path, file_name+'.npy')
        np.save(full_output_path, layer_feature)


def extract_features_audio(audio_paths: list[str], layers: list[int], model_name: str, signal_duration: float, save_output: bool=False, output_path: str='/proj/berzelius-2023-179/users/datasets/BVCC/DATA/') -> dict[str, dict[str, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_sr = get_ssl_model(model_name)
    if isinstance(model, ModelWithBuffer):
        model.model = model.model.to(device)
    else:
        model = model.to(device)

    audio_features = {}

    for audio_path in tqdm.tqdm(audio_paths):
        samples = process_audio(audio_path, model_sr, signal_duration).to(device)

        selected_features = {}

        with torch.inference_mode():
            features, _ = model.extract_features(samples)

        for i in range(len(features)):
            if i not in layers:
                continue
            selected_features[f'layer{i}']=features[i].squeeze().T.cpu().numpy()

        if save_output:
            write_output(output_path, model_name, audio_path, selected_features)
        else:
            audio_features[audio_path] = selected_features
  
    return audio_features


import glob

parser = argparse.ArgumentParser(description='Gin and save path.')
parser.add_argument('--wav_path', type=str, required=True, help='Path to the wav.')
parser.add_argument('--save_path', type=str, required=True, help='Path to directory storing results.')

args = parser.parse_args()

audio_dir = args.wav_path
output_feature_dir = args.save_path

model_name = 'w2v2_xlsr_2b'
signal_duration = 10
selected_layers = [8]

audio_paths = glob.glob(os.path.join(audio_dir, '*.wav'))

extract_features_audio(
    audio_paths=audio_paths,
    layers=selected_layers,
    model_name=model_name,
    signal_duration=signal_duration,
    save_output=True,
    output_path=output_feature_dir
)













