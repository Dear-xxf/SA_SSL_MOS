# SA_SSL_MOS
This is the official implementation of "SA-SSL-MOS: SELF-SUPERVISED LEARNING MOS PREDICTION WITH SPECTRAL AUGMENTATION FOR GENERALIZED MULTI-RATE SPEECH ASSESSMENT". SA_SSL_MOS is a speech quality assessment model designed to handle cross-sampling-rate datasets, taking as input a speech clip along with the SSL features extracted from the same utterance and outputting a Gaussian mean opinion score (MOS) distribution.

Authors: Fengyuan Cao, Xinyu Liang, Fredrik Cumlin
Emails: [fencao@kth.se](mailto: fencao@kth.se), [hopeliang990504@gmail.com](mailto:hopeliang990504@gmail.com), [fcumlin@gmail.com](mailto:fcumlin@gmail.com)

## Inference

The released checkpoint is trained as follows:

- Pre-trained on the **NISQA** dataset for 30 epochs
- Fine-tuned on the **AudioMOS_train** dataset for 3 epochs
- The best-performing checkpoint during fine-tuning is selected for evaluation

This model achieves the best performance after the fine-tuning stage.

To use the code, first you should generate ssl features.

```python
python extract_ssl_features.py --wav_path 'your_wav_directory_path' --save_path 'your_save_directory_path'
```

After generating ssl features, you can use the following code to run inference:

```python
import numpy as np
import torch
import audio
import librosa
import utils

model = torch.jit.load('model path', map_location=torch.device('cpu'))

wav_path = 'path_to_your_audio_file.wav'  # Replace with your audio file path.
ssl_path = 'path_to_your_SSL_path.npy'  # Replace with your SSL feature file path.

# generate the STFT and load the SSL features
wav, sr = librosa.load(wav_path, sr=48000)
signal = audio.Audio(wav, sr)
signal = signal.repetitive_crop(10 * sr)
samples = np.squeeze(signal.samples)
spec = torch.FloatTensor(utils.stft(samples))
ssl_feature = torch.FloatTensor(np.load(ssl_path))

with torch.no_grad():
    prediction = model(ssl_data = ssl_feature[None, ...], stft_data = spec[None, ...])
mean = prediction[:, 0]
variance = prediction[:, 1]
print(f'{mean=}, {variance=}')
```

## Installation

Installation with pip:

```
pip install -r requirements.txt
```

## Dataset

Training process mainly use two datasets. [NISQA](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus) and [AudioMOS2025_train](https://sites.google.com/view/voicemos-challenge/audiomos-challenge-2025) (track 3).

The dataset structure is 

```
datasets/
│
├── NISQA_Corpus/
│   ├── NISQA_TRAIN_SIM/
│		├── deg/
│			├── ssl_feature_folder/ # e.g., feature_w2v2_xlsr_2b_layer8
│			└── ***.wav
│		├── ref/
│		└── NISQA_TRAIN_SIM_file.csv
│	├── NISQA_TEST_SIM/
│   └── ...
│
├── audiomos2025_train/
│   ├── DATA/
│   	├── wav/
│   	├── ssl_feature_folder/ # e.g., feature_w2v2_xlsr_2b_layer8
│   	├── train.csv
│   	└── val.csv
│
├── audiomos2025_eval/
│   ├── DATA/
│   	├── wav/
│   	├── ssl_feature_folder/ # e.g., feature_w2v2_xlsr_2b_layer8
│   	└── eval.csv
```



## Training

The framework is Gin configurable; specifying model and dataset is done with a Gin config. See config files in gin_config/*.gin.

```
python src/train.py \
    --gin_path gin_config/train_config.gin \
    --save_path results/ \
    --features_folder <your_feature_folder>  # e.g., feature_w2v2_xlsr_2b_layer8
```

