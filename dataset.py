import os

import audio
import utils

import gin
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import tqdm
from collections import OrderedDict

import librosa

@gin.configurable
class Nisqa(Dataset):
    """The NISQA_Corpus dataset."""
    
    def __init__(
        self,
        data_path: str = '../../datasets/NISQA_Corpus',
        features_folder: str = 'feature_w2v2_xlsr_2b_layer8',
        valid: str = 'train',
        test_set_name: str = '',
        debug: bool = False,
        data_type: str = 'mixed',
    ):
        """Initializes the instance.
        
        Args:
            data_path: Path to the dataset.
            valid: The data type. Can be 'train', 'val', or 'test'.
        """
        self._data_path = data_path
        self._features_folder = features_folder
        self._debug = debug
        self._test_set_name = test_set_name

        self._df = self._load_df(valid)
        self._num_samples = len(self._df)
        self._valid = valid
        self._data_type = data_type

        self._waves, self._systems, self._features, self._labels = self._load_clips()
    
    @property
    def features_shape(self) -> int:
        return self._features[0].shape
    
    @property
    def stft_shape(self) -> int:
        return self._waves[0].shape

    def _load_df(self, valid: str) -> pd.DataFrame:
        if valid == 'train':
            train_sim_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_TRAIN_SIM', 'NISQA_TRAIN_SIM_file.csv'))
            train_live_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_TRAIN_LIVE', 'NISQA_TRAIN_LIVE_file.csv'))
            df = pd.concat([train_sim_df, train_live_df])
        elif valid == 'val':
            val_sim_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_VAL_SIM', 'NISQA_VAL_SIM_file.csv'))
            val_live_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_VAL_LIVE', 'NISQA_VAL_LIVE_file.csv'))
            df = pd.concat([val_sim_df, val_live_df])
        elif valid == 'test':
            if self._test_set_name == 'FOR':
                test_for_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_TEST_FOR', 'NISQA_TEST_FOR_file.csv'))
            elif self._test_set_name == 'LIVETALK':
                test_for_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_TEST_LIVETALK', 'NISQA_TEST_LIVETALK_file.csv'))
            elif self._test_set_name == 'P501':
                test_for_df = pd.read_csv(os.path.join(self._data_path, 'NISQA_TEST_P501', 'NISQA_TEST_P501_file.csv'))
            else:
                raise ValueError(f'{self._test_set_name} is not a valid test set name.')
            df = test_for_df
        else:
            raise ValueError(f'{valid} is not valid.')
        df = df[['db', 'filename_deg', 'mos']]
        df = df.rename(columns={'db': 'foldernames', 'filename_deg': 'filenames', 'mos': 'labels'})
        return df

    @property
    def unique_systems(self) -> set[str]:
        return set(self._df['systems'])

    def _load_clips(self):
        """Loads the clips, applies augmentations (if so), and transforms to spectrograms"""
        waves, systems, features, labels = [], [], [], []
        for splitfoldername, filename, label in tqdm.tqdm(zip(self._df['foldernames'], self._df['filenames'], self._df['labels']), total=self._num_samples, desc='Loading clips...'):
            try:
                feature = np.load(os.path.join(self._data_path, splitfoldername, self._features_folder,
                                               filename.replace('.wav', '.npy')))
                if self._data_type == 'mixed':
                    wav, sr = librosa.load(os.path.join(self._data_path, splitfoldername, 'deg', filename), sr=48000)
                    signal = audio.Audio(wav, sr)
                    signal = signal.repetitive_crop(10 * sr)
                    samples = np.squeeze(signal.samples)
                    spec = utils.stft(samples)
                    waves.append(spec)
                elif self._data_type == 'ssl':
                    waves.append(np.zeros(1))

                systems.append('system')
                features.append(feature)
                labels.append(label)
            except FileNotFoundError:
                print(f'Signal {filename} was not found.')

        return waves, systems, features, labels

    def __getitem__(self, idx: int):
        """Returns a spectrogram with label and augmentation applied."""
        return self._waves[idx], self._systems[idx], self._features[idx], self._labels[idx]
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._features)

    def collate_fn(self, batch: list):
        """Returns a batch consisting of tensors."""
        waves, systems, features, labels = zip(*batch)
        waves = torch.FloatTensor(np.array(waves))
        features = torch.FloatTensor(np.array(features))
        labels = torch.FloatTensor(labels)
        return waves, systems, features, labels

    def __str__(self):
        if self._valid != 'test':
            return "Nisqa_" + self._valid
        else:
            return "Nisqa_" + self._valid + '_' + self._test_set_name

@gin.configurable
class AudioMos(Dataset):
    def __init__(
        self,
        data_dir: str = '../../datasets/',
        valid: str = 'train',
        features_folder: str = 'feature_w2v2_xlsr_2b_layer8',
        data_type: str = 'mixed',
    ):
        """
        Args:
            data_dir (str): Root directory of the dataset, where all data files and feature folders are stored.
            valid (str): Data split type, options are 'train', 'val', or 'test'.
            features_folder (str): Name of the subfolder containing features,
            data_type (str): Type or processing mode of the data, only support 'mixed' (ssl features and stft features), 'ssl' (only ssl features).
        """
        self._data_dir = data_dir
        self._valid = valid
        self._features_folder = features_folder

        self._data_type = data_type

        if valid == 'train':
            self._data_path = os.path.join(self._data_dir, 'audiomos2025_train/DATA')
            csv_path = os.path.join(self._data_path, 'train.csv')
        elif valid == 'val':
            self._data_path = os.path.join(self._data_dir, 'audiomos2025_train/DATA')
            csv_path = os.path.join(self._data_path, 'val.csv')
        elif valid == 'test':
            self._data_path = os.path.join(self._data_dir, 'audiomos2025_eval/DATA')
            csv_path = os.path.join(self._data_path, 'eval.csv')
        else:
            raise ValueError(f"Unsupported valid: {valid}")

        self._df = pd.read_csv(csv_path)
        self._num_samples = len(self._df)
        self._waves, self._systems, self._features, self._labels = self._load_clips()

    @property
    def unique_systems(self) -> list:
        return list(OrderedDict.fromkeys(self._systems))

    @property
    def features_shape(self) -> int:
        return self._features[0].shape
    
    @property
    def stft_shape(self) -> int:
        return self._waves[0].shape

    def _load_clips(self):
        waves, systems, features, labels = [], [], [], []
        for system, filename, label in tqdm.tqdm(
                zip(self._df['sysID'], self._df['uttID'], self._df['rating']), total=self._num_samples,
                desc='Loading clips...'):
            try:
                feature = np.load(
                    os.path.join(self._data_path, self._features_folder, filename.replace('.wav', '.npy')))

                if self._data_type == 'mixed':
                    wav, sr = librosa.load(os.path.join(self._data_path, 'wav', filename), sr=48000)
                    signal = audio.Audio(wav, sr)
                    signal = signal.repetitive_crop(10 * sr)
                    samples = np.squeeze(signal.samples)
                    spec = utils.stft(samples)
                    waves.append(spec)
                elif self._data_type == 'ssl':
                    waves.append(np.zeros(1))

                systems.append(system)
                features.append(feature)
                labels.append(label)
            except FileNotFoundError:
                print(f'Signal {filename} was not found.')
        return waves, systems, features, labels

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx: int):
        """Returns a spectrogram with label and augmentation applied."""
        return self._waves[idx], self._systems[idx], self._features[idx], self._labels[idx]

    def collate_fn(self, batch: list):
        """Returns a batch consisting of tensors."""
        waves, systems, features, labels = zip(*batch)
        waves = torch.FloatTensor(np.array(waves))
        features = torch.FloatTensor(np.array(features))
        labels = torch.FloatTensor(labels)
        return waves, systems, features, labels
    
    def __str__(self):
        return "AudioMos_" + self._valid


@gin.configurable
class Tencent(Dataset):
    def __init__(
        self,
        data_dir: str = '../../datasets/TencentCorpus',
        with_reverberation: bool = False,
        features_folder: str = 'feature_w2v2_xlsr_2b_layer8',
        data_type: str = 'mixed',
    ):
        
        self._data_dir = data_dir
        self._features_folder = features_folder

        self._data_type = data_type

        if with_reverberation:
            self._data_path = os.path.join(self._data_dir, 'withReverberationTrainDev')
            csv_path = os.path.join(self._data_dir, 'withReverberationTrainDevMOS.csv')
        else:
            self._data_path = os.path.join(self._data_dir, 'withoutReverberationTrainDev')
            csv_path = os.path.join(self._data_dir, 'withoutReverberationTrainDevMOS.csv')

        self.with_reverberation = with_reverberation
        self._df = pd.read_csv(csv_path)
        self._num_samples = len(self._df)
        self._waves, self._systems, self._features, self._labels = self._load_clips()

    @property
    def features_shape(self) -> int:
        return self._features[0].shape
    
    @property
    def stft_shape(self) -> int:
        return self._waves[0].shape

    def _load_clips(self):
        waves, systems, features, labels = [], [], [], []
        for filepath, label in tqdm.tqdm(
                zip(self._df['deg_wav'], self._df['mos']), total=self._num_samples,
                desc='Loading clips...'):
            try:
                filename = filepath.split('/')[-1]
                feature = np.load(
                    os.path.join(self._data_path, self._features_folder, filename.replace('.wav', '.npy')))

                if self._data_type == 'mixed':
                    wav, sr = librosa.load(os.path.join(self._data_path, filename), sr=48000)
                    signal = audio.Audio(wav, sr)
                    signal = signal.repetitive_crop(10 * sr)
                    samples = np.squeeze(signal.samples)
                    spec = utils.stft(samples)
                    waves.append(spec)
                elif self._data_type == 'ssl':
                    waves.append(np.zeros(1))

                systems.append('system')
                features.append(feature)
                labels.append(label)
            except FileNotFoundError:
                print(f'Signal {filename} was not found.')
        return waves, systems, features, labels

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx: int):
        """Returns a spectrogram with label and augmentation applied."""
        return self._waves[idx], self._systems[idx], self._features[idx], self._labels[idx]

    def collate_fn(self, batch: list):
        """Returns a batch consisting of tensors."""
        waves, systems, features, labels = zip(*batch)
        waves = torch.FloatTensor(np.array(waves))
        features = torch.FloatTensor(np.array(features))
        labels = torch.FloatTensor(labels)
        return waves, systems, features, labels
    
    def __str__(self):
        if self.with_reverberation:
            return "Tencent_withReverberation"
        else:
            return "Tencent_withoutReverberation"


@gin.configurable
class TCD_VOIP(Dataset):
    def __init__(
        self,
        data_dir: str = '../../datasets/TCD-VOIP',
        features_folder: str = 'feature_w2v2_xlsr_2b_layer8',
        data_type: str = 'mixed',
    ):
        
        self._data_dir = data_dir
        self._data_path = os.path.join(self._data_dir, 'Test Set')
        self._features_folder = features_folder

        self._data_type = data_type

        csv_path = os.path.join(self._data_dir, 'metadata.csv')
        self._df = pd.read_csv(csv_path)
        self._df = self._df[['Filename', 'sample MOS']]

        self._num_samples = len(self._df)
        self._waves, self._systems, self._features, self._labels = self._load_clips()

    @property
    def features_shape(self) -> int:
        return self._features[0].shape
    
    @property
    def stft_shape(self) -> int:
        return self._waves[0].shape

    def _load_clips(self):
        waves, systems, features, labels = [], [], [], []
        for filename, label in tqdm.tqdm(
                zip(self._df['Filename'], self._df['sample MOS']), total=self._num_samples,
                desc='Loading clips...'):
            try:
                file_directory = self._extract_middle_label(filename)
                feature = np.load(
                    os.path.join(self._data_path, file_directory, self._features_folder, filename.replace('.wav', '.npy')))

                if self._data_type == 'mixed':
                    wav, sr = librosa.load(os.path.join(self._data_path, file_directory, filename), sr=48000)
                    signal = audio.Audio(wav, sr)
                    signal = signal.repetitive_crop(10 * sr)
                    samples = np.squeeze(signal.samples)
                    spec = utils.stft(samples)
                    waves.append(spec)
                elif self._data_type == 'ssl':
                    waves.append(np.zeros(1))

                systems.append('system')
                features.append(feature)
                labels.append(label)
            except FileNotFoundError:
                print(f'Signal {filename} was not found.')
        return waves, systems, features, labels

    def _extract_middle_label(self, filename: str, index: int = 2) -> str:
        parts = filename.split('_')
        if index >= len(parts):
            raise ValueError(f"Index {index} out of range for filename parts: {parts}")
        return parts[index].lower()

    def __len__(self):
        return len(self._features)

    def __getitem__(self, idx: int):
        """Returns a spectrogram with label and augmentation applied."""
        return self._waves[idx], self._systems[idx], self._features[idx], self._labels[idx]

    def collate_fn(self, batch: list):
        """Returns a batch consisting of tensors."""
        waves, systems, features, labels = zip(*batch)
        waves = torch.FloatTensor(np.array(waves))
        features = torch.FloatTensor(np.array(features))
        labels = torch.FloatTensor(labels)
        return waves, systems, features, labels
    
    def __str__(self):
        return 'TCD_VOIP'

@gin.configurable
def get_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Returns a dataloader of the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
