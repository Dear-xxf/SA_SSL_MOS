import os.path
import torch
from dataset import Nisqa, AudioMos, Tencent, TCD_VOIP, get_dataloader
from torch.utils.data import Dataset
import logging
from typing import Type
import gin
import numpy as np
import tqdm
import scipy.stats as stats
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--gin_path', type=str, help='Path to gin config.')
parser.add_argument('--model_path', type=str, help='Path to the model.')
args = parser.parse_args()


@gin.configurable
class TestingLoop:
    def __init__(
        self,
        model_path: str = '',
        dataset_cls: Type[Dataset] = AudioMos,
        batch_size_test: int = 1,
        is_baseline: bool = True,
    ):
        self.model_path = model_path
        self.dataset_cls = dataset_cls
        self.batch_size_test = batch_size_test
        self.is_baseline = is_baseline

        self.data_type = 'ssl' if self.is_baseline else 'mixed'

        self.dataset = self.dataset_cls(
            features_folder='feature_w2v2_xlsr_2b_layer8',
            data_type=self.data_type
        )
        self.dataloader = get_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size_test,
            num_workers=12,
            shuffle=True
        )

    def evaluate(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dir = os.path.dirname(self.model_path)
        model = torch.jit.load(self.model_path).to(device)
        model.eval()
        predictions, labels, freqs = [], [], []
        # system_predictions = {system: [] for system in self.dataloader.dataset.unique_systems}
        # system_labels = copy.deepcopy(system_predictions)

        for batch in tqdm.tqdm(self.dataloader, desc='Evaluating', unit='batch'):
            wav, sys, data, label = batch
            # sys_name = sys[0]
            wav = wav.to(device)
            data = data.to(device)
            label = label.to(device)

            with torch.no_grad():
                prediction = model(data) if self.is_baseline else model(data, wav)

                if prediction.shape[1] == 2:
                    prediction = prediction[:, 0]
                else:
                    prediction = prediction.squeeze(-1)

            predictions.extend(prediction.cpu().tolist())
            labels.extend(label.cpu().tolist())
            # system_predictions[sys_name].extend(prediction.cpu().tolist())
            # system_labels[sys_name].extend(label.tolist())

            import re
            for s in sys:
                match = re.search(r'(16k|24k|48k)', s)
                freqs.append(match.group(1) if match else "unknown")

        predictions = np.array(predictions)
        labels = np.array(labels)
        freqs = np.array(freqs) 

        mse = np.mean((labels - predictions) ** 2)
        pcc = np.corrcoef(labels, predictions)[0][1]
        srcc = stats.spearmanr(labels, predictions)[0]

        # system_predictions_array = np.array([np.mean(scores) for scores in system_predictions.values()])
        # system_labels_array = np.array([np.mean(scores) for scores in system_labels.values()])
        # sys_mse=np.mean((system_labels_array-system_predictions_array)**2)
        # sys_pcc=np.corrcoef(system_labels_array, system_predictions_array)[0][1]
        # sys_srcc=stats.spearmanr(system_labels_array, system_predictions_array)[0]

        save_path = os.path.join(model_dir, self.dataset.__str__())
        if os.path.exists(save_path) is False:
            os.makedirs(save_path)
        logging.basicConfig(filename=os.path.join(save_path, 'test.log'), level=logging.INFO)
        logging.info(f"[TEST][UTT][ MSE = {mse:.4f} | PCC = {pcc:.4f} | SRCC = {srcc:.4f} ]")
        # logging.info(f"[TEST][SYS][ MSE = {sys_mse:.4f} | LCC = {sys_pcc:.4f} | SRCC = {sys_srcc:.4f} ]")

        colors = {
            "16k": "#1f77b4",
            "24k": "#ff7f0e",
            "48k": "#2ca02c",
            "unknown": "#7f7f7f"
        }
        for freq in np.unique(freqs):
            idx = freqs == freq
            plt.scatter(labels[idx], predictions[idx],
                label=freq,
                s=20,
                c=colors.get(freq, "#000000"),
                alpha=1)   

        plt.xlabel('MOS')
        plt.ylabel("Prediction")
        plt.title("Predictions vs Ground Truth")
        plt.xlim([0.9, 5.1])
        plt.ylim([0.9, 5.1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(title="Frequency")
        plt.savefig(os.path.join(save_path, "test_scatter.png"))
        plt.close()


def main():
    gin.parse_config_file(args.gin_path)

    testloop = TestingLoop(
        model_path=args.model_path
    )
    testloop.evaluate()


if __name__ == '__main__':
    main()
