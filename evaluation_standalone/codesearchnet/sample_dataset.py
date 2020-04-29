from torch.utils.data import Dataset
import os
import json


class SampleDataset(Dataset):
    def __init__(self, sample_path, directory='test', number_of_samples=None):
        self.samples = []
        counter = 0

        for file in os.listdir(os.path.join(sample_path, directory)):
            if number_of_samples and counter >= number_of_samples:
                break
            if ".gz" in file:
                continue

            assert os.path.isfile(os.path.join(sample_path, directory, file))
            with open(os.path.join(sample_path, directory, file)) as f:
                for i, line in enumerate(f.readlines(), 1):
                    if number_of_samples and counter >= number_of_samples:
                        break
                    self.samples.append(json.loads(line))
                    counter += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
