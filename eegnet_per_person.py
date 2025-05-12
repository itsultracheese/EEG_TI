import os
import mne
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

import torch
from braindecode import EEGClassifier
from braindecode.models import EEGNetv4

import warnings
warnings.filterwarnings('ignore')

S_FREQ = 500
LEN = 6

files = [x for x in os.listdir('/mnt/s3-data2/amiftakhova/eeg_ti/resampled/') if 'TI' in x]
persons = [x[3:5] for x in files]

list_epochs = []
for p in persons:
    epoch = mne.read_epochs(f'/mnt/s3-data2/amiftakhova/eeg_ti/resampled/TI_{p}.fif', verbose=False)
    list_epochs.append(epoch)

list_epochs[13] = list_epochs[13].filter(1, 30, verbose=False)

list_labels = []
for id, epochs in enumerate(list_epochs):
    list_labels.append(epochs.events[:, -1] - 2)

# add TS data to train

standard_1020 = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8',
    'Cz', 'C3', 'C4', 'T7', 'T8', 'P7', 'P8',
    'Pz', 'P3', 'P4', 'O1', 'O2'
]
around_c3 = ['C3', 'C1', 'C5', 'FC3', 'CP3', 'FCC5h', 'FCC3h', 'CCP5h', 'CCP3h']
selected = ['C3', 'CP3', 'P3', 'P4', 'O1', 'O2']

sfreq = 500
electrode = 'all'

batch_size = 64
n_epochs = 30
lr=1e-4

scores = {}

resampled = list_epochs

batch_sizes = [8, 16]
lrs = [5e-4, 1e-3]

n_chans = 127
new_resampled = []
if electrode == '10-20':
    for i in range(len(resampled)):
        epoch = resampled[i].copy()
        new_resampled.append(epoch.pick(standard_1020))
    n_chans = len(standard_1020)
elif electrode == '9':
    for i in range(len(resampled)):
        epoch = resampled[i].copy()
        new_resampled.append(epoch.pick(around_c3))
    n_chans = len(around_c3)
elif electrode == 'selected':
    for i in range(len(resampled)):
        epoch = resampled[i].copy()
        new_resampled.append(epoch.pick(selected))
    n_chans = len(selected)
else:
    new_resampled = resampled

for j in range(len(persons)):
    X_data = resampled[j].get_data()
    y_data = list_labels[j]

    for bs in batch_sizes:
        for lr in lrs:
            kf_scores = []

            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for train_ids, test_ids in kf.split(X_data, y_data):
                X_train = X_data[train_ids]
                y_train = y_data[train_ids]
                X_test = X_data[test_ids]
                y_test = y_data[test_ids]
                X_train_normed = (X_train - np.mean(X_train, axis=2, keepdims=True)) / np.std(X_train, axis=2, keepdims=True)
                X_test_normed = (X_test - np.mean(X_test, axis=2, keepdims=True)) / np.std(X_test, axis=2, keepdims=True)

        
                model = EEGNetv4(
                    n_chans=n_chans,
                    n_outputs=2,
                    n_times=X_train.shape[2],
                    kernel_length=64
                )

                net = EEGClassifier(
                    model,
                    cropped=False,   
                    train_split=None,
                    criterion=torch.nn.CrossEntropyLoss,
                    optimizer=torch.optim.Adam,
                    optimizer__lr=lr,
                    batch_size=bs,
                    callbacks=[
                        "accuracy",
                        # ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
                    ],
                    device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
                )

                net = net.fit(X_train_normed, y=y_train, epochs=n_epochs)
                score = net.score(X_test_normed, y=y_test)
                kf_scores.append(score)
            if f'lr={lr:.4f}, bs={bs}' in scores:
                scores[f'lr={lr:.4f}, bs={bs}'][persons[j]] = np.mean(kf_scores)
            else:
                scores[f'lr={lr:.4f}, bs={bs}'] = {persons[j]: np.mean(kf_scores)}
        with open('results_per_person/eegnet.json', 'w') as f:
            json.dump(scores, f)

for k, v in scores.items():
    results = []
    for _, score in v.items():
        results.append(score)
    print(k, np.mean(results), np.std(results))