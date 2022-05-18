from torch.utils.data import DataLoader, TensorDataset
from natsort import natsorted
import glob
from torchvision import datasets
import torch
import numpy as np
import os
import re



def load_data(BATCH_SIZE_TRAIN, BATCH_SIZE_TEST):
    data_list = np.loadtxt("./data/train/train.csv", skiprows=1, encoding="utf-8_sig", delimiter=',')

    race_idx = np.where(data_list[:, 0] == 0)
    x, y  = np.array([]), np.array([])
    for idx in range(0, len(race_idx[0])):
        st = idx
        en = idx + 1
        if idx == len(race_idx[0])-1:
            en = -1

        race = data_list[race_idx[0][st]:race_idx[0][en], 1:-1]
        rank = data_list[race_idx[0][st]:race_idx[0][en], -1]

        if idx == len(race_idx[0])-1:
            race = data_list[race_idx[0][st]:, 1:-1]
            rank = data_list[race_idx[0][st]:, -1]

        for i in range(0, 18):
            if len(race) < 18:
                add = np.zeros((1, 9))
                race = np.vstack((race, add))
                rank = np.append(rank, [0])
            else:
                break

        #print(race)
        if idx == 0:
            x = [race]
            y = [rank]
        else:
            x = np.vstack((x, [race]))
            y = np.vstack((y, [rank]))
        print(f"load train.csv {idx}/{len(race_idx[0])}")

    print("train data shape:", x.shape)
    print("train label data shape:", y.shape)
    x_train = x
    y_train = y

    #data_list = np.loadtxt("./data/train/test.csv", skiprows=1, encoding="shift-jis", delimiter=',')
    data_list = np.loadtxt("./data/train/test.csv", skiprows=1, encoding="utf-8_sig", delimiter=',')

    race_idx = np.where(data_list[:, 0] == 0)
    x, y  = np.array([]), np.array([])
    for idx in range(0, len(race_idx[0])):
        st = idx
        en = idx + 1
        if idx == len(race_idx[0])-1:
            en = -1

        race = data_list[race_idx[0][st]:race_idx[0][en], 1:-1]
        rank = data_list[race_idx[0][st]:race_idx[0][en], -1]

        if idx == len(race_idx[0])-1:
            race = data_list[race_idx[0][st]:, 1:-1]
            rank = data_list[race_idx[0][st]:, -1]

        for i in range(0, 18):
            if len(race) < 18:
                add = np.zeros((1, 9))
                race = np.vstack((race, add))
                rank = np.append(rank, [0])
            else:
                break

        #print(race)
        if idx == 0:
            x = [race]
            y = [rank]
        else:
            x = np.vstack((x, [race]))
            y = np.vstack((y, [rank]))
        print(f"load test.csv {idx}/{len(race_idx[0])}")

    print("test data shape:", x.shape)
    print("test label data shape:", y.shape)
    x_test = x
    y_test = y

    # データの準備
    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE_TRAIN, shuffle=True, drop_last=True)
    test = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE_TEST, shuffle=True, drop_last=True)

    return train_loader, test_loader



def load_cfg(cfg):
    input_dim, output_dim, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST, EPOCHS, LR, beta1, beta2, step_size, gamma, NUM_ResBlock = "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"

    with open(cfg, "r", encoding='utf-8') as f:
        text = f.read()
    lines = text.splitlines()

    for i in range(0, len(lines)):
        text = lines[i].split('#')[0]
        text = text.replace(' ', '')
        if "input_dim" in text:
            input_dim = re.split('=(.*)', re.split('input_dim(.*)', text)[1])[1]
        elif "output_dim" in text:
            output_dim = re.split('=(.*)', re.split('output_dim(.*)', text)[1])[1]
        elif "BATCH_SIZE_TRAIN" in text:
            BATCH_SIZE_TRAIN = re.split('=(.*)', re.split('BATCH_SIZE_TRAIN(.*)', text)[1])[1]
        elif "BATCH_SIZE_TEST" in text:
            BATCH_SIZE_TEST = re.split('=(.*)', re.split('BATCH_SIZE_TEST(.*)', text)[1])[1]
        elif "EPOCHS" in text:
            EPOCHS = re.split('=(.*)', re.split('EPOCHS(.*)', text)[1])[1]
        elif "LR" in text:
            LR = re.split('=(.*)', re.split('LR(.*)', text)[1])[1]
        elif "beta1" in text:
            beta1 = re.split('=(.*)', re.split('beta1(.*)', text)[1])[1]
        elif "beta2" in text:
            beta2 = re.split('=(.*)', re.split('beta2(.*)', text)[1])[1]
        elif "step_size" in text:
            step_size = re.split('=(.*)', re.split('step_size(.*)', text)[1])[1]
        elif "gamma" in text:
            gamma = re.split('=(.*)', re.split('gamma(.*)', text)[1])[1]
        elif "NUM_ResBlock" in text:
            NUM_ResBlock = re.split('=(.*)', re.split('NUM_ResBlock(.*)', text)[1])[1]
        else:
            pass

    betas = (float(beta1), float(beta2))

    return int(input_dim), int(output_dim), int(BATCH_SIZE_TRAIN), int(BATCH_SIZE_TEST), int(EPOCHS), float(LR), betas, int(step_size), float(gamma), int(NUM_ResBlock)



def load_mdl(mdl):
    with open(mdl, "r", encoding='utf-8') as f:
        text = f.read()
    lines = text.splitlines()

    for i in range(0, len(lines)):
        text = lines[i].split('#')[0]
        text = text.replace(' ', '')
        if "model" in text:
            path = re.split('="(.*)"', re.split('model(.*)', text)[1])[1]
        else:
            pass

    return path

def load_predict(path):
    data_list = np.loadtxt(path, skiprows=1, encoding="utf-8_sig", delimiter=',')

    x = data_list[:, 1:]
    for i in range(0, 18):
        if len(x) < 18:
            add = np.zeros((1, 9))
            x = np.vstack((x, add))
        else:
            break
    x = torch.from_numpy(x).float()
    print("predict data shape:", x.shape)

    return x