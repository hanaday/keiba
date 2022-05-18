from model import TResNet, K2Net
import load as L
import torch
import numpy as np
import sys
import csv
import os
import re
import copy


class PREDICT:
    def __init__(self, cfg, data, mdl):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print('Error in detect.py: device is cpu', file=sys.stderr)
            sys.exit(1)

        with open(data, "r", encoding='utf-8') as f:
            text = f.read()
        lines = text.splitlines()
        for i in range(0, len(lines)):
            text = lines[i].split('#')[0]
            text = text.replace(' ', '')
            if "race" in text:
                self.path = re.split('="(.*)"', re.split('race(.*)', text)[1])[1]
            else:
                pass

        self.input_dim, self.output_dim, self.BATCH_SIZE_TRAIN, self.BATCH_SIZE_TEST, self.EPOCHS, self.LR, self.betas, self.step_size, self.gamma, self.NUM_ResBlock = L.load_cfg(cfg)
        model_path = L.load_mdl(mdl)

        #self.model = TResNet(self.input_dim, self.output_dim, self.NUM_ResBlock).to(self.device)
        self.model = K2Net(self.input_dim, self.output_dim, self.NUM_ResBlock).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        print("load model")


    def predict(self):
        x_ori = L.load_predict(self.path)
        print(x_ori.shape)

        for num in range(100):
            x = x_ori
            index = torch.randperm(18)
            x = x[index]
            #print(x.shape)
            x_list = x[:, 2].tolist()

            x = x.view(1, 1, 18, 9).to(self.device)
            outputs = self.model(x, CulLoss=False)
            outputs = outputs.view(6, 18)
            outputs = torch.transpose(outputs, 0, 1)
            pred = outputs.tolist()

            #x_list = np.array(x_list)
            pred = np.array(pred)

            #x_list_2 = [r[2] for r in x_list]
            x_list_2 = np.array(x_list).reshape((18, 1))
            ans = np.hstack((x_list_2, pred))
            for i in range(1, 19):
                idx = np.where(ans[:, 0] == i)
                if len(idx[0]) != 0:
                    if i == 1:
                        pred = ans[idx, :].reshape((1, 7)) #.reshape((1, 13))
                    else:
                        pred = np.vstack((pred, ans[idx, :].reshape((1, 7)))) # .reshape((1, 13))))
            idx = np.where(ans[:, 0] == 0)
            for i in idx[0]:
                pred = np.vstack((pred, ans[i, :]))
            

            print(num, "/ 100")
            if num == 0:
                all_pred = pred
                print(all_pred.shape)
            else:
                all_pred[:, 1:] = (all_pred[:, 1:] + pred[:, 1:]) / 2
                print(all_pred.shape)

        pred = all_pred[:, 1:]

        rank = np.zeros((18, 3)) # np.zeros((18, 6))
        """
        for i in range(1, self.output_dim):
            max_id = np.argmax(pred[:, i])
            for j in range(5):
                if rank[max_id, j] != 0:
                    pass
                else:
                    rank[max_id, j] = i
                    break
        """

        # 強さ係数
        sum_list = []
        for i in range(18):
            sum = 0
            for j in range(1, self.output_dim):
                alpha = 12 - 2*j  # (6 - j)
                if alpha < 0:
                    alpha = 0
                sum += alpha * pred[i, j]
            sum_list.append(sum)
        sum_list = np.array(sum_list)
        for i in range(1, 6):
            max = sorted(sum_list.ravel())[-i]
            index = np.where(sum_list == max)
            rank[index, 0] = i # rank[index, 5] = i

        # 大きさ順
        rank_id = []
        for i in range(1, self.output_dim):
            box = copy.deepcopy(pred[:, i])
            #print(box.shape)
            for r_id in rank_id:
                box[r_id] = 0
            max_id = np.argmax(box)
            rank_id.append(max_id)
            rank[max_id, 1] = i
        #print(pred)

        # ハイブリッド
        rank_id = []
        for i in range(1, self.output_dim):
            box = copy.deepcopy(pred[:, i])
            Z = 0.5
            for j in range(1, i):
                if j == 1:
                    box = Z*box
                Z = Z/2 if i-j >= 2 else Z
                box = box + Z*pred[:, j] 
            #print(box.shape)
            for r_id in rank_id:
                box[r_id] = 0
            max_id = np.argmax(box)
            rank_id.append(max_id)
            rank[max_id, 2] = i

        pred = pred * 100
        rank = np.hstack((pred, rank))

        #x_list_2 = [r[2] for r in x_list]
        x_list_2 = np.array(all_pred[:, 0]).reshape((18, 1))
        rank = np.hstack((x_list_2, rank))

        
        for i in range(1, 19):
            idx = np.where(rank[:, 0] == i)
            if len(idx[0]) != 0:
                if i == 1:
                    pred = rank[idx, :].reshape((1, 10)) #.reshape((1, 13))
                else:
                    pred = np.vstack((pred, rank[idx, :].reshape((1, 10)))) # .reshape((1, 13))))
        idx = np.where(rank[:, 0] == 0)
        for i in idx[0]:
            pred = np.vstack((pred, rank[i, :]))

        print(pred)

        basename = os.path.basename(self.path)
        with open("./data/predict/predict_%s" %basename, "w", encoding='utf-8_sig', newline="" ) as f:
            writer = csv.writer(f)
            writer.writerows(pred)

        print("predict finished")

