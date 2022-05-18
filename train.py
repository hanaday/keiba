from model import K2Net #, TResNet, 
import model as mod
import load as L
import accuracy as ac
import torch
import torch.nn.init as init
from torch import nn, optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from torchsummary import summary


class TRAIN:
    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            print('Error in train.py: device is cpu', file=sys.stderr)
            sys.exit(1)

        #self.SAVE_MODEL_PATH = f"./model/TResNet/"  # モデルの保存先
        self.SAVE_MODEL_PATH = f"./model/K2Net/"  # モデルの保存先

        # 保存先のディレクトリを作成する
        os.makedirs(self.SAVE_MODEL_PATH, exist_ok=True)

        self.input_dim, self.output_dim, self.BATCH_SIZE_TRAIN, self.BATCH_SIZE_TEST, self.EPOCHS, self.LR, self.betas, self.step_size, self.gamma, self.NUM_ResBlock = L.load_cfg(cfg)
        self.train_loader, self.test_loader = L.load_data(self.BATCH_SIZE_TRAIN, self.BATCH_SIZE_TEST)
        print("load data")
        
        #summary(TResNet(self.input_dim, self.output_dim, self.NUM_ResBlock).to(self.device), (1, 18, self.input_dim), 32)
        summary(K2Net(self.input_dim, self.output_dim, self.NUM_ResBlock).to(self.device), (1, 18, self.input_dim), 32)


        #self.model = TResNet(self.input_dim, self.output_dim, self.NUM_ResBlock).to(self.device)
        self.model = K2Net(self.input_dim, self.output_dim, self.NUM_ResBlock).to(self.device)
        self.model.apply(mod.weights_init)

        print("load model")

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.LR, betas=self.betas, eps=1e-08, weight_decay=0, amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)


    def train(self):
        with open("./log/train_log.txt", "w", encoding='utf-8') as f:
            f.write("")
        with open("./log/train_log_max.txt", "w", encoding='utf-8') as f:
            f.write("")
        with open("./log/train_log_hyb.txt", "w", encoding='utf-8') as f:
            f.write("")
        loss_history = []
        accuracy_list, accuracy_1_list, accuracy_3_list, accuracy_5_list = [], [], [], []
        acc_list, acc_1_list, acc_3_list, acc_5_list = [], [], [], []

        for epoch in range(self.EPOCHS):
            total_loss = 0

            # model train
            for idx, (x, y) in enumerate(self.train_loader):
                #print(x.shape)
                #print(y.shape) 
                x = x.to('cpu').detach().numpy().copy()
                x[:, :, 8] = x[:, :, 8] * np.random.randint(0.5, 5, size=(self.BATCH_SIZE_TRAIN, 18))
                x[:, :, 8] = np.where(x[:, :, 8] < 1, 1, x[:, :, 8])
                x = torch.from_numpy(x.astype(np.float32)).clone()

                x = x.view(self.BATCH_SIZE_TRAIN, 1, 18, self.input_dim).to(self.device)
                y = y.view(self.BATCH_SIZE_TRAIN, 18).to(self.device)
                #print(x.shape)
                #print(y.shape) 

                for j in range(self.BATCH_SIZE_TRAIN):
                    torch.seed()
                    index = torch.randperm(18)
                    x[j] = x[j, 0, index]
                    y[j] = y[j, index]

                #print(x.shape)
                #print(y.shape)


                # set values
                #x = Variable(x).to(self.device)
                # noise for discriminator
                #noise1 = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * (self.EPOCHS - epoch) / self.EPOCHS), requires_grad=False).to(self.device)


                # 学習ステップ
                self.model.train()
                self.optimizer.zero_grad()

                outputs = self.model(x)
                outputs = outputs.view(self.BATCH_SIZE_TRAIN, self.output_dim, 18)

                #outputs = torch.transpose(outputs, 1, 2)
                #print(outputs.shape)

                loss = self.criterion(outputs, y.type(torch.long))
                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            loss_history.append(total_loss)
            if (epoch +1) % 10 != 0:
                print(f"epoch:{epoch+1}, trainLoss:{total_loss}")

            self.scheduler.step()

            # save model
            if (epoch + 1) % 10 == 0:
                torch.save(self.model.state_dict(),f'{self.SAVE_MODEL_PATH}/{epoch + 1}.pkl')

            # model eval
            if (epoch +1) % 10 == 0:
                test_loss = 0
                correct, correct_1, correct_2, correct_3, correct_4, correct_5 = 0, 0, 0, 0, 0, 0
                cor, cor_1, cor_2, cor_3, cor_4, cor_5 = 0, 0, 0, 0, 0, 0
                total = 0
                x_list, y_list = [], []
                for x, y in self.test_loader:
                    #x = x.view(-1, 9).to(device)
                    #y = y.view(-1).to(device)
                    x = x.view(self.BATCH_SIZE_TEST, 1, 18, self.input_dim).to(self.device)
                    y = y.view(self.BATCH_SIZE_TEST, 18).to(self.device)
                    #print(x.shape)
                    #print(y.shape)

                    for j in range(self.BATCH_SIZE_TEST):
                        index = torch.randperm(18)
                        x[j] = x[j, 0, index]
                        y[j] = y[j, index]

                    self.model.eval()
                    outputs = self.model(x)
            
                    outputs = outputs.view(self.BATCH_SIZE_TEST, self.output_dim, 18)

                    loss = self.criterion(outputs, y.type(torch.long))

                    outputs = self.model(x, CulLoss=False)
                    outputs = outputs.view(self.BATCH_SIZE_TEST, self.output_dim, 18)
            
                    outputs = torch.transpose(outputs, 1, 2)
                    x_list.append(outputs.tolist())
                    y_list.append(y.tolist())


                    total += y.size(0)
                    test_loss += loss
        
                x_list = np.array(x_list)
                y_list = np.array(y_list)
                #print(x_list, "\n", y_list)



                acc, acc_1, acc_2, acc_3, acc_4, acc_5 = ac.cul_acc(self.output_dim, x_list, y_list, total, epoch, total_loss, test_loss)
                acc_list.append(acc)
                acc_1_list.append(acc_1)
                acc_3_list.append((acc_1 + acc_2 +acc_3)/3)
                acc_5_list.append((acc_1 + acc_2 + acc_3 + acc_4 + acc_5)/5)
                acc, acc_1, acc_2, acc_3, acc_4, acc_5 = ac.cul_acc_max(self.output_dim, x_list, y_list, total, epoch, total_loss, test_loss)
                acc, acc_1, acc_2, acc_3, acc_4, acc_5 = ac.cul_acc_hyb(self.output_dim, x_list, y_list, total, epoch, total_loss, test_loss)

                """
                for id_1, x_lis in enumerate(x_list):
                    for id_2, x_li in enumerate(x_lis):
                        for i in range(1, self.output_dim):
                            max_id = np.argmax(x_li[:, i])
                            if y_list[id_1, id_2, max_id] == 1 or y_list[id_1, id_2, max_id] == 2 or y_list[id_1, id_2, max_id] == 3 and i < 4:
                                correct += 1
                            if y_list[id_1, id_2, max_id] == i:
                                if i == 1:
                                    correct_1 += 1
                                elif i == 2:
                                    correct_2 += 1
                                elif i == 3:
                                    correct_3 += 1
                                elif i == 4:
                                    correct_4 += 1
                                elif i == 5:
                                    correct_5 += 1
                                        


                uma_1 = []
                for id_1, x_lis in enumerate(x_list):
                    uma_2 = []
                    for id_2, x_li in enumerate(x_lis):
                        uma_3 = []
                        for j in range(18):
                            uma = 0
                            for i in range(1, self.output_dim):
                                uma += (12 - 2*i)*x_li[j, i]
                            uma_3.append(uma)

                        uma_3_np = np.array(uma_3)
                        for k in range(1, self.output_dim):
                            max = sorted(uma_3_np.ravel())[-k]
                            index = np.where(uma_3 == max)
                            if y_list[id_1, id_2, index] == 1 or y_list[id_1, id_2, index] == 2 or y_list[id_1, id_2, index] == 3 and k < 4:
                                cor += 1
                            if y_list[id_1, id_2, index] == k:
                                cor += 1
                                if k == 1:
                                    cor_1 += 1
                                elif k == 2:
                                    cor_2 += 1
                                elif k == 3:
                                    cor_3 += 1
                                elif k == 4:
                                    cor_4 += 1
                                elif k == 5:
                                    cor_5 += 1

                        uma_2.append(uma_3)
                    uma_1.append(uma_2)

                accuracy = int(correct)/total*100/3
                accuracy_1 = int(correct_1)/total*100
                accuracy_2 = int(correct_2)/total*100
                accuracy_3 = int(correct_3)/total*100
                accuracy_4 = int(correct_4)/total*100
                accuracy_5 = int(correct_5)/total*100
                acc = int(cor)/total*100/5
                acc_1 = int(cor_1)/total*100
                acc_2 = int(cor_2)/total*100
                acc_3 = int(cor_3)/total*100
                acc_4 = int(cor_4)/total*100
                acc_5 = int(cor_5)/total*100

                accuracy_list.append(accuracy)
                accuracy_1_list.append(accuracy_1)
                accuracy_3_list.append((accuracy_1 + accuracy_2 +accuracy_3)/3)
                accuracy_5_list.append((accuracy_1 + accuracy_2 + accuracy_3 + accuracy_4 + accuracy_5)/5)
                acc_list.append(acc)
                acc_1_list.append(acc_1)
                acc_3_list.append((acc_1 + acc_2 +acc_3)/3)
                acc_5_list.append((acc_1 + acc_2 + acc_3 + acc_4 + acc_5)/5)
                with open("./log/train_log.txt", "a") as f:
                    print(f"epoch:{epoch+1}, trainLoss:{total_loss}, testLoss:{test_loss} 正解率 1~3:{accuracy:.3}, 1:{accuracy_1:.3f}, 2:{accuracy_2:.3f}, 3:{accuracy_3:.3f}, 4:{accuracy_4:.3f}, 5:{accuracy_5:.3f}", file=f)
                print(f"epoch:{epoch+1}, trainLoss:{total_loss}, testLoss:{test_loss}")
                print(f'正解率 1:{accuracy_1:.3f}, 2:{accuracy_2:.3f}, 3:{accuracy_3:.3f}, 4:{accuracy_4:.3f}, 5:{accuracy_5:.3f}')
                print('3着以上の正解率', accuracy)
                print(f'正解率 1:{acc_1:.3f}, 2:{acc_2:.3f}, 3:{acc_3:.3f}, 4:{acc_4:.3f}, 5:{acc_5:.3f}')
                print('3着以上の正解率', acc)
                """

        print("train finished")
        #print('正解率', accuracy_list)
        #print('3着以上の正解率', accuracy_3_list)
        #print('1着の正解率', accuracy_1_list)

        plt.plot(loss_history)
        plt.show()

        """
        plt.plot(accuracy_5_list)
        plt.plot(accuracy_3_list)
        plt.plot(accuracy_1_list)
        plt.plot(accuracy_list)
        #plt.legend()
        plt.show()
        """

        plt.plot(acc_5_list)
        plt.plot(acc_3_list)
        plt.plot(acc_1_list)
        plt.plot(acc_list)
        #plt.legend()
        plt.show()
