import numpy as np

def cul_acc(output_dim, x_list, y_list, total, epoch, total_loss, test_loss):
    cor, cor_1, cor_2, cor_3, cor_4, cor_5 = 0, 0, 0, 0, 0, 0
    hukusho, baren, wide, batan, sanrenhuku, sanrentan = 0, 0, 0, 0, 0, 0
    
    uma_1 = []  # uma_1 = (12, 8, 18)
    for id_1, x_lis in enumerate(x_list):
        uma_2 = []  # uma_2 = (8, 18)
        for id_2, x_li in enumerate(x_lis):
            uma_3 = []  # uma_3 = (18, )
            for j in range(18):
                uma = 0
                for i in range(1, output_dim):
                    alpha = 12 - 2*i  # (6 - i)
                    if alpha < 0:
                        alpha = 0
                    uma += alpha*x_li[j, i]
                uma_3.append(uma)

            uma_3_np = np.array(uma_3)
            batan_flag = 0
            baren_flag = False
            wide_flag = 0
            for k in range(1, output_dim):
                max = sorted(uma_3_np.ravel())[-k]
                index = np.where(uma_3 == max)
                if len(index) >= 2:
                    index = index[0]
                # y_list = (12, 8, 18)
                if (y_list[id_1, id_2, index] == 1 or y_list[id_1, id_2, index] == 2 or y_list[id_1, id_2, index] == 3) and k < 4:
                    hukusho += 1  # 後で3で割る必要あり
                    wide_flag += 1
                    if k == 1:
                        baren_flag = True
                    elif k == 2 and baren_flag == True:
                        baren += 1
                if y_list[id_1, id_2, index] == k:
                    cor += 1
                    if k == 1:
                        cor_1 += 1
                        batan_flag += 1
                    elif k == 2:
                        cor_2 += 1
                        if batan_flag == 1:
                            batan += 1
                        batan_flag += 1
                    elif k == 3:
                        cor_3 += 1
                        if batan_flag == 2:
                            sanrentan += 1
                    elif k == 4:
                        cor_4 += 1
                    elif k == 5:
                        cor_5 += 1
            wide += 3 if wide_flag == 3 else (wide_flag - 1) if (wide_flag - 1) > 0 else 0  # 後で3で割る必要あり
            sanrenhuku += 1 if wide_flag == 3 else 0

            uma_2.append(uma_3)
        uma_1.append(uma_2)

    acc = int(cor)/total*100/5
    acc_1 = int(cor_1)/total*100
    acc_2 = int(cor_2)/total*100
    acc_3 = int(cor_3)/total*100
    acc_4 = int(cor_4)/total*100
    acc_5 = int(cor_5)/total*100
    acc_hukusho  = int(hukusho)/total*100/3
    acc_baren  = int(baren)/total*100
    acc_wide  = int(wide)/total*100/3
    acc_batan  = int(batan)/total*100
    acc_sanrenhuku  = int(sanrenhuku)/total*100
    acc_sanrentan  = int(sanrentan)/total*100

    #acc_list.append(acc)
    #acc_1_list.append(acc_1)
    #acc_3_list.append((acc_1 + acc_2 +acc_3)/3)
    #acc_5_list.append((acc_1 + acc_2 + acc_3 + acc_4 + acc_5)/5)
    with open("./log/train_log.txt", "a") as f:
        print(f"epoch:{epoch+1}, trainLoss:{total_loss}, testLoss:{test_loss} \
                正解率 1:{acc_1:.3f}, 2:{acc_2:.3f}, 3:{acc_3:.3f}, 4:{acc_4:.3f}, 5:{acc_5:.3f} 1着から5着:{acc:.3f}, {(acc_1+acc_2+acc_3+acc_4+acc_5)/5:.3f} \
                単勝:{acc_1:.3f}, 複勝:{acc_hukusho:.3f}, 馬連:{acc_baren:.3f}, ワイド:{acc_wide:.3f}, 馬単:{acc_batan:.3f}, 3連複:{acc_sanrenhuku:.3f}, 3連単:{acc_sanrentan:.3f}", file=f)
    
    print(f"epoch:{epoch+1}, trainLoss:{total_loss}, testLoss:{test_loss}")
    print(f'正解率 1:{acc_1:.3f}, 2:{acc_2:.3f}, 3:{acc_3:.3f}, 4:{acc_4:.3f}, 5:{acc_5:.3f}')
    print(f'1着から5着の正解率:{acc:.3f}, {(acc_1+acc_2+acc_3+acc_4+acc_5)/5:.3f}')
    print(f'単勝:{acc_1:.3f}, 複勝:{acc_hukusho:.3f}, 馬連:{acc_baren:.3f}, ワイド:{acc_wide:.3f}, 馬単:{acc_batan:.3f}, 3連複:{acc_sanrenhuku:.3f}, 3連単:{acc_sanrentan:.3f}')

    return acc, acc_1, acc_2, acc_3, acc_4, acc_5



def cul_acc_max(output_dim, x_list, y_list, total, epoch, total_loss, test_loss):
    cor, cor_1, cor_2, cor_3, cor_4, cor_5 = 0, 0, 0, 0, 0, 0
    hukusho, baren, wide, batan, sanrenhuku, sanrentan = 0, 0, 0, 0, 0, 0
    
    for id_1, x_lis in enumerate(x_list):
        for id_2, x_li in enumerate(x_lis):
            rank_id, uma_3 = [], []
            for i in range(1, output_dim):
                box = x_li[:, i]
                #print(box.shape)
                for r_id in rank_id:
                    box[r_id] = 0
                max_id = np.argmax(box)
                rank_id.append(max_id)
            #uma_3.append(rank_id)
                

            #uma_3_np = np.array(uma_3)
            batan_flag = 0
            baren_flag = False
            wide_flag = 0
            for k in range(1, output_dim):
                index = rank_id[k-1]
                # y_list = (12, 8, 18)
                if (y_list[id_1, id_2, index] == 1 or y_list[id_1, id_2, index] == 2 or y_list[id_1, id_2, index] == 3) and k < 4:
                    hukusho += 1  # 後で3で割る必要あり
                    wide_flag += 1
                    if k == 1:
                        baren_flag = True
                    elif k == 2 and baren_flag == True:
                        baren += 1
                if y_list[id_1, id_2, index] == k:
                    cor += 1
                    if k == 1:
                        cor_1 += 1
                        batan_flag += 1
                    elif k == 2:
                        cor_2 += 1
                        if batan_flag == 1:
                            batan += 1
                        batan_flag += 1
                    elif k == 3:
                        cor_3 += 1
                        if batan_flag == 2:
                            sanrentan += 1
                    elif k == 4:
                        cor_4 += 1
                    elif k == 5:
                        cor_5 += 1
            wide += 3 if wide_flag == 3 else (wide_flag - 1) if (wide_flag - 1) > 0 else 0  # 後で3で割る必要あり
            sanrenhuku += 1 if wide_flag == 3 else 0

    acc = int(cor)/total*100/5
    acc_1 = int(cor_1)/total*100
    acc_2 = int(cor_2)/total*100
    acc_3 = int(cor_3)/total*100
    acc_4 = int(cor_4)/total*100
    acc_5 = int(cor_5)/total*100
    acc_hukusho  = int(hukusho)/total*100/3
    acc_baren  = int(baren)/total*100
    acc_wide  = int(wide)/total*100/3
    acc_batan  = int(batan)/total*100
    acc_sanrenhuku  = int(sanrenhuku)/total*100
    acc_sanrentan  = int(sanrentan)/total*100

    #acc_list.append(acc)
    #acc_1_list.append(acc_1)
    #acc_3_list.append((acc_1 + acc_2 +acc_3)/3)
    #acc_5_list.append((acc_1 + acc_2 + acc_3 + acc_4 + acc_5)/5)
    with open("./log/train_log_max.txt", "a") as f:
        print(f"epoch:{epoch+1}, trainLoss:{total_loss}, testLoss:{test_loss} \
                正解率 1:{acc_1:.3f}, 2:{acc_2:.3f}, 3:{acc_3:.3f}, 4:{acc_4:.3f}, 5:{acc_5:.3f} 1着から5着:{acc:.3f}, {(acc_1+acc_2+acc_3+acc_4+acc_5)/5:.3f} \
                単勝:{acc_1:.3f}, 複勝:{acc_hukusho:.3f}, 馬連:{acc_baren:.3f}, ワイド:{acc_wide:.3f}, 馬単:{acc_batan:.3f}, 3連複:{acc_sanrenhuku:.3f}, 3連単:{acc_sanrentan:.3f}", file=f)
    
    print(f"epoch:{epoch+1}, trainLoss:{total_loss}, testLoss:{test_loss}")
    print(f'正解率 1:{acc_1:.3f}, 2:{acc_2:.3f}, 3:{acc_3:.3f}, 4:{acc_4:.3f}, 5:{acc_5:.3f}')
    print(f'1着から5着の正解率:{acc:.3f}, {(acc_1+acc_2+acc_3+acc_4+acc_5)/5:.3f}')
    print(f'単勝:{acc_1:.3f}, 複勝:{acc_hukusho:.3f}, 馬連:{acc_baren:.3f}, ワイド:{acc_wide:.3f}, 馬単:{acc_batan:.3f}, 3連複:{acc_sanrenhuku:.3f}, 3連単:{acc_sanrentan:.3f}')

    return acc, acc_1, acc_2, acc_3, acc_4, acc_5




def cul_acc_hyb(output_dim, x_list, y_list, total, epoch, total_loss, test_loss):
    cor, cor_1, cor_2, cor_3, cor_4, cor_5 = 0, 0, 0, 0, 0, 0
    hukusho, baren, wide, batan, sanrenhuku, sanrentan = 0, 0, 0, 0, 0, 0
    
    for id_1, x_lis in enumerate(x_list):
        for id_2, x_li in enumerate(x_lis):
            rank_id, uma_3 = [], []
            for i in range(1, output_dim):
                box = x_li[:, i]
                Z = 0.5
                for j in range(1, i):
                    if j == 1:
                        box = Z*box
                    Z = Z/2 if i-j >= 2 else Z
                    box = box + Z*x_li[:, j] 
                #print(box.shape)
                for r_id in rank_id:
                    box[r_id] = 0
                max_id = np.argmax(box)
                rank_id.append(max_id)
            #uma_3.append(rank_id)

            #uma_3_np = np.array(uma_3)
            batan_flag = 0
            baren_flag = False
            wide_flag = 0
            for k in range(1, output_dim):
                index = rank_id[k-1]
                # y_list = (12, 8, 18)
                if (y_list[id_1, id_2, index] == 1 or y_list[id_1, id_2, index] == 2 or y_list[id_1, id_2, index] == 3) and k < 4:
                    hukusho += 1  # 後で3で割る必要あり
                    wide_flag += 1
                    if k == 1:
                        baren_flag = True
                    elif k == 2 and baren_flag == True:
                        baren += 1
                if y_list[id_1, id_2, index] == k:
                    cor += 1
                    if k == 1:
                        cor_1 += 1
                        batan_flag += 1
                    elif k == 2:
                        cor_2 += 1
                        if batan_flag == 1:
                            batan += 1
                        batan_flag += 1
                    elif k == 3:
                        cor_3 += 1
                        if batan_flag == 2:
                            sanrentan += 1
                    elif k == 4:
                        cor_4 += 1
                    elif k == 5:
                        cor_5 += 1
            wide += 3 if wide_flag == 3 else (wide_flag - 1) if (wide_flag - 1) > 0 else 0  # 後で3で割る必要あり
            sanrenhuku += 1 if wide_flag == 3 else 0

    acc = int(cor)/total*100/5
    acc_1 = int(cor_1)/total*100
    acc_2 = int(cor_2)/total*100
    acc_3 = int(cor_3)/total*100
    acc_4 = int(cor_4)/total*100
    acc_5 = int(cor_5)/total*100
    acc_hukusho  = int(hukusho)/total*100/3
    acc_baren  = int(baren)/total*100
    acc_wide  = int(wide)/total*100/3
    acc_batan  = int(batan)/total*100
    acc_sanrenhuku  = int(sanrenhuku)/total*100
    acc_sanrentan  = int(sanrentan)/total*100

    #acc_list.append(acc)
    #acc_1_list.append(acc_1)
    #acc_3_list.append((acc_1 + acc_2 +acc_3)/3)
    #acc_5_list.append((acc_1 + acc_2 + acc_3 + acc_4 + acc_5)/5)
    with open("./log/train_log_hyb.txt", "a") as f:
        print(f"epoch:{epoch+1}, trainLoss:{total_loss}, testLoss:{test_loss} \
                正解率 1:{acc_1:.3f}, 2:{acc_2:.3f}, 3:{acc_3:.3f}, 4:{acc_4:.3f}, 5:{acc_5:.3f} 1着から5着:{acc:.3f}, {(acc_1+acc_2+acc_3+acc_4+acc_5)/5:.3f} \
                単勝:{acc_1:.3f}, 複勝:{acc_hukusho:.3f}, 馬連:{acc_baren:.3f}, ワイド:{acc_wide:.3f}, 馬単:{acc_batan:.3f}, 3連複:{acc_sanrenhuku:.3f}, 3連単:{acc_sanrentan:.3f}", file=f)
    
    print(f"epoch:{epoch+1}, trainLoss:{total_loss}, testLoss:{test_loss}")
    print(f'正解率 1:{acc_1:.3f}, 2:{acc_2:.3f}, 3:{acc_3:.3f}, 4:{acc_4:.3f}, 5:{acc_5:.3f}')
    print(f'1着から5着の正解率:{acc:.3f}, {(acc_1+acc_2+acc_3+acc_4+acc_5)/5:.3f}')
    print(f'単勝:{acc_1:.3f}, 複勝:{acc_hukusho:.3f}, 馬連:{acc_baren:.3f}, ワイド:{acc_wide:.3f}, 馬単:{acc_batan:.3f}, 3連複:{acc_sanrenhuku:.3f}, 3連単:{acc_sanrentan:.3f}')

    return acc, acc_1, acc_2, acc_3, acc_4, acc_5