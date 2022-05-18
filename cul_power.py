import power
import csv
import os
import numpy as np

with open("./data/total/all/total.csv", "r", encoding="utf-8_sig", errors="", newline="" ) as f:
#with open("./data/total/file2022.csv", "r", encoding="utf-8_sig", errors="", newline="" ) as f:
    data = csv.reader(f)

    Uma, Rank, ID, Cor, Dis, Bab = [], [], [], [], [], []
    for idx, dat in enumerate(data):
        #print(dat)
        if idx == 0:
            id_id = dat.index('レースID')
            id_uma = dat.index('馬名')
            id_ran = dat.index('順位')
            id_cor = dat.index('コース')
            id_dis = dat.index('距離')
            id_bab = dat.index('馬場')
            #print(row_id_uma, row_id_jok)
        else:
            ID.append(dat[id_id])
            Uma.append(dat[id_uma])
            Rank.append(dat[id_ran])
            Cor.append(dat[id_cor])
            Dis.append(dat[id_dis])
            Bab.append(dat[id_bab])


Uma_name = []
Uma_csv = []
if os.path.exists("./data/power/horse.csv"):
    with open("./data/power/horse.csv", "r", encoding="utf-8_sig", errors="", newline="" ) as f:
        data = csv.reader(f)
        for idx, dat in enumerate(data):
            Uma_name.append(dat[0])
            Uma_csv.append([dat[0], float(dat[1])])


for id, name in enumerate(Uma_name):
    index = [i for i, x in enumerate(Uma) if x == name]

    input = []
    correct = []
    for idx in index:
        x1 = float('0.' + ID[idx][4:])

        if Cor[idx] == "芝":
            x2 = -1
        elif Cor[idx] == "ダ":
            x2 = 1
        elif Cor[idx] == "障芝":
            x2 = -5
        elif Cor[idx] == "障ダ":
            x2 = 5
        elif Cor[idx] == "障":
            x2 = 0

        x3 = float('0.' + Dis[idx])

        BABA = Bab[idx][0:2]
        if BABA == "良" or BABA == "良  " or BABA == "良 ":
            x4 = 1.0
        elif BABA == "稍重" or BABA == "稍" or BABA == "稍重  ":
            x4 = 0.5
        elif BABA == "重" or BABA == "重  " or BABA == "重 ":
            x4 = -0.5
        elif BABA == "不良" or BABA == "不" or BABA == "不良  ":
            x4 = -1.0

        if Rank[idx] == "取消" or Rank[idx] == "中止" or Rank[idx] == "除外" or Rank[idx] == "取" or Rank[idx] == "中" or Rank[idx] == "除":
            continue
        else:
            correct.append(int(Rank[idx]))

        input.append([x1, x2, x3, x4])

    #print(len(input), len(correct))
    #print(input, correct)
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16 = power.cul_power(input, correct)
    print(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16)
    Uma_csv[id] = [Uma_csv[id][0], Uma_csv[id][1], c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16]
    print(f"{id}/{len(Uma_name)}:{name}")


with open("./data/power/horse_use_14_22.csv", "w", encoding='utf-8_sig', newline="" ) as f:
    writer = csv.writer(f)
    writer.writerows(Uma_csv)

print("finish")
