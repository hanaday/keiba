import csv
import numpy as np


file_name = "race202209021210"


Uma_list, Jok_list, data_list = [], [], []

with open("./data/power/horse_use_14_22.csv", "r", encoding='utf-8_sig') as f:
  data = csv.reader(f)
  for dat in data:
    Uma_list.append(dat)

with open("./data/power/jokey.csv", "r", encoding='utf-8_sig') as f:
  data = csv.reader(f)
  for dat in data:
    Jok_list.append(dat)

with open("./data/origin/predict/%s.csv" %file_name, "r", encoding="utf-8_sig", errors="", newline="" ) as f:
  data = csv.reader(f)

  for idx, dat in enumerate(data):
    if idx == 0:
      id_musi = dat.index('レースクラス')
      dat = dat[0:id_musi] + dat[id_musi+1:]
      id_id = dat.index('レースID')
      id_uma = dat.index('馬名')
      id_jok = dat.index('騎手')
      id_cor = dat.index('コース')
      id_dis = dat.index('距離')
      id_bab = dat.index('馬場')
      #id_odd = dat.index('オッズ')
      data_list.append(dat)
      continue

    dat = dat[0:id_musi] + dat[id_musi+1:]
    #print(dat)

    Uma_list = np.array(Uma_list)
    #print(Uma_list[:, 0])
    index = np.where(dat[id_uma] == Uma_list[:, 0])
    #print(index)
    #index = Uma_list[:][0].index(dat[5])
    #print(float(Uma_list[index, 1]))
    #print(dat[5])
    dat[id_uma] = float(Uma_list[index, 1])

    dat[id_id] = float('0.' + dat[id_id][4:]) * float(Uma_list[index, 2])

    dat[id_dis] = float('0.' + dat[id_dis]) * float(Uma_list[index, 4])

    

    Jok_list = np.array(Jok_list)
    #print(Jok_list[:, 0])
    name = dat[id_jok]
    name = name.replace('．', '')
    if len(name) > 3:
        index = np.where(name[0:3] == Jok_list[:, 0])
    else:
        index = np.where(name == Jok_list[:, 0])
    #index = Jok_list[:][0].index(dat[6])
    dat[id_jok] = float(Jok_list[index, 1])

    if dat[id_cor] == "芝":
      dat[id_cor] = -1
    elif dat[id_cor] == "ダ":
      dat[id_cor] = 1
    elif dat[id_cor] == "障芝":
      dat[id_cor] = -5
    elif dat[id_cor] == "障ダ":
      dat[id_cor] = 5
    elif dat[id_cor] == "障":
      dat[id_cor] = 0
    dat[id_cor] = dat[id_cor] * float(Uma_list[index, 3])

    BABA = dat[id_bab][0:2]
    if BABA == "良" or BABA == "良  " or BABA == "良 ":
      dat[id_bab] = 1.0
    elif BABA == "稍重" or BABA == "稍" or BABA == "稍重  ":
      dat[id_bab] = 0.5
    elif BABA == "重" or BABA == "重  " or BABA == "重 ":
      dat[id_bab] = -0.5
    elif BABA == "不良" or BABA == "不" or BABA == "不良  ":
      dat[id_bab] = -1.0
    dat[id_bab] = dat[id_bab] * float(Uma_list[index, 5])


    data_list.append(dat)
    print(idx)

with open("./data/predict/%s.csv" %file_name, "w", encoding='utf-8_sig', newline="" ) as f:
    writer = csv.writer(f)
    writer.writerows(data_list)

