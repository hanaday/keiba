import csv
import numpy as np
import random
import os


Year = 2022

#with open("./file2022a.csv", "r", encoding="shift-jis", errors="", newline="" ) as f:
with open("./data/total/all/total.csv", "r", encoding="utf-8_sig", errors="", newline="" ) as f:
  data = csv.reader(f)

  Uma = []
  Jok = []
  Cla = []
  Rank = []
  ID = []
  for idx, dat in enumerate(data):
    #print(dat)
    if idx == 0:
      id_id = dat.index('レースID')
      id_uma = dat.index('馬名')
      id_jok = dat.index('騎手')
      id_cla = dat.index('レースクラス')
      id_ran = dat.index('順位')
      #print(row_id_uma, row_id_jok)
    else:
      ID.append(dat[id_id])
      Uma.append(dat[id_uma])
      Jok.append(dat[id_jok])
      Cla.append(dat[id_cla])
      Rank.append(dat[id_ran])



Uma_csv = []
Uma_name = []
Uma_str = ''
Jok_csv = []
Jok_str = ''
Jok_name = []

if os.path.exists("./data/power/horse.csv"):
    with open("./data/power/horse.csv", "r", encoding="utf-8_sig", errors="", newline="" ) as f:
      data = csv.reader(f)
      for idx, dat in enumerate(data):
          Uma_str += ('/' + dat[0] + '/')
          Uma_name.append(dat[0])
          Uma_csv.append([dat[0], float(dat[1])]) #, float(dat[2]), float(dat[3]), float(dat[4]), float(dat[5])])

if os.path.exists("./data/power/jokey.csv"):
    with open("./data/power/jokey.csv", "r", encoding="utf-8_sig", errors="", newline="" ) as f:
      data = csv.reader(f)
      for idx, dat in enumerate(data):
          Jok_str += ('/' + dat[0] + '/')
          Jok_name.append(dat[0])
          Jok_csv.append([dat[0], float(dat[1])])




for idx, name in enumerate(Uma):
  if Cla[idx] == 'G1':
      C = 36
  elif Cla[idx] == 'G2':
      C = 34
  elif Cla[idx] == 'G3':
      C = 32
  else:
      C = 18
  if Rank[idx] == "取消" or Rank[idx] == "中止" or Rank[idx] == "除外" or Rank[idx] == "取" or Rank[idx] == "中" or Rank[idx] == "除":
      R = 0
  else:
      R = int(Rank[idx])
  x = C - R
  if x < 0:
      x = 0
  alpha = 0.8**(Year - int(ID[idx][0:4]))
  if ('/' + name + '/') not in Uma_str:
    Uma_str += ('/' + name + '/')
    Uma_csv.append([name, alpha*x])
    Uma_name.append(name)
  else:
    index = Uma_name.index(name)
    beta = 0.5 * 0.8**int(Year - int(ID[idx][0:4]))
    Uma_csv[index][1] = ((1-beta)*Uma_csv[index][1] + beta*alpha*x)

  print(f"{idx}/{len(Uma)}")


for idx, name in enumerate(Jok):
  name = name.replace('．', '')
  if len(name) > 3:
      name = name[0:3]
  C = 18
  
  if Rank[idx] == "取消" or Rank[idx] == "中止" or Rank[idx] == "除外" or Rank[idx] == "取" or Rank[idx] == "中" or Rank[idx] == "除":
      R = 0
  else:
      R = int(Rank[idx])
  x = C - R
  if x < 0:
      x = 0
  alpha = 0.8**int(Year - int(ID[idx][0:4]))
  if ('/' + name + '/') not in Jok_str:
    Jok_str += ('/' + name + '/')
    Jok_csv.append([name, alpha*x])
    Jok_name.append(name)
  else:
    index = Jok_name.index(name)
    beta = 0.5 * 0.8**(Year - int(ID[idx][0:4]))
    Jok_csv[index][1] = ((1-beta)*Jok_csv[index][1] + beta*alpha*x)

  print(f"{idx}/{len(Jok)}")


with open("./data/power/horse.csv", "w", encoding='utf-8_sig', newline="" ) as f:
  writer = csv.writer(f)
  writer.writerows(Uma_csv)

with open("./data/power/jokey.csv", "w", encoding='utf-8_sig', newline="" ) as f:
  writer = csv.writer(f)
  writer.writerows(Jok_csv)

