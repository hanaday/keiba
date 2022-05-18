#import csv
import os
import glob
import datatable as dt
#from datetime import datetime as date
from datetime import datetime as date


todate = date.now()
tstr = todate.strftime('%Y/%m/%d/%H/%M/%S')

# all
files = glob.glob("./data/origin/all/*.csv")
for idx, file in enumerate(files):
    if idx == 0:
        df = dt.fread(file, encoding="utf-8_sig")
    else:
        df_add = dt.fread(file, encoding="utf-8_sig")
        #del df_add[0, :]
        df.rbind(df_add)

#os.rename('./data/total/all/total.csv', './data/total/all/backup/%stotal.csv' %tstr.replace("/", "")) 
# エンコーディングが指定できないなら一度pandasなどにしてから保存
df.to_csv('./data/total/all/file2014_21.csv',bom=True)


"""
# train
files = glob.glob("./data/origin/train/*.csv")
for idx, file in enumerate(files):
    if idx == 0:
        df = dt.fread(file)
    else:
        df_add = dt.fread(file)
        del df[0, :]
        df.rbind(df_add)

os.rename('./data/total/all/train.csv', './data/total/all/backup/%strain.csv' %tstr) 
df.to_csv('./data/total/train/train.csv')


# test
files = glob.glob("./data/origin/test/*.csv")
for idx, file in enumerate(files):
    if idx == 0:
        df = dt.fread(file)
    else:
        df_add = dt.fread(file)
        del df[0, :]
        df.rbind(df_add)

os.rename('./data/total/all/test.csv', './data/total/all/backup/%stest.csv' %tstr) 
df.to_csv('./data/total/test/test.csv')
"""



print("finish")