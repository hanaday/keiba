#スクレイピングに必要なモジュール
import requests
from bs4 import BeautifulSoup
import pandas as pd


save_path = "./data/origin/predict/race2022.csv"


# レースIDの作成
# 2022 05 01 08 11

import itertools
YEAR = ['2022']
CODE = ['09'] #['03', '05', '09'] #[str(num+1).zfill(2) for num in range(4, 10)] # 会場
RACE_COUNT = ['02'] #['01', '02'] #[str(num+1).zfill(2) for num in range(2)] #['01']  # 01--06 何回目
DAYS = ['12'] #[str(num+1).zfill(2) for num in range(9)] #['01']  # 01--09 何日目
RACE_NUM = ['11'] #['10', '11']  # 10, 11 何レース目
race_ids = list(itertools.product(YEAR,CODE,RACE_COUNT,DAYS,RACE_NUM))

# サイトURLの作成
# https://race.netkeiba.com/race/shutuba.html?race_id=202205020311&rf=race_list
#SITE_URL = ["https://race.netkeiba.com/race/result.html?race_id={}".format(''.join(race_id)) for race_id in race_ids]
SITE_URL = ["https://race.netkeiba.com/race/shutuba.html?race_id={}&rf=race_list".format(''.join(race_id)) for race_id in race_ids]

import time #sleep用
import sys  #エラー検知用
import re   #正規表現
import numpy    #csv操作
import pandas as pd


result_df = pd.DataFrame()


print(len(SITE_URL))


#サイトURLをループしてデータを取得する
for sitename,race_id in zip(SITE_URL,race_ids):
    print(sitename)

    # 時間をあけてアクセスするように、sleepを設定する
    time.sleep(3)
    
    try:
        # スクレイピング対象の URL にリクエストを送り HTML を取得する
        res = requests.get(sitename)
        #res.encoding = 'utf-8_jis'
        print(res.encoding)

        res.raise_for_status()  #URLが正しくない場合，例外を発生させる

        # レスポンスの HTML から BeautifulSoup オブジェクトを作る
        soup = BeautifulSoup(res.content, 'html.parser')

        # title タグの文字列を取得する
        title_text = soup.find('title').get_text()
        print(title_text)
        if "G1" in title_text:
            title = "G1"
        elif "G2" in title_text:
            title = "G2"
        elif "G3" in title_text:
            title = "G3"
        else:
            title = "Other"

        #順位のリスト作成
        #Ranks = soup.find_all('div', class_='Rank')
        #Ranks_list = []
        #for Rank in Ranks:
        #    Rank = Rank.get_text()
        #    #リスト作成
        #    Ranks_list.append(Rank)

        #馬名取得
        Horse_Names = soup.find_all('span', class_='HorseName')
        Horse_Names_list = []
        for Horse_Name in Horse_Names:
            #馬名のみ取得(lstrip()先頭の空白削除，rstrip()改行削除)
            Horse_Name = Horse_Name.get_text().lstrip().rstrip('\n')
            #リスト作成
            Horse_Names_list.append(Horse_Name)
        Horse_Names_list = Horse_Names_list[1:]

        #オッズ取得
        Odds = soup.find_all('td', class_=re.compile('Txt_R Popular'))
        Odds_list = []
        Odds_list = numpy.zeros(len(Horse_Names_list))
        #for Odd in Odds:
        #    Odd = Odd.getText()
        #    #Odd = Odd.get_text().replace('\n','')
        #    print(Odd)
            #リスト作成
        #    Odds_list.append(Odd)


        #枠取得
        Wakus = soup.find_all('td', class_=re.compile(".*Waku.*"))
        Wakus_list = []
        for Waku in Wakus:
            Waku = Waku.get_text().replace('\n','')
            #リスト作成
            Wakus_list.append(Waku)


        #馬番取得
        Umabans = soup.find_all('td', class_=re.compile(".*Umaban.*"))
        Umabans_list = []
        for Umaban in Umabans:
            Umaban = Umaban.get_text().replace('\n','')
            #リスト作成
            Umabans_list.append(Umaban)


        #騎手取得
        Kishus = soup.find_all('td', class_=re.compile("^Jockey$"))
        Kishus_list = []
        for Kishu in Kishus:
            Kishu = Kishu.get_text().replace('\n','').replace(' ','').replace('△','').replace('▲','').replace('☆','').replace('★','')
            #リスト作成
            Kishus_list.append(Kishu)
            if len(Kishus_list) == len(Umabans_list):
                break


        #コース,距離取得
        Distance_Course = soup.find_all('span')
        Babas = soup.find_all('span', class_=re.compile("^Item03$"))
        Babas_list = []
        Baba = ""
        for Baba1 in Babas:
            Baba = Baba1.get_text().replace('\n','').replace('/ ','').replace('馬場:','')
        if Baba == "":
            Baba = "不"

        Distance_Course = re.search(r'.[0-9]+m', str(Distance_Course))
        Course = Distance_Course.group()[0]
        if Course == "障":
            Course = Distance_Course.group()[0:2]
        Distance = re.sub("\\D", "", Distance_Course.group())


        df = pd.DataFrame({
            'レースID':''.join(race_id),
            'レースクラス':title,
            '枠':Wakus_list,
            '馬番':Umabans_list,
            '馬名':Horse_Names_list,
            '騎手':Kishus_list,
            'コース':Course,
            '距離':Distance,
            '馬場':Baba,
            'オッズ':Odds_list,
            #'順位':Ranks_list,
        })
        
        result_df = pd.concat([result_df,df],axis=0)
        print("取得")
    except:
        print(sys.exc_info())
        print("サイト取得エラー")


#print(result_df)
result_df.to_csv(save_path, encoding='utf-8_sig')
print("finish")
