from kafka import KafkaProducer
import requests
import json
import datetime


def parse_stock(stockstr: str):
    stockId = stockstr.split('=')[0]
    content = stockstr[11:].split(',')
    return {
        'stockId': stockId,
        'stockName': content[0],
        'todayStartPrice': content[1],
        'yesterdayEndPrice': content[2],
        'currentPrice': content[3],
        'todayTopPrice': content[4],
        'todayLowPrice': content[5],
        'buyInPrice': content[6],
        'sellOutPrice': content[7],
        'completedNumber': content[8],
        'completedPrice': content[9],
        'firstBuyInNumber': content[10],
        'firstBuyInPrice': content[11],
        'secondBuyNumber': content[12],
        'secondBuyPrice': content[13],
        'thirdBuyInNumber': content[14],
        'thirdBuyInPrice': content[15],
        'forthBuyNumber': content[16],
        'forthBuyPrice': content[17],
        'fifthBuyNumber': content[18],
        'fifthBuyPrice': content[19],
        'firstSellOutNumber': content[20],
        'firstSellOutPrice': content[21],
        'secondSellOutNumber': content[22],
        'secondSellOutPrice': content[23],
        'thirdSellOutNumber': content[24],
        'thirdSellOutPrice': content[25],
        'forthSellOutNumber': content[26],
        'forthSellOutPrice': content[27],
        'fifthSellOutNumber': content[28],
        'fifthSellOutPrice': content[29],
        'date': content[30],
        'time': content[31]
    }


url = 'http://hq.sinajs.cn/list=sh600000,sh600008,sh600009,sh600010,sh600011,sh600015,sh600016,sh600018,sh600019,sh600021,sh600023,sh600028,sh600029,sh600030,sh600031,sh600036,sh600038,sh600048,sh600050,sh600061,sh600066,sh600068,sh600074,sh600085,sh600089,sh600100,sh600104,sh600109,sh600111,sh600115,sh600118,sh600153,sh600157,sh600170,sh600177,sh600188,sh600196,sh600208,sh600219,sh600221,sh600233,sh600271,sh600276,sh600297,sh600309,sh600332,sh600340,sh600352,sh600362,sh600369,sh600372,sh600373,sh600376,sh600383,sh600390,sh600406,sh600415,sh600436,sh600482,sh600485,sh600489,sh600498,sh600518,sh600519,sh600522,sh600535,sh600547,sh600549,sh600570,sh600583,sh600585,sh600588,sh600606,sh600637,sh600649,sh600660,sh600663,sh600674,sh600682,sh600685,sh600688,sh600690,sh600703,sh600704,sh600705,sh600739,sh600741,sh600795,sh600804,sh600816,sh600820,sh600827,sh600837,sh600871,sh600886,sh600887,sh600893,sh600895,sh600900,sh600909,sh600919,sh600926,sh600958,sh600959,sh600977,sh600999,sh601006,sh601009,sh601012,sh601018,sh601021,sh601088,sh601099,sh601111,sh601117,sh601118,sh601155,sh601163,sh601166,sh601169,sh601186,sh601198,sh601211,sh601212,sh601216,sh601225,sh601228,sh601229,sh601288,sh601318,sh601328,sh601333,sh601336,sh601375,sh601377,sh601390,sh601398,sh601555,sh601600,sh601601,sh601607,sh601608,sh601611,sh601618,sh601628,sh601633,sh601668,sh601669,sh601688,sh601718,sh601727,sh601766,sh601788,sh601800,sh601818,sh601857,sh601866,sh601872,sh601877,sh601878,sh601881,sh601888,sh601898,sh601899,sh601901,sh601919,sh601933,sh601939,sh601958,sh601966,sh601985,sh601988,sh601989,sh601991,sh601992,sh601997,sh601998,sh603160,sh603799,sh603833,sh603858,sh603993,sz000001,sz000002,sz000008,sz000060,sz000063,sz000069,sz000100,sz000157,sz000166,sz000333,sz000338,sz000402,sz000413,sz000415,sz000423,sz000425,sz000503,sz000538,sz000540,sz000559,sz000568,sz000623,sz000625,sz000627,sz000630,sz000651,sz000671,sz000686,sz000709,sz000723,sz000725,sz000728,sz000738,sz000750,sz000768,sz000776,sz000783,sz000792,sz000826,sz000839,sz000858,sz000876,sz000895,sz000898,sz000938,sz000959,sz000961,sz000963,sz000983,sz001979,sz002007,sz002008,sz002024,sz002027,sz002044,sz002065,sz002074,sz002081,sz002142,sz002146,sz002153,sz002174,sz002202,sz002230,sz002236,sz002241,sz002252,sz002292,sz002294,sz002304,sz002310,sz002352,sz002385,sz002411,sz002415,sz002424,sz002426,sz002450,sz002456,sz002460,sz002465,sz002466,sz002468,sz002470,sz002475,sz002500,sz002508,sz002555,sz002558,sz002572,sz002594,sz002601,sz002602,sz002608,sz002624,sz002673,sz002714,sz002736,sz002739,sz002797,sz002831,sz002839,sz002841,sz300003,sz300015,sz300017,sz300024,sz300027,sz300033,sz300059,sz300070,sz300072,sz300122,sz300124,sz300136,sz300144,sz300251,s300315'

response = requests.get(url)

current_date = datetime.datetime.now().strftime("%Y-%m-%d")

# producer = KafkaProducer(bootstrap_servers=['hdp2.domain:6667', 'hdp3.domain:6667', 'hdp4.domain:6667'], api_version=(0, 10, 1), value_serializer=lambda v: json.dumps(v).encode('utf-8'))
producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'], api_version=(0, 10, 1),
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

for line in response.content.decode('gb2312').split(';')[:-1]:
    content = line.split('_')[2].replace('="', ',').replace('"', '')
    if len(content.split(',')) > 31 and content.split(',')[31] == current_date:
        print(content)
        # print(parse_stock(content))
        # producer.send('test', value=content)
        # producer.flush()



response = response.content.decode('gb2312').split(';')[:-1]

response = map(lambda x: x.split('_')[2].replace('="', ',').replace('"', ''), response)
response = filter(lambda x: len(x.split(',')) > 30, response)
response = map(parse_stock, response)
response = filter(lambda x: x['date'] == current_date, response)

for line in response:
    producer.send('test1', value=line)
    print(line)
producer.flush()
