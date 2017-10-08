import csv

from urllib.request import urlopen
from bs4 import BeautifulSoup

url = 'https://www.lianjia.com'

html = urlopen(url).read()

bsobj = BeautifulSoup(html, 'html5lib')
city_tags = bsobj.find('div', {'class': 'link-list'}).div.dd.findChildren('a')

with open('./cities.csv', 'w') as file:
    writer = csv.writer(file)
    for city_tag in city_tags:
        city_url = city_tag.get('href')
        city_name = city_tag.get_text()
        writer.writerow((city_name, city_url))
        print(city_name, city_url)
