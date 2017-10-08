import sys
import csv

from urllib.request import urlopen
from bs4 import BeautifulSoup
from house_info import house


def get_city_dict():
    city_dict = {}
    with open('./cities.csv', 'r') as file:
        cities = csv.reader(file)
        for city in cities:
            city_dict[city[0]] = city[1]
        return city_dict


def get_district_dict(url):
    district_dict = {}
    html = urlopen(url).read()
    bsobj = BeautifulSoup(html, 'html5lib')
    roles = bsobj.find('div', {'data-role': 'ershoufang'}).findChildren('a')
    for role in roles:
        district_url = role.get('href')
        district_name = role.get_text()
        district_dict[district_name] = district_url
    return district_dict


def run():
    city_dict = get_city_dict()
    for city in city_dict.keys():
        print(city)
    print()

    input_city = input('Please input city name: ')
    city_url = city_dict.get(input_city)
    if not city_url:
        print('Wrong city name')
        sys.exit()

    ershoufang_city_url = city_url
    district_dict = get_district_dict(ershoufang_city_url)

    for district in district_dict.keys():
        print(district)
    print()

    input_district = input('Please input district name: ')
    district_url = district_dict.get(input_district)

    if not district_url:
        print('Wrong district name')
        sys.exit()

    house_info_url = city_url + district_url[12:]
    print(house_info_url)

    house(house_info_url)


if __name__ == '__main__':
    run()
