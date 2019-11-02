import requests
from bs4 import BeautifulSoup

res = requests.get('https://ionicons.com/v2/')

soup = BeautifulSoup(res.text)

icon_names =  [ li.get('class')[0] for li in soup.find_all('li')]

print(icon_names)

with open("ionic-name.txt", "w+") as f:
    names = "\n".join(icon_names)
    f.write(names)
