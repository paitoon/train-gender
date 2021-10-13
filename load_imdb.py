import os
import requests
import bs4
import urllib

base = "http://www.imdb.com"
gender = "male"
images = "images"

rootPath = os.path.join(images, gender)
if not os.path.exists(rootPath):
    os.mkdir(rootPath)

for start in range(200, 4000, 100):
    print("Sending GET request ...")

    url = '{}/search/name?gender={}&count=100&start={}'.format(base, gender, start)
    r = requests.get(url)
    html = r.text
    soup = bs4.BeautifulSoup(html, 'html.parser')

    for img in soup.select('.lister-item .lister-item-image'):
        link = img.find('a').get('href')
        name = img.find('img').get('alt')

        print("Going to {} profile ...".format(name))

        r = requests.get(base + link)
        html = r.text
        soup = bs4.BeautifulSoup(html, 'html.parser')
        selector = soup.find('time')
        if selector is None:
            continue
        date = selector.get('datetime')

        selector = soup.find('img', {"id": "name-poster"})
        if selector is None:
            continue
        image = selector.get('src')

        print("Downloading profile picture ...")
        image_file = urllib.request.urlopen(image)
        imagePath = os.path.join(rootPath, "{}_{}_{}.jpg".format(gender, start, date))
        with open(imagePath, 'wb') as output:
            output.write(image_file.read())
