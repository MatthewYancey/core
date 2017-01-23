#Beautiful Soup
from bs4 import BeautifulSoup
import requests

#set up
bad_url = 'http://adventuretime.wikia.com/wiki/asdf'
url = 'http://adventuretime.wikia.com/wiki/Episodes'

#Requests Package
at = requests.get(bad_url)
at
at = requests.get(url)
at

#Soup
#basic commands
soup = BeautifulSoup(at.text)
soup.prettify
soup.title
soup.find('tr')
soup.find_all('dl')
soup.find(class_='skiplink')
soup.find('div')
soup.find('div', class_='page-width-container')
div = soup.find('div')
div.get('class')

#example
results = soup.find_all(style='text-align: center; background: #f2f2f2')

#gets all show links
links = []
for r in results:
	link = r.find_all('a');
	links.append(link[1].get('href'))

short_url = 'http://adventuretime.wikia.com'
at = requests.get(short_url + links[0] + '/Transcript') #adds transcript
soup = BeautifulSoup(at.text)
dls = soup.find_all('dl')

#gets the text
for dl in dls:
	dl.get_text() 