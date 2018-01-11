from bs4 import BeautifulSoup
import requests
import csv
import re

class weather_scraper(object):
    # function for getting the date
    def get_date(self, this_url):
        front = 'http://www.wunderground.com/history/airport'
        date = this_url[len(front) + 6:this_url.index('DailyHistory') - 1]
        return date

    def get_next_url(self, this_soup):
        div = this_soup.find(class_="next-link")
        url = div.find('a').get('href')
        return 'http://www.wunderground.com' + url

    # takes the zip code and gets the history url
    def get_start_page(self, zip_code, start_date):
        # goes to the history page
        url = 'https://www.wunderground.com/cgi-bin/findweather/getForecast?query=' + str(zip_code)
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'lxml')
        url_hist = soup.find(id='city-nav-history').get('href')

        # goes to the right date
        url_front = 'https://www.wunderground.com/' + url_hist[:url_hist.find('/Daily') - 10]
        url_back = url_hist[url_hist.find('/Daily'):]
        url_full = url_front + start_date + url_back

        return  url_full

    def __init__(self, zip_code, save_file, start_date, end_date):

        start_url = self.get_start_page(zip_code, start_date)

        keys = ['Mean Temperature', 'Max Temperature', 'Min Temperature', 'Precipitation', 'Wind Speed', 'Max Wind Speed', 'Max Gust Speed', 'Visibility', 'Average Humidity', 'Maximum Humidity', 'Minimum Humidity', 'Snow', 'Snow Depth']
        results = []

        this_date = ''
        this_url = start_url

        while this_date != end_date:
            this_date = self.get_date(this_url)
            r = requests.get(this_url)
            soup = BeautifulSoup(r.text, 'lxml')
            table = soup.find('table', id='historyTable')
            table_text = table.get_text()
            table_text = table_text.replace('\n', ' ')
            table_text = table_text.replace('\xa0', '')

            # removes all the double spaces
            while '  ' in table_text:
                table_text = table_text.replace('  ', ' ')

            # removes the double precipitation
            table_text = table_text.replace('Precipitation Precipitation', 'Precipitation')

            new_row = []


            # goes through each of the key indicators I want to collect
            for k in keys:
                # the if statement takes care of if the indicator isn't pressent for that day.
                if k in table_text and " " in table_text[table_text.index(k) + len(k) + 1:]:
                    start = table_text.index(k) + len(k) + 1
                    end = table_text[start:].index(" ") + start
                    # new_row.append(table_text[start:end - 2].replace('m', '').replace('il', ''))
                    text = table_text[start:end]
                    text = re.sub('[^0-9.]', '', text)
                    new_row.append(text)
                else:
                    new_row.append('NA')

            # adds the date
            new_row.append(this_date)

            # adds the data to the results
            results.append(new_row)

            # updates the url and date
            this_url = self.get_next_url(soup)
            this_date = self.get_date(this_url)

        results.insert(0, keys + ['date'])
        with open(save_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)
