from bs4 import BeautifulSoup
import requests
import pandas as pd
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
        this_date = ''
        this_url = start_url
        res = []

        while this_date != end_date:
            this_date = self.get_date(this_url)
            r = requests.get(this_url)
            soup = BeautifulSoup(r.text, 'lxml')
            table = soup.find('div', id='observations_details')

            # loops through the hourly data. This is not always present
            try:
                table = table.find('tbody')

                for r in table.find_all('tr'):
                    new_row = []

                    for entry in r.find_all('td'):
                        text = entry.text.replace('\n', '')
                        text = text.replace('\xa0', '')
                        text = text.replace('\t', '')
                        new_row.append(text)

                    # sometimes windchill is missing, this corrects the spacing
                    if len(new_row) == 12:
                        new_row = new_row[:2] + ['NA'] + new_row[2:]

                    # appends the date and zip values
                    new_row.append(zip_code)
                    new_row.append(this_date)

                    # appends the new rows
                    res.append(new_row)
            except:
                pass

            # updates the url and date
            this_url = self.get_next_url(soup)
            this_date = self.get_date(this_url)

        # saves the results
        res = pd.DataFrame(res, columns=['time', 'temp', 'wind_chill', 'dew_point', 'humidity', 'pressure', 'visibility', 'wind_dir', 'wind_speed', 'gust_speed', 'precip', 'events', 'conditions', 'zip', 'date'])
        res.to_csv(save_file, index = False)
