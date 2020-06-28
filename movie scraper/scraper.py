import argparse
import sys
from bs4 import BeautifulSoup
import requests
import urllib.request as urllib2
import pandas as pd
import re

top_movies_url = 'https://www.imdb.com/chart/top/?ref_=nv_mv_250'
most_pop_movies_url = 'https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm'


def parse_web(url):
    headers = {'accept-language':'en-US'}
    respone = requests.get(url, headers=headers)
    soup = BeautifulSoup(respone.text, 'html.parser')
    title_column = soup.select('#main > div > span > div > div > div.lister > table > tbody > tr > td.titleColumn')
    poster_column = soup.select('#main > div > span > div > div > div.lister > table > tbody > tr > td.posterColumn')
    return title_column, poster_column


def extract_titles_and_posters(parsed_titles, parsed_posters):
    titles = [title.select('a')[0].text for title in parsed_titles]
    years = [re.sub('[^\w]','',year.select('td.titleColumn > span')[0].text) for year in parsed_titles]
    poster_urls = [poster.find('img').get('src') for poster in parsed_posters]
    data = pd.DataFrame({'title':titles, 'year':years, 'poster_url':poster_urls})
    return data


def run(args):

    print('Extracting movie data...')

    if args.movies == 'top250':
        url = top_movies_url
    elif args.movies == 'pop100':
        url = most_pop_movies_url

    parsed_titles = parse_web(url)[0]
    parsed_posters = parse_web(url)[1]

    movie_data = extract_titles_and_posters(parsed_titles, parsed_posters)

    movie_data.to_csv(args.path, sep=';', index=False)

    print('...done')


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Scrapter to get top movie titles and posters')

    # Add the arguments
    parser.add_argument('--movies', 
                        dest='movies', 
                        default='top250', 
                        help='type top250 for the top 250 best rated movie; type pop100 for the current top 100 most popular movies')

    parser.add_argument('--path',
                        dest='path',
                        default='../dash/data/movies.csv',
                        help='output path')

    # Execute the parse_args() method
    args = parser.parse_args()

    run(args)


