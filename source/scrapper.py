import click
import csv
from matplotlib import pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import pandas as pd


_OUTPUT_FOLDER = "dataset/"
_OUTPUT_FILEPATH = "dataset.csv"
_OUTPUT_PNG_FILEPATH = "plot.png"
_DEFAULT_PAGE = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
_BASE_PAGE = "https://en.wikipedia.org"


def _fetch_data(url):
    """Fetch data from the given URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None


def _parse_data(html):
    """Parse the retrieved HTML, extract relevant data and clean it."""
    soup = BeautifulSoup(html, 'html.parser')
    
    table = soup.find("table", {"class": "wikitable"})
    rows = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if cols:
            rows.append([col.text.strip() for col in cols])

    df = pd.DataFrame(rows, columns=["Country/Territory", "Forecast (IMF)", "Year (IMF)", "Estimate (WB)", "Year (WB)", "Estimate (UN)", "Year (UN)"])

    df.replace('', np.nan, inplace=True)
    df.iloc[:, 1:-3] = df.iloc[:, 1:-3].replace({r'[^\d.]': '', r',': ''}, regex=True)
    df.iloc[:, 1:-3] = df.iloc[:, 1:-3].apply(pd.to_numeric, errors='coerce')
    df.iloc[:, 1:-3] = df.iloc[:, 1:-3].astype(float)
    df.iloc[:, -3:] = df.iloc[:, -3:].replace({r'[^\d]': '', r',': ''}, regex=True)
    df.iloc[:, -3:] = df.iloc[:, -3:].replace('', np.nan)
    df.iloc[:, -3:] = df.iloc[:, -3:].infer_objects()
    df.iloc[:, -3:] = df.iloc[:, -3:].fillna(-1).astype(int)
        
    return df


def _discover_data(html):
    """Parse the retrieved HTML on aditional data, extract it and clean it."""
    soup = BeautifulSoup(html, 'html.parser')
    target_url = None
    for link in soup.find_all('a'):
        if link.get('title') == "List of countries by GDP (PPP)":
            target_url = _BASE_PAGE + link.get('href')
            break
    if not target_url:
        raise ValueError("Target page not found while discovering adittional data")

    response = requests.get(target_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find("table", class_="wikitable")
    if not table:
        raise ValueError("Table not found while discovering adittional data")
    rows = []
    for tr in table.find_all('tr')[1:]:
        cols = [td.text.strip() for td in tr.find_all('td')]
        if len(cols) >= 7:
            rows.append(cols[:7])
    df = pd.DataFrame(rows, columns=["Country/Territory", "Forecast (IMF)", "Year (IMF)", "Estimate (WB)", "Year (WB)", "Estimate (CIA)", "Year (CIA)"])
    print("\n RESULTS FROM DISCOVERING ADDITIONAL DATA:")
    print(df)


def _save_to_png(data):
    """Save extracted data to a PNG file."""
    os.makedirs(os.path.dirname(_OUTPUT_FOLDER), exist_ok=True)

    data = data.sort_values(by=data.columns[1], ascending=False)

    plt.figure(figsize=(30, 20))
    plt.bar(data.iloc[:, 0], data.iloc[:, 1])
    plt.xticks(rotation=90)
    plt.savefig(_OUTPUT_FOLDER + _OUTPUT_PNG_FILEPATH)
    plt.close()


def _save_to_csv(data):
    """Save extracted data to a CSV file."""
    os.makedirs(os.path.dirname(_OUTPUT_FOLDER), exist_ok=True)
    
    data_rows = data.values.tolist()
    with open(_OUTPUT_FOLDER + _OUTPUT_FILEPATH, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data.columns)
        writer.writerows(data_rows)


@click.command()
@click.option('--url', required=False, default=_DEFAULT_PAGE, help='Target URL to scrape')
@click.option('--save-csv', required=False, is_flag=True, default=False, help='Save locally the output in a CSV file')
@click.option('--save-png', required=False, is_flag=True, default=False, help='Save locally a plot of the output in a png file')
def main(url, save_csv:bool, save_png:bool):
    html = _fetch_data(url)
    if html:
        data = _parse_data(html)
        _discover_data(html)
        if not data.empty:
            print("Data scraping was successfull")
            if save_csv:
                _save_to_csv(data)
                print(f"    - Results saved in csv: {_OUTPUT_FILEPATH}")
            if save_png:
                _save_to_png(data)
                print(f"    - Results saved in png: {_OUTPUT_PNG_FILEPATH}")
        else:
            raise Exception(f"There was an error parsing de data from the URL: {url}")
    else:
        raise Exception(f"There was an error fetching de data from the URL: {url}")

if __name__ == "__main__":
    main()
