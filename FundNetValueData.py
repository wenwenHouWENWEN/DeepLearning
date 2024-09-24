import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta


def fetch_fund_data(fund_name, fund_code):
    base_api_url = f'http://fundf10.eastmoney.com/F10DataApi.aspx?type=lsjz&code={fund_code}&page='

    current_date = datetime.now()
    two_years_ago = current_date - timedelta(days=2 * 365)  

    page = 1
    all_data = []

    while True:
        api_url = base_api_url + str(page)
        response = requests.get(api_url)

        if response.status_code != 200:
            print(f"Failed to fetch data from page {page}, status code: {response.status_code}")
            break

        soup = BeautifulSoup(response.text, 'html.parser')
        table_rows = soup.find_all('tr')

        if len(table_rows) <= 1:
            break

        for row in table_rows[1:]:
            cols = row.find_all('td')
            if len(cols) < 2:
                continue
            date = cols[0].text.strip()
            nav = cols[1].text.strip()
            if nav and nav != '暂无数据':
                data_date = datetime.strptime(date, '%Y-%m-%d')
                if data_date >= two_years_ago:
                    all_data.append([date, nav])
                else:
                    break

        if data_date < two_years_ago:
            print(
                f"Data fetching stopped at page {page} because date {data_date.strftime('%Y-%m-%d')} is older than two years ago.")
            break

        print(f"Fetched page {page} data for {fund_name}.")
        page += 1

    df = pd.DataFrame(all_data, columns=['Date', 'Net Value'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)

    output_file = f"{fund_name}_net_value_2years.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Data for {fund_name} saved to {output_file}")


# List of all funds with their names and codes
funds = [
    ('China Europe Medical and Health Hybrid C', '003096'),
    ('Tianhong CSI Photovoltaic Industry Index C', '011103'),
    ('Guotai CSI New Energy Vehicle Index A', '160225'),
    ('China Asset Management CSI Semiconductor Chip ETF Connection C', '008888'),
    ('ICBC New Energy Vehicle Hybrid C', '005940')
]

# Fetch data for each fund
for fund_name, fund_code in funds:
    fetch_fund_data(fund_name, fund_code)


# Chinese version
'''
funds = [
    ('中欧医疗健康混合C', '003096'),
    ('天弘中证光伏产业指数C', '011103'),
    ('国泰国证新能源汽车指数A', '160225'),
    ('华夏国证半导体芯片ETF联接C', '008888'),
    ('工银新能源汽车混合C', '005940')
]

# 对每个基金进行数据抓取
for fund_name, fund_code in funds:
    fetch_fund_data(fund_name, fund_code)
'''
