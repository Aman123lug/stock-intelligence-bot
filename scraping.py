import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re


companies = [
    "TCS", "INFY", "RELIANCE", "HDFCBANK", "ICICIBANK",
    "ITC", "BHARTIARTL", "MARUTI", "TITAN", "HDFC",
    "HINDUNILVR", "KOTAKBANK", "HCLTECH", "AXISBANK", "BAJAJFINSV",
    "BAJAJ-AUTO", "NESTLEIND", "INDUSINDBK", "SBIN", "DRREDDY",
    "ULTRACEMCO", "TECHM"
]

# Base URL
BASE_URL = "https://www.screener.in/company/{}/"

# Headers to avoid bot detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_text(text):
    """Removes newlines, multiple spaces, and trims text."""
    return " ".join(text.split()).strip()


def clean_numeric(value):
    """Converts currency and percentage strings to integers."""
    if isinstance(value, str):
        value = re.sub(r"[₹,%]", "", value)  
        value = re.sub(r"[,.]", "", value)  
        try:
            return int(value) 
        except ValueError:
            return None  
    return value

def scrape_company_data(company):
    """Scrapes financial data for a given company from Screener.in"""
    url = BASE_URL.format(company)
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Failed to fetch {company}. Status code: {response.status_code}")
        return {"Company": company, "Error": "Failed to fetch data"}
    
    soup = BeautifulSoup(response.text, "html.parser")

    try:
        # Extract Company Description & Revenue Breakup
        about_section = soup.select_one(".about p")
        company_description = clean_text(about_section.text) if about_section else "N/A"

        revenue_section = soup.select_one(".commentary p")
        revenue_breakup = clean_text(revenue_section.get_text(separator=" | ")) if revenue_section else "N/A"

        # Extract Key Financial Ratios
        financial_data = {}
        for li in soup.select("#top-ratios li"):
            key = clean_text(li.select_one(".name").text)
            value = clean_text(li.select_one(".value").text)
            financial_data[key] = clean_numeric(value)  # Convert to int

        return {
            "Company": company,
            "Description": company_description,
            "Revenue Breakup": revenue_breakup,
            "Market Cap": financial_data.get("Market Cap", None),
            "Current Price": financial_data.get("Current Price", None),
            "High/Low": financial_data.get("High / Low", None),
            "Stock P/E": financial_data.get("Stock P/E", None),
            "Book Value": financial_data.get("Book Value", None),
            "Dividend Yield": financial_data.get("Dividend Yield", None),
            "ROCE": financial_data.get("ROCE", None),
            "ROE": financial_data.get("ROE", None),
            "Face Value": financial_data.get("Face Value", None),
        }

    except Exception as e:
        print(f"Error scraping {company}: {e}")
        return {"Company": company, "Error": "Scraping error"}


def main():
    """Runs the scraper for multiple companies"""
    data = []
    for company in companies:
        print(f"Scraping: {company}")
        company_data = scrape_company_data(company)
        data.append(company_data)
        time.sleep(random.uniform(2, 5))
    
    # Save data to CSV
    df = pd.DataFrame(data)
    df.to_csv("data/screener_data_cleaned.csv", index=False)
    print(" Scraping complete! Data saved to screener_data_cleaned.csv")

if __name__ == "__main__":
    main()
