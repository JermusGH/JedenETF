"""Temporary script to investigate Invesco API endpoints."""
import re
import requests

url = "https://www.invesco.com/uk/en/financial-products/etfs/invesco-eqqq-nasdaq-100-ucits-etf-dist.html"
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

# Look for API URLs
apis = re.findall(r'(https?://[^"\'<>\s]*?(?:api|dng)[^"\'<>\s]*)', resp.text)
unique_apis = sorted(set(apis))
print("=== API endpoints found ===")
for a in unique_apis[:30]:
    print(f"  {a}")

# Look for holdings-related JSON data
print("\n=== Holdings-related patterns ===")
holdings_matches = re.findall(r'"([^"]*[Hh]olding[^"]*)"', resp.text)
unique_holdings = sorted(set(holdings_matches))
for h in unique_holdings[:20]:
    print(f"  {h}")

# Look for the ISIN in data attributes or JSON
print("\n=== ISIN references ===")
isin_context = re.findall(r'.{0,100}IE0032077012.{0,100}', resp.text)
for ctx in isin_context[:5]:
    print(f"  ...{ctx[:200]}...")
