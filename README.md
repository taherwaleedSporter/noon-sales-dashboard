# Noon Sales Dashboard (USD only)

A simple Streamlit app that combines:
- Noon orders export
- Product reference file
- FX rates file

## What it shows
- Daily Performance
- Monthly performance
- Product Orders
- Product Sales
- Product Metrics
- Data Quality

## Expected files
### Orders file
Required columns:
- partner_sku
- status
- order_timestamp
- currency_code
- offer_price

### Product reference
Required columns:
- partner_sku
- product_title
- brand
- cogs_aed
- fee_percent

### FX rates
Required columns:
- currency
- to USD

## Business rules
- Only statuses counted: Processing, Shipped, Delivered
- Reporting date: order_timestamp
- Sales are converted to USD with the FX table
- COGS are stored in AED and converted using the AED row in the FX table
- GP = sales_usd - cogs_usd - estimated_fees_usd
- Each row in the Noon file is treated as one order

## Run locally
1. Install Python 3.11+ if needed
2. In Terminal:
   pip install -r requirements.txt
   streamlit run app.py

## Notes
- The app can load the provided sample files automatically when they are in the same folder.
- If a SKU is missing COGS or fee percent, it appears in the Data Quality tab.
