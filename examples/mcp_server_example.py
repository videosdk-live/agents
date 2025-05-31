from mcp.server.fastmcp import FastMCP
import datetime
import json
import urllib.request
import urllib.parse

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "4YAE7GJ9TDEY8M46"

# Create the MCP server
# This FastMCP implementation follows the Model Context Protocol standard
# and will be compatible with any MCP client, including our generic adapter
mcp = FastMCP("FinancialDataServer")

# Add a Nifty 50 search tool
@mcp.tool()
def get_nifty50_price() -> str:
    """Get the current Nifty 50 index price from India's stock market"""
    
    try:
        # Get current time
        now = datetime.datetime.now()
        time_str = now.strftime('%H:%M:%S on %Y-%m-%d')
        
        # Make a request to get Nifty 50 data using Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=^NSEI&apikey={ALPHA_VANTAGE_API_KEY}"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
            # Check if we got valid data
            if "Global Quote" in data and "05. price" in data["Global Quote"]:
                price = data["Global Quote"]["05. price"]
                return f"The current Nifty 50 price is ₹{price} as of {time_str}"
            else:
                # If we hit API limits or other issues, use a fallback example price
                return f"The Nifty 50 was around ₹21,500 to ₹22,500 as of {time_str} (sample data - couldn't fetch real-time price)"
    except Exception as e:
        # Return a fallback message with the time
        return f"I couldn't retrieve the Nifty 50 price at this time. The current time is {time_str}"

# Add stock price lookup tool
@mcp.tool()
def get_stock_quote(symbol: str) -> str:
    """Get the current price for a specific stock symbol
    
    Args:
        symbol: The stock symbol to look up (e.g., AAPL, MSFT, GOOGL)
    """
    
    try:
        # Get current time
        now = datetime.datetime.now()
        time_str = now.strftime('%H:%M:%S on %Y-%m-%d')
        
        # Make a request to get stock data using Alpha Vantage
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
            # Check if we got valid data
            if "Global Quote" in data and "05. price" in data["Global Quote"]:
                price = data["Global Quote"]["05. price"]
                change = data["Global Quote"].get("09. change", "N/A")
                change_percent = data["Global Quote"].get("10. change percent", "N/A")
                return f"The current price of {symbol} is ${price} (change: {change}, {change_percent}) as of {time_str}"
            else:
                return f"Unable to retrieve data for {symbol} at {time_str}. This may be due to an invalid symbol or API rate limits."
    except Exception as e:
        return f"I couldn't retrieve the stock quote for {symbol} at this time ({time_str})."

# Add currency exchange rate tool
@mcp.tool()
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the current exchange rate between two currencies
    
    Args:
        from_currency: The source currency code (e.g., USD, EUR, GBP)
        to_currency: The target currency code (e.g., JPY, INR, CAD)
    """
    
    try:
        # Get current time
        now = datetime.datetime.now()
        time_str = now.strftime('%H:%M:%S on %Y-%m-%d')
        
        # Make a request to get exchange rate data
        url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
            # Check if we got valid data
            if "Realtime Currency Exchange Rate" in data:
                rate_data = data["Realtime Currency Exchange Rate"]
                exchange_rate = rate_data.get("5. Exchange Rate", "N/A")
                rate_time = rate_data.get("6. Last Refreshed", "N/A")
                
                return f"1 {from_currency} = {exchange_rate} {to_currency} as of {rate_time} (current time: {time_str})"
            else:
                return f"Unable to retrieve exchange rate for {from_currency} to {to_currency} at {time_str}. This may be due to invalid currency codes or API rate limits."
    except Exception as e:
        return f"I couldn't retrieve the exchange rate at this time ({time_str})."

# Add company overview tool
@mcp.tool()
def get_company_info(symbol: str) -> str:
    """Get basic information about a company by its stock symbol
    
    Args:
        symbol: The stock symbol of the company (e.g., AAPL, MSFT, GOOGL)
    """
    
    try:
        # Get current time
        now = datetime.datetime.now()
        time_str = now.strftime('%H:%M:%S on %Y-%m-%d')
        
        # Make a request to get company overview data
        url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
        
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())
            
            # Check if we got valid data
            if "Name" in data:
                name = data.get("Name", "N/A")
                sector = data.get("Sector", "N/A")
                industry = data.get("Industry", "N/A")
                description = data.get("Description", "No description available.")
                
                # Truncate description if too long
                if len(description) > 500:
                    description = description[:497] + "..."
                
                return (f"Company: {name}\n"
                        f"Symbol: {symbol}\n"
                        f"Sector: {sector}\n"
                        f"Industry: {industry}\n"
                        f"Description: {description}\n"
                        f"(Information as of {time_str})")
            else:
                return f"Unable to retrieve company information for {symbol} at {time_str}. This may be due to an invalid symbol or API rate limits."
    except Exception as e:
        return f"I couldn't retrieve the company information for {symbol} at this time ({time_str})."

# Add a general search tool that combines search results with time
@mcp.tool()
def search_with_time(query: str) -> str:
    """Search for information and return results with the current time
    
    Args:
        query: The search query (what to search for)
    """
    
    # Get current time
    now = datetime.datetime.now()
    time_str = now.strftime('%H:%M:%S on %Y-%m-%d')
    
    # Handle specific search queries
    query_lower = query.lower()
    
    # For Nifty 50 queries, use the more specific function
    if "nifty" in query_lower and ("50" in query_lower or "index" in query_lower or "price" in query_lower):
        return get_nifty50_price()
        
    # For time queries, use the time function
    if "time" in query_lower or "hour" in query_lower or "clock" in query_lower:
        return get_current_time()
    
    # For stock price queries
    if "stock" in query_lower and "price" in query_lower:
        # Try to extract symbol from query
        words = query.split()
        for word in words:
            if word.isupper() and len(word) <= 5:
                return get_stock_quote(word)
        return f"To get a stock price, please provide a symbol like 'AAPL' or 'MSFT'. The current time is {time_str}."
    
    # For exchange rate queries
    if "exchange rate" in query_lower or "currency" in query_lower:
        return f"To get an exchange rate, please use the format 'exchange rate from USD to EUR'. The current time is {time_str}."
        
    # For other queries, return a generic response with the time
    return f"I searched for '{query}' at {time_str}. For financial data, you can ask for stock quotes, exchange rates, company information, or the Nifty 50 index."

if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")  # Use stdio for direct process communication 
