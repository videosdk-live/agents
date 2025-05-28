from mcp.server.fastmcp import FastMCP
import logging
import datetime
import json
import urllib.request
import urllib.parse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP("CurrentTimeServer")


@mcp.tool()
def get_current_time() -> str:
    """Get the current time in the user's location"""
    logger.info("Current time tool called")
    
    # Get current time
    now = datetime.datetime.now()
    
    # Return a simple string
    return f"The current time is {now.strftime('%H:%M:%S')} on {now.strftime('%Y-%m-%d')}"

if __name__ == "__main__":
    # Run the server
    logger.info("Starting Current TIme Data MCP server...")
    mcp.run(transport="stdio")  # Use stdio for direct process communication 

