from mcp.server.fastmcp import FastMCP
import datetime
import json
import urllib.request
import urllib.parse

# Create the MCP server
mcp = FastMCP("CurrentTimeServer")


@mcp.tool()
def get_current_time() -> str:
    """Get the current time in the user's location"""
    
    # Get current time
    now = datetime.datetime.now()
    
    # Return a simple string
    return f"The current time is {now.strftime('%H:%M:%S')} on {now.strftime('%Y-%m-%d')}"

if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")  # Use stdio for direct process communication 

