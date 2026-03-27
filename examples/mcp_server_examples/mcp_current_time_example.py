import logging
from mcp.server.fastmcp import FastMCP
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

mcp = FastMCP("CurrentTimeServer")


@mcp.tool()
def get_current_time() -> str:
    """Get the current time in the user's location"""
    
    now = datetime.datetime.now()
    
    return f"The current time is {now.strftime('%H:%M:%S')} on {now.strftime('%Y-%m-%d')}"

if __name__ == "__main__":
    mcp.run(transport="stdio") 

