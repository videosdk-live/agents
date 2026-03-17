import os
import asyncio
from dotenv import load_dotenv

import discord
from discord.ext import commands
from mcp.server.fastmcp import FastMCP

class DiscordHuman:
    def __init__(self, user_id: int, channel_id: int):
        self.user_id = user_id
        self.channel_id = channel_id
        self.bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
        self.loop = asyncio.get_event_loop()
        self.response_future = None

        @self.bot.event
        async def on_ready():
            print(f"Logged in as {self.bot.user} (ID: {self.bot.user.id})")

        @self.bot.event
        async def on_message(message: discord.Message):
            if (
                message.author.id == self.user_id and
                self.response_future and not self.response_future.done()
            ):
                self.response_future.set_result(message.content)

    async def start(self, token):
        await self.bot.start(token)

    async def ask(self, question: str) -> str:
        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            raise RuntimeError("Could not find channel with given ID")

        thread = await channel.create_thread(
            name=question[:100],
            type=discord.ChannelType.public_thread
        )
        await thread.send(f"<@{self.user_id}> {question}")

        self.response_future = self.loop.create_future()
        try:
            return await asyncio.wait_for(self.response_future, timeout=600)
        except asyncio.TimeoutError:
            return "⏱️ Timed out waiting for a human response"

load_dotenv()

async def main():
    discord_token = os.getenv("DISCORD_TOKEN")
    user_id = int(os.getenv("DISCORD_USER_ID"))
    channel_id = int(os.getenv("DISCORD_CHANNEL_ID"))

    discord_human = DiscordHuman(user_id, channel_id)

    asyncio.create_task(discord_human.start(discord_token))

    mcp = FastMCP("HumanInTheLoopServer")

    @mcp.tool(description="Ask a human agent via Discord for a specific user query such as discount percentage, etc.")
    async def ask_human(question: str) -> str:
        return await discord_human.ask(question)

    await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())