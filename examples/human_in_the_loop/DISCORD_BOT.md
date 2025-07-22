### Create Discord Bot

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. In the "Bot" tab, create a bot and copy the token (this is your DISCORD_TOKEN)
4. Enable privileged gateway intents in the "Bot" tab:
   - Message Content Intent
5. In the "OAuth2" tab → "URL Generator":
   - Select "bot" scope
   - Under "Bot Permissions", check:
     - Send Messages
     - Create Public Threads
     - Read Message History
   - Copy the generated URL and use it to invite/add the bot to your Discord server

## Finding Discord IDs

### Getting Channel ID
1. Enable Developer Mode in Discord (Settings → Advanced → Developer Mode)
2. Right-click on the target channel → "Copy ID" (this is your DISCORD_CHANNEL_ID)

### Getting User ID
1. Right-click on the target user (yourself or the human responder) → "Copy ID" (this is your DISCORD_USER_ID)