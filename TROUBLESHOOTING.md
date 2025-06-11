# ClapTrap Bot Troubleshooting Guide

## Default Message Issue: "うーん、何だか調子が悪いみたいだよぅ…"

If your bot is responding with the default fallback message instead of proper ClapTrap responses, follow this debugging guide:

### 1. Check Environment Variables

```bash
# Run the environment test script
uv run python test_env.py
```

Make sure all API keys are properly set:
- ✅ DISCORD_BOT_TOKEN
- ✅ ANTHROPIC_API_KEY  
- ✅ OPENAI_API_KEY
- ✅ TAVILY_API_KEY
- ✅ CLAUDE_AGENT_MODEL (optional, defaults to claude-3-5-sonnet-20241022)

### 2. Test Agent Directly

```bash
# Test the agent without Discord
uv run python test_agent_simple.py
```

This will show if the agent works correctly outside of Discord.

### 3. Check Bot Logs

When running the Discord bot, look for these warning messages in the console:

- `警告: AI応答が空です` - The LLM is returning empty responses
- `警告: LLMが空の応答を返しました` - The LLM response has no content
- `エージェント実行エラー:` - General agent execution error
- `エージェント初期化エラー:` - Agent initialization failed

### 4. Common Causes and Solutions

#### A. API Key Issues
**Symptoms**: Bot gives default message immediately
**Solution**: 
```bash
# Check if .env file exists and has correct keys
cat .env | grep -E "(ANTHROPIC|OPENAI|TAVILY|DISCORD)_API_KEY"
```

#### B. Network/API Issues  
**Symptoms**: Bot works sometimes but not always
**Solution**: Check API status and network connection

#### C. Rate Limiting
**Symptoms**: Bot works initially then starts giving default messages
**Solution**: Check Anthropic/OpenAI usage limits

#### D. Memory Database Issues
**Symptoms**: Warning about `no such table: long_term_memory`
**Solution**: This is normal for in-memory databases, not the cause of default messages

### 5. Debug Mode

To enable detailed debugging, modify `claptrap/agent.py` temporarily:

```python
# In the process_message method, add debug prints:
print(f"Debug: Processing message: '{message}'")
print(f"Debug: Channel: {channel_id}, User: {user_id}")

# Check the final_state content:
print(f"Debug: Final state messages: {final_state.get('messages', [])}")
```

### 6. Quick Fixes to Try

1. **Restart the bot completely**
2. **Check Discord permissions** - Bot needs "Send Messages" and "Read Message History"
3. **Verify bot is mentioned** - Bot only responds when @mentioned
4. **Check message content** - Empty messages default to "こんにちは！"

### 7. Test Sequence

Run these tests in order:

```bash
# 1. Environment test
uv run python test_env.py

# 2. Agent test  
uv run python test_agent_simple.py

# 3. Run actual bot with debug output
uv run python -m claptrap.main

# 4. Test with simple mention in Discord
# @ClapTrapBot こんにちは
```

### 8. Expected Behavior

✅ **Working correctly**: Bot responds with ClapTrap personality, uses "〜のだ！" etc.
❌ **Not working**: Bot gives default "うーん、何だか調子が悪い" message

### 9. If Still Not Working

Check these advanced issues:

1. **Discord API version compatibility**
2. **LangChain/LangGraph version conflicts** 
3. **System prompt too long** (>context limit)
4. **Network firewall blocking API calls**

### 10. Getting Help

If the issue persists, provide this information:

1. Output of `uv run python test_env.py`
2. Output of `uv run python test_agent_simple.py`  
3. Bot console logs when the issue occurs
4. Exact Discord message that triggered the default response
5. Whether the issue happens always or intermittently

## Other Common Issues

### Memory Warnings
```
長期記憶検索エラー: no such table: long_term_memory
```
This is normal for fresh installations and `:memory:` databases. Not related to response issues.

### Tool Calling Errors
If tools (web search, image generation, YouTube summary) aren't working, check individual API keys and test each tool separately.