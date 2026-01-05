import asyncio
from rlinf.workers.agent.agent_loop import SGLangClient

async def main():
    client = SGLangClient(
        llm_ip="172.27.155.18",
        llm_port=30000,  # 建议用 int
        llm_type="Qwen3-30B-A3B-Instruct-2507"
    )

    messages = [
        {"role": "system", "content": "What's your name"},
        {"role": "user", "content": "What's your name"}
    ]

    result_text = await client.call_sglang_api(messages)
    print(f"\n\nsglang response: {result_text}\n\n")

if __name__ == "__main__":
    asyncio.run(main())
