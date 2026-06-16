import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI

load_dotenv('/Users/timotheemaurin/trading/.env')

client = AsyncOpenAI(
    base_url=os.getenv('DOUBLEWORD_BASE_URL'),
    api_key=os.getenv('DOUBLEWORD_API_KEY'),
)

model = OpenAIChatModel(
    'nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4',
    provider=OpenAIProvider(openai_client=client),
)

agent = Agent(model=model)

result = agent.run_sync('Say "Pydantic + Doubleword connected" and nothing else.')
print(result.output)
