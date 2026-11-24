import pytest_asyncio

from helpers import make_runtime
from voice_agent.harness import SimClient


@pytest_asyncio.fixture
async def rt():
    runtime = make_runtime()
    yield runtime
    await runtime.stop()


@pytest_asyncio.fixture
async def client(rt):
    return await SimClient.open(rt)
