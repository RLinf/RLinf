# Copyright (c) Microsoft. All rights reserved.

"""This sample code demonstrates how to define a Calc-X agent trainable with Agent-lightning
with latest Agent-lightning API (v0.2+)."""

import asyncio
import os
import re
from typing import TypedDict, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from eval_utils import evaluate

import agentlightning as agl


class MathProblem(TypedDict):
    """This TypedDict defines the structure of each training sample.

    Your task structure should contain all the information needed for:

    - The agent to process the task (e.g., 'question')
    - Evaluation (e.g., 'result' for ground truth)

    This type is optional. Not necessary to make the example work.
    """

    # The fields come from the dataset
    id: str
    question: str  # The math problem for the agent to solve
    chain: str  # Step-by-step solution (not used in training)
    result: str  # Ground truth answer for evaluation
    source: str


def autogen_assistant_agent(
    model: str, openai_base_url: str, temperature: float, workbench: McpWorkbench
) -> AssistantAgent:

    model_client = OpenAIChatCompletionClient(
        model=model,
        base_url=openai_base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "token-abc123"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
        },
        temperature=temperature,
    )

    calc_agent = AssistantAgent(
        name="calc",
        model_client=model_client,
        workbench=workbench,
        reflect_on_tool_use=False,
        max_tool_iterations=2,
    )
    return calc_agent


@agl.rollout
async def calc_agent(task: MathProblem, llm: agl.LLM) -> None:
    """Calc-X agent rollout function.

    It would accept a math problem and a LLM endpoint resource.
    It's expected to return None, and emit reward via `agl.emit_reward`.
    It can also return the reward directly without `agl.emit_reward`.
    You can choose either way, but not both.
    """

    calculator_mcp_server = StdioServerParams(command="mcp-server-calculator", args=[])

    async with McpWorkbench(calculator_mcp_server) as workbench:
        calc_agent = autogen_assistant_agent(
            llm.model,
            llm.endpoint,
            llm.sampling_parameters.get("temperature", 0.7),
            workbench,
        )
        try:
            output_format = "Output the answer when you are ready. The answer should be surrounded by three sharps (`###`), in the form of ### ANSWER: <answer> ###."
            prompt = task["question"] + " " + output_format
            # Sometimes MCP tools can timeout. In that case, the whole agent will block.
            # We thus set a timeout of 5 minutes so that the agent will not block indefinitely.
            result = await asyncio.wait_for(calc_agent.run(task=prompt), timeout=300.0)
            # evaluate
            last_message = cast(str, result.messages[-1].content)  # type: ignore
            answer = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", last_message)
            if answer:
                answer = answer.group(1)
            else:
                answer = last_message
        except asyncio.TimeoutError as e:
            print("Timeout occurred. Error:", str(e))
            answer = "None"
        except Exception as e:
            print("Failure:", str(e))
            answer = "None"
        reward = await evaluate(answer, str(task["result"]))
        agl.emit_reward(reward)  # Emit reward for tracing
        # Disabled verbose logging
        print("answer: {} ground_truth: {} reward: {}".format(answer, task["result"], reward))
