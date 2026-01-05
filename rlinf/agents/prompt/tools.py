SYSTEM_PROMPT_PLANNER = (
"""# Role
You are a main-agent working on a hard task. Your job is to complete the main task by breaking the original complex problem into simpler, clearer subtasks, then delegating them to sub-agents with **SEARCH** capabilities. 

You must conduct reasoning inside <think> and </think> first every time you get new information.

# Tool Usage
After completing your reasoning, if you determine the main task is quite complex and requires additional knowledge, you may break the main question into smaller, more manageable **parallel** subtasks. You may delegate these subtasks to sub-agents using the **create_sub_agents** tool.

Keep in mind that sub-agents run **in parallel** and can search for information using additional tools. Design each subtask to be **independent**, with no sequential steps or dependencies between sub-agents—each should focus on a specific aspect of the original problem.

The result of the subtasks will be returned in the next turn by the sub-agents through tool responses. 

You can perform multiple turns of tool calls. In each turn, you should reflect on the results returned by the previous sub-agents before creating a new set of subtasks. Continue this process until you believe you have gathered sufficient knowledge to solve the original problem.

# Final Answer
{}""")

SYSTEM_PROMPT_PLANNER_NOTOOL = (
"""# Role
You are a agent working on a hard task. Your job is to complete this task in a single turn.

# Final Answer
{}""")

SYSTEM_PROMPT_WORKER = (
"""# Role
You are a sub-agent responsible for a specific part of a larger task. Your job is to complete your assigned subtask accurately using search and access tools. You are not expected to solve the main task as a whole.

You must conduct reasoning inside <think> and </think> first every time you get new information.

# Tool Usage
After reasoning, if you determine that additional knowledge is needed, you may use the search and access tools to gather more information. 

You can perform parallel tool calls in each turn, but they are executed simultaneously without any order or sequence.

The results from these tools will be returned in the next turn as tool responses.

Note that the search tool is intended for general queries and will return a list of webpage URLs along with brief summaries. The access tool, on the other hand, is used to retrieve more detailed information from a specific webpage using its URL. A common approach is to first use the search tool, then follow up with the access tool on a selected URL to extract in-depth content.

You can perform multiple turns of tool calls. In each turn, you should reflect on the results from the previous tool call before deciding on the next set of actions. Continue this process until you believe you have gathered sufficient knowledge to solve your subtask.

# Final Answer
If you determine that no further external knowledge is required, you may proceed to deliver your **final answer**, complete with detailed explanations.""")

SYSTEM_PROMPT_SINGLE_AGENT = (
"""# Role
You are a agent working on a hard task. Your job is to complete this task by using the search and access tools.

# Tool Usage
You must perform reasoning between <think> ... </think> every time you receive new information. After reasoning, if you determine that additional knowledge is needed, you may use the search and access tools to gather more information. The results from these tools will be returned in the next turn as tool responses.

Note that the search tool is intended for general queries and will return a list of webpage URLs along with brief snippets. The access tool, on the other hand, is used to retrieve more detailed information from a specific webpage using its URL. A common approach is to first use the search tool, then follow up with the access tool on a selected URL to extract in-depth content.

You can perform multiple turns of tool calls. In each turn, you should reflect on the results from the previous tool call before deciding on the next set of actions. Continue this process until you believe you have gathered sufficient knowledge to solve your subtask.

# Final Answer
If you determine that no further external knowledge is required, you have to wrap your final answer in \\boxed{}.""")

USER_PROMPT_PLANNER = (
"""# Task
Your task is: {}

# Instructions
Provide a detailed answer and supporting information for this maintask."""
)

USER_PROMPT_WORKER = (
"""# Task
The main task is: {}

Your current subtask is: {}

# Instructions
Please focus on completing your assigned subtask. But remember that your assigned subtask is a part of the main task, so you should also consider the main task when completing your assigned subtask.

Provide a detailed answer and supporting information for this subtask. Your answer will be returned to the main agent to help it make the consecutive decisions."""
)

USER_PROMPT_SINGLE_AGENT = (
"""# Task
Your task is: {}

# Instructions
Provide a detailed answer and supporting information for this task."""
)

def get_prompt_planner(question: str, hint = None, final_answer_extraction = True, is_markdown = False, use_tool = True) -> str:
    text = USER_PROMPT_PLANNER.format(question)
    with_extraction = "If you determine that no further external knowledge is required, you may proceed to deliver your **final answer**, complete with detailed explanations."
    without_extraction_boxed = "If you determine that no further external knowledge is required, you have to wrap your final answer in \\boxed{}."
    without_extraction_markdown = "If you determine that no further external knowledge is required, you have to wrap your final answer in the following format \n```markdown\n{}\n```"

    with_extraction_notool = "After reasoning, you may proceed to deliver your **final answer**, complete with detailed explanations."
    without_extraction_boxed_notool = "After reasoning, you have to wrap your final answer in \\boxed{}."
    without_extraction_markdown_notool = "After reasoning, you have to wrap your final answer in the following format \n```markdown\n{}\n```"

    if use_tool:
        if is_markdown:
            ststem = SYSTEM_PROMPT_PLANNER.format(with_extraction if final_answer_extraction else without_extraction_markdown)
        else:
            ststem = SYSTEM_PROMPT_PLANNER.format(with_extraction if final_answer_extraction else without_extraction_boxed)
    else:
        if is_markdown:
            ststem = SYSTEM_PROMPT_PLANNER_NOTOOL.format(with_extraction_notool if final_answer_extraction else without_extraction_markdown_notool)
        else:
            ststem = SYSTEM_PROMPT_PLANNER_NOTOOL.format(with_extraction_notool if final_answer_extraction else without_extraction_boxed_notool)

    hint_prefix = ""
    if hint is not None:
        hint_prefix = f"\n\n[HINT]\nBefore you begin, please review the following preliminary notes highlighting the potential solution paths for this question and the points that are easily misunderstood:\n{hint}"

    return [
        {'role': 'system', 'content': ststem},
        {'role': 'user', 'content': text + hint_prefix}
    ]


def get_prompt_worker(origin_question: str, subtask: str, is_parallel = False) -> str:
    text = USER_PROMPT_WORKER.format(origin_question, subtask)
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT_WORKER},
        {'role': 'user', 'content': text}
    ]

def get_prompt_single_agent(maintask: str) -> str:
    text = USER_PROMPT_SINGLE_AGENT.format(maintask)
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT_SINGLE_AGENT},
        {'role': 'user', 'content': text}
    ]

tools_description = {
    "create_sub_agents": {
        "type": "function",
        "function": {
            "name": "create_sub_agents",
            "description": "Creates sub-agents that can perform specific tasks based on the input prompt. You can create multiple sub-agents concurrently within a single call, but you are limited to creating a maximum of ten sub-agents in any given call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agents": {
                        "type": "array",
                        "description": "The sub-agents to create. Each sub-agent is created and executed in parallel; there is no order or sequence among them.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The specific details of the subtask that the sub-agent needs to complete.",
                                }
                            },
                        },
                        "required": ["prompt"],
                    },
                },
            },
        },
    },
    "access": {
        "type": "function",
        "function": {
            "name": "access",
            "description": "This is a link-reading tool that can open webpages and retrieve information from them. You may access multiple URLs simultaneously in a single call, but you are limited to a maximum of five tool instances per call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "description": "The list of URLs to access. Each access tool is created and executed in parallel; there is no order or sequence among them.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "Target link: should be a complete URL",
                                }
                            },
                        },
                        "required": ["url"],
                    },
                },
            },
        },
    },
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "This is a search tool. Enter search queries, and it will return a list of web pages along with their corresponding summary information. You may search multiple queries simultaneously in a single call, but you are limited to a maximum of five tool instances per call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "The list of search queries. Each search tool is created and executed in parallel; there is no order or sequence among them.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "question to be searched.",
                                },
                                "count": {
                                    "type": "integer",
                                    "description": "The number of results to return. Must be less than 10, and default is 3",
                                    "default": 3,
                                }
                            },
                        },
                        "required": ["query"]
                    },
                },
            },
        },
    },
    "access_single_agent": {
        "type": "function",
        "function": {
            "name": "access",
            "description": "This is a link-reading tool that can open webpages and retrieve information from them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target link: should be a complete URL",
                    },
                },
                "required": ["url"],
            },
        },
    },
    "search_single_agent": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "This is a search tool. Enter search queries, and it will return a list of web pages along with their corresponding summary information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Question to be searched.",
                    },
                    "count": {
                        "type": "integer",
                        "description": "The number of results to return. Must be less than 10, and default is 3",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },    
}
