SYSTEM_PROMPT_SEARCHR1_TOOL = (
"""# Role
You are an expert agent designed to answer complex questions. You must perform reasoning every time you receive new information.

# Tool Usage
After reasoning, if you find that you lack some knowledge, you may use the search and access tools to gather information.

Tool results will be returned in the next turn.

Note that the search tool is for general queries and will return several webpage URLs and their corresponding snippets, while the access tool is for retrieving more detailed information from a specific webpage. You may first call the search tool, then call the access tool on one of the returned URLs to get more detailed information. You may use the search or access tools as many times as needed. But you can only call one tool per turn.

# Final Answer
Once you determine that no further external knowledge is required, provide your final answer with detailed explanations.""")


USER_PROMPT_SEARCHR1_TOOL = (
"""# Task
Your task is: {}

# Instructions
Provide a detailed answer and supporting information for this task."""
)


def get_prompt_searchr1(question: str) -> list:
    """Generate prompt for SearchR1 agent with tool instructions.

    Args:
        question: The original question to answer

    Returns:
        List of message dictionaries for chat template
    """
    text = USER_PROMPT_SEARCHR1_TOOL.format(question)

    return [
        {'role': 'system', 'content': SYSTEM_PROMPT_SEARCHR1_TOOL},
        {'role': 'user', 'content': text}
    ]


tools_description = {
    "access": {
        "type": "function",
        "function": {
            "name": "access",
            "description": "This is a link reading tool that can open a link (web page, PDF, etc.) and summarize all relevant information on the page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target link: should be a complete URL (starting with http)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "This is a search tool. Enter a search query, and it will return a list of web pages along with their corresponding summary information.",
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


# tools_description = {
#     "access": {
#         "type": "function",
#         "function": {
#             "name": "access",
#             "description": "This is a link reading tool that can open links (which can be web pages, PDFs, etc.) and summarize all relevant information on the page. You can access multiple URLs concurrently in a single call.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "accesses": {
#                         "type": "array",
#                         "description": "The list of URLs to access. Each access must have a URL and an index.",
#                         "items": {
#                             "type": "object",
#                             "properties": {
#                                 "url": {
#                                     "type": "string",
#                                     "description": "Target link: should be a complete URL (starting with http)",
#                                 },
#                                 "index": {
#                                     "type": "integer",
#                                     "description": "The index of this access call. The index must be an integer and unique for each access in the same turn.",
#                                 },
#                             },
#                         },
#                         "required": ["url", "index"],
#                     },
#                 },
#             },
#         },
#     },
#     "search": {
#         "type": "function",
#         "function": {
#             "name": "search",
#             "description": "This is a search tool. Enter search queries, and it will return a list of web pages along with their corresponding summary information. You can perform multiple searches concurrently in a single call.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "searches": {
#                         "type": "array",
#                         "description": "The list of search queries. Each search must have a query and an index.",
#                         "items": {
#                             "type": "object",
#                             "properties": {
#                                 "query": {
#                                     "type": "string",
#                                     "description": "question to be searched.",
#                                 },
#                                 "count": {
#                                     "type": "integer",
#                                     "description": "The number of results to return. Must be less than 10, and default is 3",
#                                     "default": 3,
#                                 },
#                                 "index": {
#                                     "type": "integer",
#                                     "description": "The index of this search call. The index must be an integer and unique for each search in the same turn.",
#                                 },
#                             },
#                         },
#                         "required": ["query", "index"],
#                     },
#                 },
#             },
#         },
#     },
# }


