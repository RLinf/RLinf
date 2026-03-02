# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

tools_description_en = {
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
                            "required": ["prompt"],
                        },
                    },
                },
                "required": ["sub_agents"],
            },
        },
    },
    "access": {
        "type": "function",
        "function": {
            "name": "access",
            "description": "This is a link-reading tool that opens webpages and retrieves information from them based on your intent. You may access multiple URLs simultaneously in a single call, but you are limited to a maximum of five tool instances per call.",
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
                                    "description": "Target link: should be a complete URL. Remember to only use the URLs provided by the search tool",
                                },
                                "info_to_extract": {
                                    "type": "string",
                                    "description": "The specific question or information to extract from this URL",
                                },
                            },
                            "required": ["url", "info_to_extract"],
                        },
                    },
                },
                "required": ["urls"],
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
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                "required": ["queries"],
            },
        },
    },
    "access_single_agent": {
        "type": "function",
        "function": {
            "name": "access",
            "description": "This is a link-reading tool that opens webpages and retrieves information from them based on your intent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Target link: should be a complete URL. Remember to only use the URLs provided by the search tool",
                    },
                    "info_to_extract": {
                        "type": "string",
                        "description": "The specific question or information to extract from the URL",
                    },
                },
                "required": ["url", "info_to_extract"],
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

tools_description_zh = {
    "create_sub_agents": {
        "type": "function",
        "function": {
            "name": "create_sub_agents",
            "description": "创建可根据输入提示执行特定任务的子代理。你可以在一次调用中并发创建多个子代理，但每次调用最多只能创建十个子代理。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sub_agents": {
                        "type": "array",
                        "description": "要创建的子代理列表。每个子代理会并行创建并执行；它们之间没有顺序或先后关系。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "该子代理需要完成的子任务的具体细节。",
                                }
                            },
                            "required": ["prompt"],
                        },
                    },
                },
                "required": ["sub_agents"],
            },
        },
    },
    "access": {
        "type": "function",
        "function": {
            "name": "access",
            "description": "这是一个链接读取工具，会根据你的意图打开网页并从中获取信息。你可以在一次调用中同时访问多个 URL，但每次调用最多只能使用五个工具实例。",
            "parameters": {
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "description": "要访问的 URL 列表。每个 access 工具会并行创建并执行；它们之间没有顺序或先后关系。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "目标链接：应为完整的 URL。请记得只使用搜索工具提供的 URL。",
                                },
                                "info_to_extract": {
                                    "type": "string",
                                    "description": "需要从该 URL 中提取的具体问题或信息。",
                                },
                            },
                            "required": ["url", "info_to_extract"],
                        },
                    },
                },
                "required": ["urls"],
            },
        },
    },
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "这是一个搜索工具。输入搜索查询后，它会返回网页列表及其对应的摘要信息。你可以在一次调用中同时发起多个查询，但每次调用最多只能使用五个工具实例。",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "description": "搜索查询列表。每个 search 工具会并行创建并执行；它们之间没有顺序或先后关系。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "要搜索的问题。",
                                },
                                "count": {
                                    "type": "integer",
                                    "description": "要返回的结果数量。必须小于 10，默认值为 3。",
                                    "default": 3,
                                },
                            },
                            "required": ["query"],
                        },
                    },
                },
                "required": ["queries"],
            },
        },
    },
    "access_single_agent": {
        "type": "function",
        "function": {
            "name": "access",
            "description": "这是一个链接读取工具，会根据你的意图打开网页并从中获取信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "目标链接：应为完整的 URL。请记得只使用搜索工具提供的 URL。",
                    },
                    "info_to_extract": {
                        "type": "string",
                        "description": "需要从该 URL 中提取的具体问题或信息。",
                    },
                },
                "required": ["url", "info_to_extract"],
            },
        },
    },
    "search_single_agent": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "这是一个搜索工具。输入搜索查询后，它会返回网页列表及其对应的摘要信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的问题。",
                    },
                    "count": {
                        "type": "integer",
                        "description": "要返回的结果数量。必须小于 10，默认值为 3。",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
}
