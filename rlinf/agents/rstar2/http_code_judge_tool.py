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

import base64
from typing import Any, Callable, Union, Optional, Dict

from rlinf.data.tool_call.tool_io_struct import ToolChannelRequest, ToolChannelResponse


class ToolBase:
    name = None

    def __init__(self, cfg):
        self.cfg = cfg

    async def execute(self, request: ToolChannelRequest, **kwargs) -> ToolChannelResponse:
        """Execute the tool call."""
        raise NotImplementedError()

    def tool_schema(self) -> Dict:
        """
        A JSON Schema, giving the name, description and argument types for the tool.
        Ref: https://huggingface.co/docs/transformers/en/chat_extras#json-schemas
        """
        raise NotImplementedError()

    def validate(self, request: ToolChannelRequest) -> Optional[ToolChannelResponse]:
        """Validate the request call the right tool and the schema is right."""
        raise NotImplementedError()


class CodeJudgeToolBase(ToolBase):
    name = None

    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.url = f"http://{self.cfg.tools.codejudge.host_addr}:{self.cfg.tools.codejudge.host_port}/run/long-batch"

    def _postprocess(self, result):
        if result["run_success"] and result["success"]:
            output_parts = []
            output_parts.append("Tool call success")
            if result["stdout"]:
                output_parts.append(f"stdout: {result['stdout']}")
            if result["stderr"]:
                output_parts.append(f"stderr: {result['stderr']}")
            output_parts.append(f"execution time: {result['cost']:.2f}s")
            result = "\n".join(output_parts)
            return ToolChannelResponse(success=True, result=result)
        else:
            output_parts = []
            output_parts.append("Tool call failure")
            output_parts.append(f"reason: {result['reason']}")
            if result["stdout"]:
                output_parts.append(f"stdout: {result['stdout']}")
            if result["stderr"]:
                output_parts.append(f"stderr: {result['stderr']}")
            output_parts.append(f"execution time: {result['cost']:.2f}s")
            result = "\n".join(output_parts)
            return ToolChannelResponse(success=False, result=result)


code_template_setup = '''
import os
import base64
import sys
import ast
import traceback
from typing import Optional, Any
import linecache
from types import CodeType
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

class CodeExecutionError(Exception):
    """Custom exception for code execution errors with line information"""
    def __init__(self, original_error: Exception, code: str, line_offset: int = 0):
        self.original_error = original_error
        self.code = code
        self.line_offset = line_offset
        
        # Get error line number
        if hasattr(original_error, 'lineno'):
            self.lineno = original_error.lineno
        else:
            tb = getattr(original_error, '__traceback__', None)
            if tb:
                while tb.tb_next:
                    tb = tb.tb_next
                self.lineno = tb.tb_lineno
            else:
                self.lineno = -1
        
        # Adjust line number for code segment
        if self.lineno != -1:
            self.lineno += line_offset
        
        # Format error message
        error_type = type(original_error).__name__
        error_msg = str(original_error)
        
        if self.lineno != -1:
            # Get the problematic line
            lines = code.splitlines()
            if 0 <= self.lineno - 1 < len(lines):
                error_line = lines[self.lineno - 1]
                # Create error message with line information
                super().__init__(f"{error_type} at line {self.lineno}: {error_msg}\\n  {error_line}")
                return
        
        super().__init__(f"{error_type}: {error_msg}")

class PersistentExecutor:
    def __init__(self):
        self.exec_globals = {
            '__name__': '__main__',
            '__file__': '<string>',
            '__builtins__': __builtins__
        }

    def split_code(self, code: str) -> tuple[str, Optional[str]]:
        """
        Intelligently split code into main body and last expression
        
        Args:
            code: The source code string
            
        Returns:
            tuple[str, Optional[str]]: (main code body, last expression if exists)
        """
        try:
            # Parse code into AST
            tree = ast.parse(code)
            if not tree.body:
                return code, None
            
            # Check if the last node is a pure expression (not a call)
            last_node = tree.body[-1]
            if isinstance(last_node, ast.Expr):
                # Get the line range of the last expression
                last_expr_start = last_node.lineno
                last_expr_end = last_node.end_lineno if hasattr(last_node, 'end_lineno') else last_node.lineno
                
                # Split the code
                lines = code.splitlines()
                main_code = '\\n'.join(lines[:last_expr_start-1])
                last_expr = '\\n'.join(lines[last_expr_start-1:last_expr_end])
                return main_code, last_expr
        except SyntaxError as e:
            raise CodeExecutionError(e, code)
        return code, None

    def execute_code(self, code: str, replay_history_code: bool) -> None:
        """
        Execute code while maintaining persistent environment state.
        If the last line is an expression, its value will be printed to stdout.
        
        Args:
            code: The source code string to execute
            replay_history_code: If True, suppress stdout and stderr output
        """
        try:
            # Split code intelligently
            main_code, last_expr = self.split_code(code)
            
            # Set up output redirection if replay_history_code is True
            if replay_history_code:
                stdout_capture = StringIO()
                stderr_capture = StringIO()
                stdout_context = redirect_stdout(stdout_capture)
                stderr_context = redirect_stderr(stderr_capture)
            else:
                stdout_context = redirect_stdout(sys.stdout)
                stderr_context = redirect_stderr(sys.stderr)
            
            # Execute main code body
            if main_code:
                try:
                    # Compile code to get better error line numbers
                    compiled_code = compile(main_code, '<string>', 'exec')
                    with stdout_context, stderr_context:
                        exec(compiled_code, self.exec_globals)
                except Exception as e:
                    raise CodeExecutionError(e, main_code)
            
            # If there's a last expression, try to evaluate and print it
            if last_expr:
                try:
                    # Compile expression to get better error line numbers
                    compiled_expr = compile(last_expr, '<string>', 'eval')
                    with stdout_context, stderr_context:
                        last_value = eval(compiled_expr, self.exec_globals)
                    
                    # Only print the result if not in replay mode
                    if last_value is not None and not replay_history_code:
                        print(repr(last_value), file=sys.stdout)
                except Exception as e:
                    # Try executing as statement if evaluation fails
                    try:
                        compiled_stmt = compile(last_expr, '<string>', 'exec')
                        with stdout_context, stderr_context:
                            exec(compiled_stmt, self.exec_globals)
                    except Exception as e:
                        # Calculate line offset for the last expression
                        line_offset = len(main_code.splitlines()) if main_code else 0
                        raise CodeExecutionError(e, last_expr, line_offset)
                    
        except Exception as e:
            if replay_history_code:
                return
            if isinstance(e, CodeExecutionError):
                print(str(e), file=sys.stderr)
            else:
                traceback.print_exc(file=sys.stderr)
            os._exit(1)
            return

persistent_executor = PersistentExecutor()
'''


code_template_exec = """
code_to_execute = base64.b64decode("{}".encode()).decode()
persistent_executor.execute_code(code_to_execute, replay_history_code={})
"""

class PythonTool(CodeJudgeToolBase):
    name = "execute_python_code_with_standard_io"

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    async def execute(self, request: ToolChannelRequest, send_request_func: Callable[[str, Dict], ToolChannelResponse]):
        err_msg = self.validate(request)
        if err_msg:
            return err_msg

        # convert the code to the code exec on code-judge
        code_to_execute = base64.b64encode(request.tool_args["code"].encode()).decode()
        final_code = code_template_setup
        # TODO: add history code here
        final_code += code_template_exec.format(code_to_execute, "False")

        submission = {
            "type": "python",
            "solution": final_code,
            "input": request.tool_args["input"],
        }

        data = {
            "type": "batch",
            "submissions": [submission]
        }

        try:
            results = (await send_request_func(self.url, data))["results"]
            assert len(results) == 1, f"{results}"
            return self._postprocess(results[0])
        except Exception as e:
            return ToolChannelResponse(success=False, result=f"Error: send request failed: {str(e)}")

    def tool_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "execute_python_code_with_standard_io", 
                "description": "Execute Python code with standard input and capture standard output. This function takes a Python code string and an input string, provides the input string through standard input (stdin) to the code, and captures and returns any output produced through standard output (stdout). If the executed code raises an exception, the error message will be captured and returned instead.", 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string", 
                            "description": "A string containing Python code to be executed. The code can read from standard input using the input() function."
                        }, 
                        "input": {
                            "type": "string", 
                            "description": "A string that will be provided as standard input to the code when it calls input()."
                        }
                    }, 
                    "required": ["code", "input"],
                },
                "return": {
                    "type": "string", 
                    "description": "str: The output produced by the executed code through standard output.",
                },
            }
        }

    def validate(self, request) -> Optional[ToolChannelResponse]:
        tool_args = request.tool_args

        assert request.tool_name == self.name, f"Name mismatch, {self.name} != {request.tool_name}"

        if not isinstance(tool_args, dict):
            return ToolChannelResponse(
                success=False,
                result=f"parameters format error, expect a json format, but get {type(tool_args)}\n"
            )
    
        required_param_msg = ""
        for required_name in self.tool_schema()["function"]["parameters"]["required"]:
            if required_name not in tool_args:
                required_param_msg += f"parameters format error, '{required_name}' is a required parameter but not found\n"
        if required_param_msg:
            return ToolChannelResponse(
                success=False,
                result=required_param_msg
            )

        if not isinstance(tool_args["code"], str):
            return ToolChannelResponse(
                success=False,
                result=f"parameters format error, 'code' should be a string but get {type(tool_args['code'])}"
            )
        
        if not isinstance(tool_args["input"], str):
            return ToolChannelResponse(
                success=False,
                result=f"parameters format error, 'input' should be a string but get {type(tool_args['input'])}"
            )


class JupyterTool(CodeJudgeToolBase):
    name = "execute_jupyter_code"

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

    async def execute(self, request: ToolChannelRequest, send_request_func: Callable[[str, Dict], ToolChannelResponse]):
        err_msg = self.validate(request)
        if err_msg:
            return err_msg

        # convert the code to the code exec on code-judge
        code_to_execute = base64.b64encode(request.tool_args["code"].encode()).decode()
        final_code = code_template_setup
        # TODO: add history code here
        final_code += code_template_exec.format(code_to_execute, "False")

        submission = {
            "type": "python",
            "solution": final_code,
        }

        data = {
            "type": "batch",
            "submissions": [submission]
        }

        try:
            results = (await send_request_func(self.url, data))["results"]
            assert len(results) == 1, f"{results}"
            return self._postprocess(results[0])
        except Exception as e:
            return ToolChannelResponse(success=False, result=f"Error: send request failed: {str(e)}")

    def tool_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Execute python code in a Jupyter notebook cell and return result.",
                "parameters":{
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description":  "The python code to execute in a single cell."
                        }
                    },
                    "required": ["code"],
                }
            }
        }

    def validate(self, request) -> Optional[ToolChannelResponse]:
        tool_args = request.tool_args

        assert request.tool_name == self.name, f"Name mismatch, {self.name} != {request.tool_name}"

        if not isinstance(tool_args, dict):
            return ToolChannelResponse(
                success=False,
                result=f"parameters format error, expect a json format, but get {type(tool_args)}\n"
            )
    
        required_param_msg = ""
        for required_name in self.tool_schema()["function"]["parameters"]["required"]:
            if required_name not in tool_args:
                required_param_msg += f"parameters format error, '{required_name}' is a required parameter but not found\n"
        if required_param_msg:
            return ToolChannelResponse(
                success=False,
                result=required_param_msg
            )

        if not isinstance(tool_args["code"], str):
            return ToolChannelResponse(
                success=False,
                result=f"parameters format error, 'code' should be a string but get {type(tool_args['code'])}"
            )


class LeanTool(CodeJudgeToolBase):
    name = "execute_lean_code"

    async def execute(self, request: ToolChannelRequest, send_request_func: Callable[[str, Dict], ToolChannelResponse]):
        err_msg = self.validate(request)
        if err_msg:
            return err_msg

        code_to_execute = request.tool_args["code"]

        submission = {
            "type": "lean",
            "solution": code_to_execute,
        }

        data = {
            "type": "batch",
            "submissions": [submission]
        }

        try:
            results = (await send_request_func(self.url, data))["results"]
            assert len(results) == 1, f"{results}"
            return self._postprocess(results[0])
        except Exception as e:
            return ToolChannelResponse(success=False, result=f"Error: send request failed: {str(e)}")

    def tool_schema(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Executes a snippet of Lean code and returns the output or errors.",
                "parameters":{
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description":  "The Lean code snippet to be executed."
                        }
                    },
                    "required": ["code"],
                }
            }
        }
        

    def validate(self, request) -> Optional[ToolChannelResponse]:
        tool_args = request.tool_args

        assert request.tool_name == self.name, f"Name mismatch, {self.name} != {request.tool_name}"

        if not isinstance(tool_args, dict):
            return ToolChannelResponse(
                success=False,
                result=f"parameters format error, expect a json format, but get {type(tool_args)}\n"
            )
    
        required_param_msg = ""
        for required_name in self.tool_schema()["function"]["parameters"]["required"]:
            if required_name not in tool_args:
                required_param_msg += f"parameters format error, '{required_name}' is a required parameter but not found\n"
        if required_param_msg:
            return ToolChannelResponse(
                success=False,
                result=required_param_msg
            )

        if not isinstance(tool_args["code"], str):
            return ToolChannelResponse(
                success=False,
                result=f"parameters format error, 'code' should be a string but get {type(tool_args['code'])}"
            )