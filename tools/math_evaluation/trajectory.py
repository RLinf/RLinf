import re

"""
trajcectory:
[
    {"role": "rationale", "content": "..."},
    {"role": "program", "content": "..."},
    {"role": "output", "content": "..."},
    {"role": "rationale", "content": "..."},
    ...
]
"""


def text_to_trajectory(traj_str: str) -> None:
    """ """
    # parse the above interleaved string of raionale, program, output, raionale, program, output, ...
    # output a list of dict
    trajectory = []
    cur_role = "rationale"
    cur_content = ""

    # print(traj_str)
    for i, line in enumerate(traj_str.split("\n")):
        if line == "```python":  # program begin
            assert cur_role == "rationale"
            if cur_content:
                trajectory.append({"role": cur_role, "content": cur_content})
                cur_content = ""
            cur_role = "program"
        elif cur_role == "program" and line == "```":  # program end
            assert cur_content
            trajectory.append({"role": cur_role, "content": cur_content})
            cur_content = ""
            cur_role = "output"
        elif cur_role == "output" and line.startswith("```output"):  # output begin
            assert cur_content == ""
        elif cur_role == "output" and line == "```":  # output end
            trajectory.append({"role": cur_role, "content": cur_content})
            cur_content = ""
            cur_role = "rationale"
        else:  # content
            cur_content += line
            if i < len(traj_str.split("\n")) - 1:
                cur_content += "\n"
    # the last content
    if cur_content:
        trajectory.append({"role": cur_role, "content": cur_content})
    return trajectory


def trajectory_to_text(trajectory: list) -> str:
    text = ""
    for item in trajectory:
        content = item["content"]
        if item["role"] == "program":
            content = f"```python\n{content}```\n"
        elif item["role"] == "output":
            content = f"```output\n{content}```\n"
        text += content
    return text


def is_execution_success(output):
    error_key_words = [
        "error",
        "exception",
        "no algorithms",
        "no algorithms",
        "cannot",
        "nan",
        "...",
    ]
    success = all(k not in output.lower() for k in error_key_words)
    return success


def extract_program(text=None, trajectory=None, last_only=False):
    assert text is not None or trajectory is not None, (
        "Either text or trajectory should be provided."
    )
    if trajectory is None:
        try:
            trajectory = text_to_trajectory(text)
        except:  # noqa: E722
            return "raise ValueError('Invalid trajectory')"

    program_list = []
    import_lines = []
    for i, item in enumerate(trajectory):
        if item["role"] == "program":
            cur_program = item["content"]
            if i < len(trajectory) - 1:
                assert trajectory[i + 1]["role"] == "output"
                output = trajectory[i + 1]["content"].strip()
                if is_execution_success(output):
                    program_list.append(cur_program)
                else:
                    # extract import lines only
                    for line in cur_program.split("\n"):
                        if line.startswith("import") or line.startswith("from"):
                            import_lines.append(line)
            else:
                program_list.append(cur_program)
    # add import lines to the first program
    if len(program_list) == 0:
        program_list.append("")
    if len(import_lines) > 0:
        program_list[0] = "\n".join(import_lines) + "\n" + program_list[0]
    for i, program in enumerate(program_list[:-1]):
        program_list[i] = "\n".join(
            [
                line
                for line in program.split("\n")
                if not line.strip().startswith("print(")
            ]
        )

    if last_only:
        program = program_list[-1]
    else:
        program = "\n".join(program_list)
    return program


def extract_program_output(pred_str, last_only=True):
    """
    extract output between ```output\n...\n```, use regex, there might be multiple outputs, each output may have multiple lines
    """
    outputs = re.findall(r"```output\n(.*?)\n```", pred_str, re.DOTALL)
    if last_only:
        return outputs[-1] if len(outputs) > 0 else ""
    else:
        return outputs
