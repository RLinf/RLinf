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

import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Union

import regex
from latex2sympy2 import latex2sympy
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

from toolkits.math_verifier.parser import extract_answer

global_executor = ProcessPoolExecutor(max_workers=40)


def choice_answer_clean(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    # Clean the answer based on the dataset
    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]
    # Remove the period at the end, again!
    pred = pred.rstrip(".").rstrip("/")
    return pred


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = r"\begin{pmatrix}" + m.replace(",", "\\") + r"\end{pmatrix}"
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except Exception:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception:
                pass
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def numeric_equal(prediction: float, reference: float):
    from math import isclose

    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_latex, parse_expr, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception:
                try:
                    return f(s)
                except Exception:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    # direct equal
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:
        pass

    # simplify equal
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:
        pass

    # equation equal
    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception:
        pass

    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception:
        pass

    # matrix
    try:
        # if a and b are matrix
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception:
        pass

    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    # print("Judge:", prediction, reference)
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean(prediction) == reference
    ):
        return True

    try:  # 1. numerical equal
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    # 2. symbolic equal
    reference = str(reference).strip()
    prediction = str(prediction).strip()

    # pmatrix (amps)
    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    # deal with [], (), {}
    pred_str, ref_str = prediction, reference
    if (
        prediction.startswith("[")
        and prediction.endswith("]")
        and not reference.startswith("(")
    ) or (
        prediction.startswith("(")
        and prediction.endswith(")")
        and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    # [a, b] vs. [c, d], return a==c and b==d
    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                math_equal(pred_parts[i], ref_parts[i], include_percentage, is_close)
                for i in range(len(pred_parts))
            ):
                return True
    if (
        (
            prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")
        )
        and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
        and (
            reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        )
        and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    ):
        pred_lines = [
            line.strip()
            for line in prediction[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[
                len("\\begin{pmatrix}") : -len("\\end{pmatrix}")
            ].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        math_equal(
                            pred_parts[i],
                            ref_parts[i],
                            include_percentage,
                            is_close,
                        )
                        for i in range(len(pred_parts))
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    # symbolic equal with sympy
    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def call_with_timeout(func, *args, timeout=3, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


def process_results(answer, solution):
    try:
        extracted_answer = extract_answer(answer, "math", use_last_number=False)
        extracted_solution = extract_answer(solution, "math", use_last_number=True)

        if extracted_answer is None or extracted_answer.strip() in ["None", "none", ""]:
            retval = 0
        elif extracted_solution is None or extracted_solution.strip() in [
            "None",
            "none",
            "",
        ]:
            retval = 0
        elif math_equal(extracted_answer, extracted_solution, timeout=False):
            # elif call_with_timeout(math_equal, extracted_answer, extracted_solution):
            retval = 1
        else:
            retval = 0

        return retval, (extracted_answer, extracted_solution)
    except Exception:
        return 0, ("None", "None")


def process_results_process(a, b, output_queue):
    result = process_results(a, b)
    output_queue.put(result)


def reset_global_process_pool():
    global global_executor
    for process in global_executor._processes.values():
        process.kill()
    global_executor.shutdown(wait=True, cancel_futures=True)
    global_executor = ProcessPoolExecutor(max_workers=40)


def verify_math_solution(answer: str, solution: str):
    return process_results(answer, solution)[0]


def math_verify_call(
    responses: List[str],
    references: List[str],
    timeout: int = 10,
    check_xml_format=False,
) -> List:
    assert len(responses) == len(references), (
        len(responses),
        len(references),
    )

    all_jobs = []
    for solutions, gen in zip(references, responses):
        jobs = []
        for sol in solutions:
            job = global_executor.submit(verify_math_solution, gen, sol)
            jobs.append(job)
        all_jobs.append(jobs)

    labels = []
    has_timeout = False
    for jobs in all_jobs:
        label = 0
        try:
            for job in as_completed(jobs, timeout=timeout):
                x = job.result()
                label = label or x
        except TimeoutError:
            has_timeout = True
        labels.append(label)

    if has_timeout:
        reset_global_process_pool()
    return labels


class MathRewardModel:
    def __init__(self, scale: float):
        self.scale = scale

    def get_reward(
        self, response: List[str], reference: List[List[str]]
    ) -> List[float]:
        """
        Calculates reward scores for a list of responses compared to corresponding lists of reference answers.
        For each response, the function checks if it matches any of the provided references using the `process_results` function.
        The reward for each response is computed as the first element of the result (converted to float) multiplied by `self.scale`.
        Args:
            response (List[str]): A list of response strings to be evaluated.
            reference (List[List[str]]): A list where each element is a list of reference strings corresponding to each response.
        Returns:
            List[float]: A list of reward scores, one for each response.
        """

        results = []
        for resp, refs in zip(response, reference):
            result = any(process_results(resp, ref)[0] for ref in refs)
            results.append((1 if result else -1) * self.scale)
        return results


if __name__ == "__main__":
    sample = {
        "answers": ["\\boxed{-\\frac{2}{3}}"],
        "solutions": [
            "1. **Apply the operation $\\otimes$ to the innermost parentheses first:**\n   \\[\n   (1 \\otimes 2) \\otimes 3 = \\left(\\frac{1^2}{2}\\right) \\otimes 3 = \\frac{1}{2} \\otimes 3\n   \\]\n   \\[\n   1 \\otimes (2 \\otimes 3) = 1 \\otimes \\left(\\frac{2^2}{3}\\right) = 1 \\otimes \\frac{4}{3}\n   \\]\n\n2. **Calculate each part using the definition of $\\otimes$:**\n   \\[\n   \\frac{1}{2} \\otimes 3 = \\frac{\\left(\\frac{1}{2}\\right)^2}{3} = \\frac{\\frac{1}{4}}{3} = \\frac{1}{12}\n   \\]\n   \\[\n   1 \\otimes \\frac{4}{3} = \\frac{1^2}{\\frac{4}{3}} = \\frac{1}{\\frac{4}{3}} = \\frac{3}{4}\n   \\]\n\n3. **Subtract the two results:**\n   \\[\n   \\left(\\frac{1}{12}\\right) - \\left(\\frac{3}{4}\\right) = \\frac{1}{12} - \\frac{9}{12} = -\\frac{8}{12} = -\\frac{2}{3}\n   \\]\n\n4. **Conclude with the final answer:**\n   \\[\n   \\boxed{A}\n   \\]",
            "\\boxed{-\\frac{2}{3}}",
        ],
    }

    print(
        math_verify_call(
            sample["answers"] * 100,
            [sample["solutions"]] * 100,
            check_xml_format=True,
        )
    )
