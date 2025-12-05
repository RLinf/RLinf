import os
import re
import random
import yaml
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import subprocess
# from apache_beam.examples.dataframe.flight_delays import input_date


# -----------------------------
# 你已有的类：BDDLParser / 5个扰动器
# （保持实现不变，直接复用你上面的代码）
# -----------------------------

class BDDLParser:
    """解析BDDL文件并提取相关信息"""

    def __init__(self, file_content: str):
        self.file_content = file_content
        self.objects_of_interest = self._parse_obj_of_interest()
        self.initial_states = self._parse_initial_states()

    def _parse_obj_of_interest(self) -> List[str]:
        """解析感兴趣的物体"""
        obj_pattern = r'\(:obj_of_interest(.*?)\)'
        obj_match = re.search(obj_pattern, self.file_content, re.DOTALL)
        if not obj_match:
            return []
        obj_content = obj_match.group(1)
        objects = re.findall(r'(\w+_\d+)', obj_content)
        return objects

    def _parse_initial_states(self) -> Dict[str, str]:
        """
        解析 bddl (:init ...) 部分，返回 initial_states[obj] = region
        """
        initial_states = {}
        init_block_match = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", self.file_content, re.S)
        if not init_block_match:
            return initial_states
        init_text = init_block_match.group(1)
        for match in re.finditer(r"\(On\s+(\w+)\s+(\w+)\)", init_text):
            obj, region = match.groups()
            initial_states[obj] = region
        return initial_states


class SwapPerturbator:
    """根据配置文件进行交换扰动"""

    def __init__(self, parser: BDDLParser, config_path: str):
        self.parser = parser
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def perturb(self, task_suite_name: str, task_name: str) -> str:
        content = self.parser.file_content
        objs_interest = list(self.parser.objects_of_interest or [])
        if not objs_interest:
            print("没有找到感兴趣物体")
            return content

        task_cfg = self.config.get(task_suite_name, {}).get(task_name, None)
        if task_cfg is None:
            print(f"任务 {task_name} 没有配置 allowed_swaps")
            return content

        init_states = dict(self.parser.initial_states)

        used = set()
        pairs = []

        random.shuffle(objs_interest)

        def candidates_for(obji: str):
            if isinstance(task_cfg, dict):
                if obji in task_cfg and isinstance(task_cfg[obji], list):
                    return list(task_cfg[obji])
                if "__any__" in task_cfg and isinstance(task_cfg["__any__"], list):
                    return list(task_cfg["__any__"])
                return []
            elif isinstance(task_cfg, list):
                return list(task_cfg)
            else:
                return []

        for obj in objs_interest:
            if obj in used:
                continue
            cand_pool = candidates_for(obj)
            cand_pool = [
                x for x in cand_pool
                if x != obj and x in init_states and x not in used
            ]
            if not cand_pool:
                print(f"[跳过] 感兴趣物体 {obj} 没有可用候选（可能未在 init 中或已被占用）")
                continue

            swap_obj = random.choice(cand_pool)

            reg_a = init_states.get(obj)
            reg_b = init_states.get(swap_obj)
            if not reg_a or not reg_b:
                print(f"[跳过] {obj} 或 {swap_obj} 不在 init 中，无法交换")
                continue

            pat_a = rf"\(On\s+{re.escape(obj)}\s+{re.escape(reg_a)}\s*\)"
            pat_b = rf"\(On\s+{re.escape(swap_obj)}\s+{re.escape(reg_b)}\s*\)"

            content, n1 = re.subn(pat_a, f"(On {obj} {reg_b})", content, count=1)
            content, n2 = re.subn(pat_b, f"(On {swap_obj} {reg_a})", content, count=1)

            if n1 > 0 and n2 > 0:
                init_states[obj], init_states[swap_obj] = reg_b, reg_a
                used.add(obj)
                used.add(swap_obj)
                pairs.append((obj, swap_obj))
                print(f"任务 {task_name}: 已将 {obj} 与 {swap_obj} 交换位置")
            else:
                print(f"[警告] {obj} 或 {swap_obj} 的 On 语句未匹配到，可能 BDDL 格式与正则不一致")

        if not pairs:
            print("没有形成任何交换对，未修改文件")

        return content


class ObjectReplacePerturbator:
    """
    根据 ood_object.yaml 进行“物体替换扰动”
    """

    def __init__(self, parser: BDDLParser, config_path: str):
        self.parser = parser
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def _extract_language_span(self, text: str):
        m = re.search(r"\(:language\b.*?\)", text, flags=re.S)
        return m.span() if m else None

    def perturb(self, task_suite_name: str, task_name: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        suite_cfg = self.config.get(task_suite_name, {})
        task_cfg: Dict[str, List[str]] = suite_cfg.get(task_name, {})

        if not task_cfg:
            print(f"[物体替换] 任务 {task_name} 在配置中没有条目，跳过。")
            return self.parser.file_content

        mapping: Dict[str, str] = {}
        for obj_interest, candidates in task_cfg.items():
            if not candidates:
                print(f"[物体替换] {obj_interest} 没有可替换候选，跳过。")
                continue
            chosen = random.choice(candidates)
            mapping[obj_interest] = chosen

        if not mapping:
            print("[物体替换] 没有形成任何替换映射，跳过。")
            return self.parser.file_content

        content = self.parser.file_content
        lang_span = self._extract_language_span(content)
        if lang_span:
            s, e = lang_span
            prefix = content[:s]
            language_block = content[s:e]
            suffix = content[e:]
        else:
            prefix, language_block, suffix = content, "", ""

        for old_name, new_name in mapping.items():
            prefix = prefix.replace(old_name, new_name)
            suffix = suffix.replace(old_name, new_name)

        new_content = prefix + language_block + suffix

        for k, v in mapping.items():
            print(f"[物体替换] {task_name}: {k} -> {v}")

        return new_content


class LanguagePerturbator:
    """
    从 ood_language.yaml 读取候选指令文本，随机选择一条，替换 (:language ...)
    """

    def __init__(self, parser: BDDLParser, config_path: str):
        self.parser = parser
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def _find_language_block(self, text: str):
        m = re.search(r"\(:language\s*(.*?)\)", text, flags=re.S)
        if not m:
            return None
        inner = m.group(1)
        inner_start = m.start(1)
        inner_end = m.end(1)
        return (inner_start, inner_end, inner)

    def perturb(self, task_suite_name: str, task_name: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        candidates = (self.config.get(task_suite_name, {}) or {}).get(task_name, [])
        if not candidates:
            print(f"[language扰动] 任务 {task_name} 在配置中没有候选，跳过。")
            return self.parser.file_content

        new_lang = random.choice(candidates)

        block = self._find_language_block(self.parser.file_content)
        if not block:
            print("[language扰动] 未找到 (:language ...) 段，跳过。")
            return self.parser.file_content

        s, e, old_inner = block
        new_content = self.parser.file_content[:s] + new_lang + self.parser.file_content[e:]

        print(f"[language扰动] {task_name}: '{old_inner}' -> '{new_lang}'")
        return new_content


class TaskPerturbator:
    """
    从 ood_task.yaml 读取候选任务，将 (:language ...) 与 (:goal ...) 同时替换，并替换 (:obj_of_interest ...)
    """

    def __init__(self, parser: BDDLParser, config_path: str):
        self.parser = parser
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def _find_language_inner_span(self, text: str):
        m = re.search(r"\(:language\s*(.*?)\)", text, flags=re.S)
        if not m:
            return None
        return (m.start(1), m.end(1), m.group(1))

    def _find_outer_block_span(self, text: str, head: str):
        start = text.find(head)
        if start < 0:
            return None
        i, depth = start, 0
        while i < len(text):
            ch = text[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
            i += 1
        return None

    def _replace_language(self, text: str, new_lang: str) -> str:
        span = self._find_language_inner_span(text)
        if not span:
            print("[TaskPerturbator] 未找到 (:language ...) 段，跳过 language 替换。")
            return text
        s, e, old = span
        print(f"[TaskPerturbator] language: '{old}' -> '{new_lang}'")
        return text[:s] + new_lang + text[e:]

    def _replace_goal(self, text: str, new_goal_expr: str) -> str:
        span = self._find_outer_block_span(text, "(:goal")
        if not span:
            print("[TaskPerturbator] 未找到 (:goal ...) 段，跳过 goal 替换。")
            return text
        start, end = span
        replacement = "(:goal\n  " + new_goal_expr + "\n)"
        print(f"[TaskPerturbator] goal: 替换为 {new_goal_expr}")
        return text[:start] + replacement + text[end:]

    def _replace_obj_of_interest(self, text: str, new_objs: list) -> str:
        span = self._find_outer_block_span(text, "(:obj_of_interest")
        if not span:
            print("[TaskPerturbator] 未找到 (:obj_of_interest ...) 段，跳过替换。")
            return text
        start, end = span
        replacement = "(:obj_of_interest\n"
        for obj in new_objs:
            replacement += f"  {obj}\n"
        replacement += ")"
        print(f"[TaskPerturbator] obj_of_interest: 替换为 {new_objs}")
        return text[:start] + replacement + text[end:]

    def perturb(self, task_suite_name: str, task_name: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        suite_cfg = self.config.get(task_suite_name, {})
        task_cfg = suite_cfg.get(task_name, {})
        if not task_cfg:
            print(f"[TaskPerturbator] 任务 {task_name} 在配置中没有候选，跳过。")
            return self.parser.file_content

        language_options = list(task_cfg.keys())
        chosen_lang = random.choice(language_options)
        chosen_cfg = task_cfg.get(chosen_lang, {})
        chosen_goal = chosen_cfg.get("goal")
        chosen_objs = chosen_cfg.get("obj_of_interest", [])

        if not chosen_goal:
            print(f"[TaskPerturbator] 任务 {task_name} 的 language '{chosen_lang}' 没有 goal，跳过。")
            return self.parser.file_content

        new_content = self.parser.file_content
        new_content = self._replace_language(new_content, chosen_lang)
        new_content = self._replace_goal(new_content, chosen_goal)
        new_content = self._replace_obj_of_interest(new_content, chosen_objs)
        return new_content


class EnvironmentReplacePerturbator:
    """
    环境替换扰动（全局直接替换）+ 修改 problem 名称场景标记 + 修正(:fixtures)右侧类型
    """

    def __init__(self, parser: BDDLParser, config_path: str):
        self.ALLOWED_ENVS = {"main_table", "kitchen_table", "living_room_table", "study_table", "floor"}
        self.ENV_TOKEN = {
            "main_table": "Tabletop",
            "kitchen_table": "Kitchen_Tabletop",
            "living_room_table": "Living_Room_Tabletop",
            "study_table": "Study_Tabletop",
            "floor": "Floor",
        }
        self.ENV_FIXTYPE = {
            "main_table": "table",
            "kitchen_table": "kitchen_table",
            "living_room_table": "living_room_table",
            "study_table": "study_table",
            "floor": "floor",
        }
        self.parser = parser
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def _extract_current_env(self, task_suite_name: str, task_name: str) -> str | None:
        suite_cfg = self.config.get(task_suite_name, {})
        entry = suite_cfg.get(task_name)
        if entry is None:
            return None
        if isinstance(entry, list):
            return entry[0] if entry else None
        if isinstance(entry, str):
            return entry
        return None

    def _rewrite_problem_env_token(self, text: str, env_name: str) -> str:
        token = self.ENV_TOKEN.get(env_name, "Tabletop")
        pattern = r"\(define\s*\(problem\s+LIBERO_[A-Za-z_]*\)"
        replacement = f"(define (problem LIBERO_{token}_Manipulation)"
        return re.sub(pattern, replacement, text, count=1)

    def _find_outer_block_span(self, text: str, head: str):
        start = text.find(head)
        if start < 0:
            return None
        i, depth = start, 0
        while i < len(text):
            ch = text[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
            i += 1
        return None

    def _rewrite_fixtures_type(self, text: str, fixture_name: str, new_type: str) -> str:
        span = self._find_outer_block_span(text, "(:fixtures")
        if not span:
            return text
        s, e = span
        block = text[s:e]
        pattern = rf"(^\s*{re.escape(fixture_name)}\s*-\s*)([A-Za-z_][A-Za-z0-9_]*)"
        new_block, n = re.subn(pattern, rf"\1{new_type}", block, count=1, flags=re.M)
        if n == 0:
            return text
        return text[:s] + new_block + text[e:]

    def perturb(self, task_suite_name: str, task_name: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        current_env = self._extract_current_env(task_suite_name, task_name)
        if not current_env:
            print(f"[环境替换] 任务 {task_name} 未在配置中找到环境（或为空列表），跳过。")
            return self.parser.file_content
        if current_env not in self.ALLOWED_ENVS:
            print(f"[环境替换] 配置环境 '{current_env}' 不在允许集合 {self.ALLOWED_ENVS} 中，跳过。")
            return self.parser.file_content

        candidates = list(self.ALLOWED_ENVS - {current_env})
        if not candidates:
            print("[环境替换] 无候选可替换环境，跳过。")
            return self.parser.file_content

        # new_env = random.choice(candidates)
        new_env = "living_room_table"
        new_content = self.parser.file_content.replace(current_env, new_env)
        new_content = self._rewrite_problem_env_token(new_content, new_env)
        new_fix_type = self.ENV_FIXTYPE.get(new_env, None)
        if new_fix_type:
            new_content = self._rewrite_fixtures_type(new_content, new_env, new_fix_type)

        print(f"[环境替换] {task_name}: {current_env} -> {new_env}")
        return new_content


# -----------------------------------------
# 新增：组合扰动器（按布尔开关混合执行）
# -----------------------------------------

@dataclass
class PerturbFlags:
    use_environment: bool = False
    use_swap: bool = False
    use_object: bool = False
    use_language: bool = False
    use_task: bool = False


class BDDLCombinedPerturbator:
    """
    组合扰动器：
    - 通过 PerturbFlags 指定哪些扰动启用
    - 通过 configs 指定每种扰动的 YAML 路径
      configs = {
        "environment": "./ood_environment.yaml",
        "swap": "./ood_spatial_relation.yaml",
        "object": "./ood_object.yaml",
        "language": "./ood_language.yaml",
        "task": "./ood_task.yaml",
      }
    - 默认执行顺序：
        environment -> object -> language -> task
      但规则：
        1) 若 use_swap=True，则 SwapPerturbator 必须最先执行
        2) 若 use_task=True，其它扰动必须全为 False
    """

    def __init__(self, configs: Dict[str, str]):
        self.configs = configs or {}

    @staticmethod
    def _task_name_from_path(input_path: str) -> str:
        return os.path.basename(input_path).replace(".bddl", "").strip()

    @staticmethod
    def _apply_and_reparse(content: str, perturbator_cls, cfg_path: str,
                           call_kwargs: Dict[str, Any],
                           task_suite_name: str, task_name: str) -> str:
        """
        用当前 content 构造 parser 和 perturbator，执行一次扰动；返回新的 content。
        """
        parser = BDDLParser(content)
        perturbator = perturbator_cls(parser, cfg_path)
        new_content = perturbator.perturb(task_suite_name=task_suite_name, task_name=task_name, **call_kwargs)
        return new_content

    def perturb_content(self,
                        content: str,
                        task_suite_name: str,
                        task_name: str,
                        flags: PerturbFlags,
                        seed: Optional[int] = None) -> str:
        current = content

        # 规则检查
        # use_task 模式必须互斥
        if flags.use_task:
            if flags.use_environment or flags.use_swap or flags.use_object or flags.use_language:
                raise ValueError("禁止在 use_task=True 时开启其它扰动！")

        # 1) 若启用 swap，则必须最先执行
        if flags.use_swap:
            cfg = self.configs.get("swap")
            if cfg and os.path.exists(cfg):
                parser = BDDLParser(current)
                perturbator = SwapPerturbator(parser, cfg)
                current = perturbator.perturb(task_suite_name=task_suite_name, task_name=task_name)
            else:
                print("[组合扰动] 缺少 swap 配置或路径不存在，跳过交换扰动。")

        # 2) 其它扰动（按顺序执行）
        if flags.use_environment:
            cfg = self.configs.get("environment")
            if cfg and os.path.exists(cfg):
                current = self._apply_and_reparse(
                    current, EnvironmentReplacePerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name
                )
            else:
                print("[组合扰动] 缺少 environment 配置或路径不存在，跳过环境替换。")

        if flags.use_object:
            cfg = self.configs.get("object")
            if cfg and os.path.exists(cfg):
                current = self._apply_and_reparse(
                    current, ObjectReplacePerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name
                )
            else:
                print("[组合扰动] 缺少 object 配置或路径不存在，跳过物体替换。")

        if flags.use_language:
            cfg = self.configs.get("language")
            if cfg and os.path.exists(cfg):
                current = self._apply_and_reparse(
                    current, LanguagePerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name
                )
            else:
                print("[组合扰动] 缺少 language 配置或路径不存在，跳过语言替换。")

        # 3) 任务扰动（若启用，且保证其它都禁用）
        if flags.use_task:
            cfg = self.configs.get("task")
            if cfg and os.path.exists(cfg):
                current = self._apply_and_reparse(
                    current, TaskPerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name
                )
            else:
                print("[组合扰动] 缺少 task 配置或路径不存在，跳过任务替换。")

        return current


class EvalEnvCreator:
    def __init__(self, input_dir: str, base_output_dir: str = None, script_path: str = "generate_init_states.py"):
        """
        初始化评估环境创建器

        :param input_dir: 输入的bddl文件目录，例如：
                          /LIBERO/libero/libero/bddl_files/libero_goal_temp
        :param base_output_dir: 输出目录的基础路径（可选）。
                                如果不传，将自动替换 input_dir 中的 "bddl_files" 为 "init_files"
        """
        self.input_dir = input_dir.rstrip("/")
        self.folder_name = os.path.basename(self.input_dir)
        self.script_path = script_path

        if base_output_dir:
            self.output_dir = os.path.join(base_output_dir, self.folder_name)
        else:
            # 自动替换 bddl_files → init_files
            self.output_dir = self.input_dir.replace("bddl_files", "init_files")

    def create_env(self):
        """
        创建评估环境，执行 generate_init_states.py
        """
        os.makedirs(self.output_dir, exist_ok=True)

        cmd = [
            sys.executable, self.script_path,
            "--bddl_base_dir", self.input_dir,
            "--output_dir", self.output_dir
        ]

        print(f"[INFO] 运行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)



# -----------------------------------------
# 便捷方法：处理单个 .bddl 文件（读入 -> 混合扰动 -> 写出）
# -----------------------------------------

def process_bddl_file_mixed(input_dir: str,
                            task_suite_name: str,
                            flags: PerturbFlags,
                            configs: Dict[str, str],
                            seed: Optional[int] = None) -> None:
    """
        对指定目录下的 BDDL 文件进行扰动，并保存到临时目录。

        Args:
            input_dir (str): 输入 BDDL 文件所在的目录。
            configs (dict): 扰动器配置。
            task_suite_name (str): 任务集名称。
            flags (dict): 扰动参数标志。
            seed (int): 随机种子。
        """
    input_path = Path(input_dir)
    output_dir = input_path.parent / f"{input_path.name}_temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in input_path.glob("*.bddl"):
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

        task_name = file_path.stem  # 去掉后缀的文件名
        pipeline = BDDLCombinedPerturbator(configs=configs)
        new_content = pipeline.perturb_content(
            content=content,
            task_suite_name=task_suite_name,
            task_name=task_name,
            flags=flags,
            seed=seed
        )

        output_path = output_dir / file_path.name
        with output_path.open("w", encoding="utf-8") as f:
            f.write(new_content)

    print(f"[组合扰动] 处理完成，输出：{output_dir}")
    return str(output_dir)


# -----------------------------------------
# 示例 main：与原始 main 等价，但改为混合扰动方式
# -----------------------------------------

def create_env(
    configs: dict = None,
):
    """
    创建评估环境

    :param input_path: 输入的 bddl 文件路径
    :param script_path: generate_init_states.py 的路径
    :param init_output_dir: 最终 init 文件输出路径
    :param task_suite_name: 任务套件名 (默认: libero_goal)
    :param seed: 随机种子 (默认: 28, None 则完全随机)
    :param flags: 各扰动开关 (默认启用 environment/swap/object/language，关闭 task)
    :param configs: 各扰动配置文件路径
    """
    # 默认扰动 flags
    flags = PerturbFlags(
        use_environment=configs.get("use_environment", False),
        use_swap=configs.get("use_swap", False),
        use_object=configs.get("use_object", False),
        use_language=configs.get("use_language", False),
        use_task=configs.get("use_task", False),
    )

    ood_task_configs = configs.get("ood_task_configs", {})

    # 生成临时的 bddl 输出路径
    temp_output_dir = process_bddl_file_mixed(
        input_dir=configs.get("bddl_files_path", ""),
        task_suite_name=configs.get("task_suite_name", ""),
        flags=flags,
        configs=ood_task_configs,
        seed=configs.get("seed", int),
    )

    # 调用 EvalEnvCreator
    creator = EvalEnvCreator(
        input_dir=temp_output_dir,
        script_path=configs.get("script_path", ""),
        base_output_dir=configs.get("init_file_dir", ""),
    )
    creator.create_env()

    return



# if __name__ == '__main__':
#     create_env(
#         input_path="/LIBERO/libero/libero/bddl_files/libero_goal/",
#         script_path="/LIBERO/notebooks/generate_init_states.py",
#         init_output_dir="/LIBERO/libero/libero/init_files/",
#         task_suite_name="libero_goal",
#         seed=42,
#     )