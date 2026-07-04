#!/usr/bin/env python
# -*- coding:utf-8 -*-


import os
import json
import time
import requests
import re
import ast
import traceback
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# ========== 幻觉类型定义 ==========
label2id = {
    "No Hallucination": 0,
    "1.1": 1, "1.2": 2,
    "2.1": 3, "2.2": 4, "2.3": 5, "2.4": 6,
    "3.1": 7, "3.2": 8,
    "4.1": 9, "4.2": 10,
    "5.1": 11, "5.2": 12,
    "6.1": 13,
    "7.1": 14, "7.2": 15,
}

id2label = {
    0: "No Hallucination (0)",
    1: "Data Type Mismatch (1.1)",
    2: "Data Structure Misinterpretation (1.2)",
    3: "Global Logic Misalignment (2.1)",
    4: "Context Inconsistency (2.2)",
    5: "Local Logic Misalignment (2.3)",
    6: "Incomplete Implementation (2.4)",
    7: "Code Duplication (3.1)",
    8: "Dead Code (3.2)",
    9: "API Usage Errors (4.1)",
    10: "External Dependency Errors (4.2)",
    11: "Robustness Defects (5.1)",
    12: "Security Vulnerabilities (5.2)",
    13: "Resource Overuse (6.1)",
    14: "Syntax Overuse (7.1)",
    15: "Non-Code Content (7.2)",
}

full_label_names = {
    0: "No Hallucination",
    1: "1.1 Data Type Mismatch (DTM)",
    2: "1.2 Data Structure Misinterpretation (DSM)",
    3: "2.1 Global Logic Misalignment (GLM)",
    4: "2.2 Context Inconsistency (CI)",
    5: "2.3 Local Logic Misalignment (LoLM)",
    6: "2.4 Incomplete Implementation (II)",
    7: "3.1 Code Duplication (CD)",
    8: "3.2 Dead Code (DC)",
    9: "4.1 API Knowledge Errors (AKE)",
    10: "4.2 External Dependency Errors",
    11: "5.1 Robustness and Security Issues (RSI)",
    12: "5.2 Security Vulnerabilities",
    13: "6.1 Resource Overuse (RO)",
    14: "7.1 Syntax Errors (SE)",
    15: "7.2 Non-Code Content (NCC)",
}

hallucination_definitions = {
    full_label_names[0]: "The generated code has no hallucination issues and meets the task requirements correctly.",
    full_label_names[1]: "The model misinterprets an operand's data type or parameter value, causing the generated code to perform operations that violate type constraints or predefined rules.",
    full_label_names[2]: "The model misunderstands the underlying data structure, leading the code to access non-existent array indices, dictionary keys, or other invalid members.",
    full_label_names[3]: "The generated snippet diverges markedly from the user's task description or intended goal at the overall functional level; in extreme cases, its logic is so confused that the core intent becomes unrecognizable.",
    full_label_names[4]: "The LLM fails to interpret or continuously maintain a coherent understanding of contextual information (both the initial prompt and previously generated code), indicating a drift from the intended logical flow and a breakdown of strict contextual consistency.",
    full_label_names[5]: "The code deviates from the expected intent at a local implementation level: a segment's semantic logic is incorrect, causing localized functional errors even though the overall structure or most logic remains sound.",
    full_label_names[6]: "The model fails to generate the whole code logic or module requested, leaving out one or more critical functionalities specified in the task description.",
    full_label_names[7]: "Excessive and unnecessary repetition of a given code fragment, leading to redundancy and possible inefficiency. Duplication appears in two forms: input-context duplication, where the model verbatim copies code already present in the prompt, and in-generation duplication, where the generated logic repeatedly reuses similar or identical blocks.",
    full_label_names[8]: "Code that is unreachable under all execution paths, or code that may run but whose results are never consumed elsewhere, leaving program state and output unaffected.",
    full_label_names[9]: "The generated code contradicts established specifications or factual knowledge about an API or module: e.g., incorrect API invocation, importing a non-existent library, or failing to load a module from the declared path.",
    full_label_names[10]: "Errors related to external dependencies such as missing or incompatible third-party libraries, incorrect dependency versions, or failure to resolve external resources.",
    full_label_names[11]: "The code fails to handle exceptional conditions, making it prone to crashes or uncaught exceptions at edge cases; it may also embed known security vulnerabilities or mishandle resources (e.g., memory leaks).",
    full_label_names[12]: "Specific security-related hallucinations, such as SQL injection vulnerabilities, cross-site scripting (XSS) flaws, or unauthorized access risks in the generated code.",
    full_label_names[13]: "The model underestimates memory, computation time, stack depth, or numeric bounds, or mismanages iteration control; as a result, execution fails due to exceeding physical limits (e.g., out-of-memory) or computational bounds (e.g., numeric overflow, infinite-loop timeouts).",
    full_label_names[14]: "The code contains syntactic violations that prevent it from passing the compiler or interpreter, rendering it non-executable.",
    full_label_names[15]: "The output is not executable code but rather natural-language prose, comments, placeholder text, or another unintended format, contradicting the code-generation objective."
}
# ========== 类型定义结束 ==========

class SimpleLLMRepairTool:
    def __init__(self, llm_config):
        self.api_key = llm_config["api_key"]
        self.model_name = llm_config.get("model_name", "Claude-HaiKu-4.5")
        self.base_url = llm_config.get("api_base", "https://api.deepbricks.ai/v1/chat/completions")
        self.temperature = llm_config.get("temperature", 0.3)
        self.max_tokens = llm_config.get("max_tokens", 4096)
        self.timeout = llm_config.get("timeout", 600)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _call_api(self, prompt, retries=3):
        print(f"📏 Prompt 字符数: {len(prompt)}")
        for attempt in range(retries):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": False
                }
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"⚠️ API调用失败（{attempt + 1}/{retries}）：{e}")
                if attempt == retries - 1:
                    return ""
                time.sleep(2)
        return ""

    def clean_code(self, code):
        if not code:
            return ""
        if "```" in code:
            parts = code.split("```")
            if len(parts) >= 2:
                code = parts[1]
        if code.startswith(("python", "Python")):
            code = "\n".join(code.split("\n")[1:])
        return code.strip()

    # ---------- 辅助方法：移除顶层函数调用 ----------
    def _remove_immediate_calls(self, code: str) -> str:
        """移除代码顶层中的独立函数调用（如 solve()、main() 等）"""
        try:
            tree = ast.parse(code)
            new_body = []
            for node in tree.body:
                # 跳过表达式语句中的函数调用
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    continue
                new_body.append(node)
            new_tree = ast.Module(body=new_body, type_ignores=[])
            ast.fix_missing_locations(new_tree)
            return ast.unparse(new_tree)
        except Exception:
            return code

    # ---------- 测试执行（与原始完整版本一致，但已修复 multiprocessing 问题） ----------
    def _normalize_test_case_to_str(self, test_case_raw):
        """归一化测试用例为可执行字符串（支持多种格式）"""
        if test_case_raw is None:
            return None
        if isinstance(test_case_raw, str):
            return test_case_raw
        if isinstance(test_case_raw, list):
            return "\n".join(map(str, test_case_raw))
        if isinstance(test_case_raw, dict):
            if "input" in test_case_raw and "output" in test_case_raw:
                inp = test_case_raw["input"]
                out = test_case_raw["output"]
                if isinstance(inp, list):
                    inp = "\n".join(map(str, inp))
                if isinstance(out, list):
                    out = "\n".join(map(str, out))
                return self._build_stdio_test_script([inp], [out])
            if "inputs" in test_case_raw and "outputs" in test_case_raw:
                inputs = test_case_raw["inputs"]
                outputs = test_case_raw["outputs"]

                def normalize_io(x):
                    if isinstance(x, str):
                        return [x]
                    if isinstance(x, list):
                        res = []
                        for item in x:
                            if isinstance(item, str):
                                res.append(item)
                            elif isinstance(item, list):
                                res.append("\n".join(map(str, item)))
                            else:
                                res.append(str(item))
                        return res
                    return [str(x)]

                inps = normalize_io(inputs)
                outs = normalize_io(outputs)
                m = min(len(inps), len(outs))
                return self._build_stdio_test_script(inps[:m], outs[:m])
            if "test" in test_case_raw and isinstance(test_case_raw["test"], str):
                return test_case_raw["test"]
            return str(test_case_raw)
        return str(test_case_raw)

    def _build_stdio_test_script(self, inputs_list, outputs_list):
        cases = []
        for inp, out in zip(inputs_list, outputs_list):
            inp = "" if inp is None else str(inp)
            out = "" if out is None else str(out)
            if not inp.endswith("\n"):
                inp += "\n"
            if not out.endswith("\n"):
                out += "\n"
            cases.append((inp, out))

        script = f'''
import sys, io
_TEST_CASES = {repr(cases)}

def _run_program(env, input_data: str):
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(input_data)
    sys.stdout = io.StringIO()
    try:
        if "solve" in env and callable(env["solve"]):
            env["solve"]()
        elif "main" in env and callable(env["main"]):
            env["main"]()
        return sys.stdout.getvalue()
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout

def check(candidate):
    env = candidate if isinstance(candidate, dict) else globals()
    for idx, (inp, expected) in enumerate(_TEST_CASES, 1):
        got = _run_program(env, inp)
        got_norm = "\\n".join([line.rstrip() for line in got.splitlines()]).strip()
        exp_norm = "\\n".join([line.rstrip() for line in expected.splitlines()]).strip()
        assert got_norm == exp_norm, f"case{{idx}} failed: expected={{exp_norm!r}}, got={{got_norm!r}}"
'''
        return script.strip()

    @staticmethod
    def _extract_assertions(test_case):
        """静态方法：提取测试用例中的断言表达式"""
        if isinstance(test_case, list):
            test_case = "\n".join(test_case)
        elif not isinstance(test_case, str):
            return []
        try:
            tree = ast.parse(test_case)
        except SyntaxError:
            return []
        assertions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                try:
                    expr_src = ast.get_source_segment(test_case, node.test)
                except Exception:
                    expr_src = None
                if expr_src is None:
                    try:
                        expr_src = ast.unparse(node.test)
                    except Exception:
                        expr_src = None
                if expr_src:
                    assertions.append(expr_src.strip())
        return list(dict.fromkeys(assertions))

    def _run_test_case_in_process(self, repaired_code, test_case, timeout=60):
        # 在父进程中预处理代码：移除顶层调用（避免子进程中调用实例方法）
        cleaned_code = self._remove_immediate_calls(repaired_code)

        def worker(code, test, result_queue):
            try:
                local_env = {"__name__": "__not_main__"}
                exec(code, local_env)
                exec(test, local_env)

                check_fn = local_env.get("check", None)
                if callable(check_fn):
                    function_name = None
                    try:
                        test_tree = ast.parse(test)
                        for node in ast.walk(test_tree):
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "check":
                                if node.args and isinstance(node.args[0], ast.Name):
                                    function_name = node.args[0].id
                                    break
                    except Exception:
                        pass

                    if function_name is None:
                        try:
                            code_tree = ast.parse(code)
                            for node in ast.walk(code_tree):
                                if isinstance(node, ast.FunctionDef):
                                    function_name = node.name
                                    break
                        except Exception:
                            pass

                    if function_name is None:
                        m = re.search(r"def\s+(\w+)\s*\(", code)
                        function_name = m.group(1) if m else "candidate"

                    if function_name not in local_env or not callable(local_env[function_name]):
                        result_queue.put(("function_not_found", f"函数 '{function_name}' 未定义", []))
                        return

                    candidate_func = local_env[function_name]
                    try:
                        # 注意：这里调用的是静态方法
                        num_assertions = len(SimpleLLMRepairTool._extract_assertions(test))
                    except Exception:
                        num_assertions = None

                    try:
                        check_fn(candidate_func)
                    except AssertionError as e:
                        msg = str(e) or "断言失败"
                        failed_assertions = [{
                            "assertion_id": 1,
                            "expression": "check(candidate)",
                            "error": msg
                        }]
                        if num_assertions:
                            result_msg = f"测试断言失败（check(candidate) 抛出 AssertionError，约包含 {num_assertions} 条 assert）: {msg}"
                        else:
                            result_msg = f"测试断言失败（check(candidate) 抛出 AssertionError）: {msg}"
                        result_queue.put(("assertion_failed", result_msg, failed_assertions))
                        return
                    except Exception as e:
                        tb = traceback.format_exc()
                        failed_assertions = [{
                            "assertion_id": 1,
                            "expression": "check(candidate)",
                            "error": f"执行错误: {type(e).__name__}: {e}"
                        }]
                        result_queue.put(("other_exception", f"测试执行错误: {type(e).__name__}: {e}\n{tb}", failed_assertions))
                        return

                    ok_msg = "check(candidate) 成功执行，所有测试断言通过"
                    if num_assertions:
                        ok_msg += f"（约 {num_assertions} 条 assert）"
                    result_queue.put(("success", ok_msg, []))
                    return

                # 没有 check 函数时，提取普通 assert
                assertions = SimpleLLMRepairTool._extract_assertions(test)
                if not assertions:
                    result_queue.put(("no_assertions", "测试用例中没有找到 assert，且未定义 check 函数", []))
                    return

                failed_assertions = []
                for idx, assert_expr in enumerate(assertions):
                    try:
                        exec(f"assert {assert_expr}", local_env)
                    except AssertionError as e:
                        failed_assertions.append({
                            "assertion_id": idx + 1,
                            "expression": assert_expr,
                            "error": str(e) or "断言失败"
                        })
                    except Exception as e:
                        failed_assertions.append({
                            "assertion_id": idx + 1,
                            "expression": assert_expr,
                            "error": f"执行错误: {type(e).__name__}: {e}"
                        })

                if failed_assertions:
                    result_queue.put(("assertion_failed", f"{len(failed_assertions)} 个断言失败", failed_assertions))
                else:
                    result_queue.put(("success", f"所有 {len(assertions)} 个断言通过", []))
            except Exception as e:
                tb = traceback.format_exc()
                result_queue.put(("other_exception", f"测试执行错误: {type(e).__name__}: {e}\n{tb}", []))

        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=worker, args=(cleaned_code, test_case, result_queue))
        try:
            process.start()
            process.join(timeout)
            if process.is_alive():
                process.terminate()
                process.join()
                return False, f"test_case 执行超时（{timeout} 秒）", []
            if result_queue.empty():
                return False, "test_case 执行无返回结果", []
            result_type, result_msg, failed_assertions = result_queue.get()
            return result_type == "success", result_msg, failed_assertions
        except Exception as e:
            return False, f"进程执行异常：{e}", []
        finally:
            if process.is_alive():
                process.terminate()
                process.join()

    def judge_repair_success(self, sample, repaired_code):
        if not repaired_code:
            return False, "修复代码为空", {}
        try:
            compile(repaired_code, "<string>", "exec")
        except SyntaxError as e:
            return False, f"语法错误: {e}", {}

        test_case_raw = sample.get("test_case")
        if test_case_raw is None:
            return False, "test_case 缺失，无法验证修复效果", {}

        test_case = self._normalize_test_case_to_str(test_case_raw)
        if not test_case or not isinstance(test_case, str):
            return False, f"test_case 无法归一化为可执行脚本（原类型={type(test_case_raw)}）", {}

        test_success, test_reason, failed_assertions = self._run_test_case_in_process(repaired_code, test_case, timeout=60)
        if not test_success:
            if failed_assertions:
                detail_lines = [f"❌ test_case未通过：{len(failed_assertions)}个断言失败"]
                for fail in failed_assertions[:3]:
                    detail_lines.append(f"  断言{fail['assertion_id']}: assert {fail['expression'][:60]}...")
                    detail_lines.append(f"      错误: {fail['error']}")
                if len(failed_assertions) > 3:
                    detail_lines.append(f"  ... 还有 {len(failed_assertions)-3} 个断言失败")
                return False, "\n".join(detail_lines), {
                    "failed_assertions": failed_assertions,
                    "total_assertions": len(failed_assertions)
                }
            return False, f"❌ test_case未通过: {test_reason}", {}
        return True, "✅ 修复成功（通过所有测试断言）", {"passed_assertions": test_reason}

    def judge_repair_success_with_timeout(self, sample, repaired_code, timeout=120):
        def _validate():
            return self.judge_repair_success(sample, repaired_code)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_validate)
            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                return False, f"验证阶段超时（{timeout}秒）", {}

    def judge_success(self, sample, repaired_code):
        """兼容旧接口，返回 (bool, str)"""
        success, reason, _ = self.judge_repair_success_with_timeout(sample, repaired_code, timeout=60)
        return success, reason

    # ---------- 生成修复 Prompt（核心修改：注入幻觉定义，允许 input()） ----------
    def generate_prompt(self, sample):
        question = sample.get("question", "").strip()
        code = sample.get("original_code", "")

        # ---- 提取幻觉类型并注入完整定义 ----
        hallucination_labels = sample.get("predicted_hallucinations")
        if not hallucination_labels:
            hallucination_labels = sample.get("hallucination_type")
            if hallucination_labels and isinstance(hallucination_labels, str):
                hallucination_labels = [hallucination_labels]
            else:
                hallucination_labels = []
        if not isinstance(hallucination_labels, list):
            hallucination_labels = [hallucination_labels] if hallucination_labels else []

        definitions_text = ""
        if hallucination_labels:
            valid_defs = []
            for label in hallucination_labels:
                if label in hallucination_definitions:
                    valid_defs.append((label, hallucination_definitions[label]))
                elif label in label2id:
                    full_name = full_label_names[label2id[label]]
                    if full_name in hallucination_definitions:
                        valid_defs.append((full_name, hallucination_definitions[full_name]))
                else:
                    for full_name, definition in hallucination_definitions.items():
                        if label in full_name or full_name in label:
                            valid_defs.append((full_name, definition))
                            break
            if valid_defs:
                definitions_text = "\n\n【检测到的幻觉类型及定义】\n"
                for full_name, definition in valid_defs:
                    definitions_text += f"- {full_name}: {definition}\n"

        # ---- 测试用例归一化 ----
        test_raw = sample.get("test_case")
        test_info = ""
        if test_raw:
            test_case = self._normalize_test_case_to_str(test_raw)
            if isinstance(test_case, str):
                test_info = f"\n测试要求：必须通过以下测试用例\n```python\n{test_case}\n```"

        # ---- 构建最终 prompt（允许使用 input()） ----
        prompt = f"""请修复以下有问题的 Python 代码，使其正确实现需求并可通过测试。

【需求描述】
{question}

【待修复代码】
```python
{code}
```
{definitions_text}
{test_info}

【修复要求】
- 仅输出修复后的完整 Python 代码
- 不要添加额外解释
- 可以使用 input()，但必须将所有输入读取逻辑放在函数（如 solve）内部
- 不要在顶层直接调用函数（不要写 solve() 或 main() 调用）
- 不要添加 if __name__ == "__main__"
- 代码末尾添加一行注释说明修复方法

请直接输出代码：
"""
        return prompt

    # ---------- 单样本修复 ----------
    def repair_one(self, sample):
        sample_id = sample.get("sample_id", "unknown")
        print(f"\n🔧 修复样本 {sample_id}")

        hall_labels = sample.get("predicted_hallucinations") or sample.get("hallucination_type")
        if hall_labels:
            print(f"  幻觉类型: {hall_labels}")

        prompt = self.generate_prompt(sample)
        response = self._call_api(prompt)

        if not response:
            sample["repaired"] = False
            sample["final_code"] = ""
            sample["failure_reason"] = "API调用失败"
            return sample

        if "# 修复方法：" in response:
            code_part, method_part = response.rsplit("# 修复方法：", 1)
            repaired_code = self.clean_code(code_part)
            repair_method = method_part.strip()
        else:
            repaired_code = self.clean_code(response)
            repair_method = "未注明"

        success, reason = self.judge_success(sample, repaired_code)

        sample["repaired"] = success
        sample["final_code"] = repaired_code if success else ""
        sample["failure_reason"] = "" if success else reason
        sample["repair_method"] = repair_method

        if success:
            print(f"  ✅ 修复成功")
        else:
            print(f"  ❌ 修复失败: {reason}")

        return sample

    def repair_all(self, samples):
        results = []
        total = len(samples)
        success_cnt = 0
        for i, s in enumerate(samples, 1):
            print(f"\n📋 [{i}/{total}]")
            repaired = self.repair_one(s)
            results.append(repaired)
            if repaired["repaired"]:
                success_cnt += 1
            time.sleep(1)

        print(f"\n📊 修复完成：成功 {success_cnt}/{total}")
        return results

    def save_results(self, samples, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "simple_repair_results.json")
        output_data = []
        for s in samples:
            output_data.append({
                "sample_id": s.get("sample_id"),
                "question": s.get("question"),
                "original_code": s.get("original_code"),
                "repaired": s.get("repaired", False),
                "final_code": s.get("final_code", ""),
                "repair_method": s.get("repair_method", ""),
                "failure_reason": s.get("failure_reason", ""),
                "hallucination_type": s.get("hallucination_type") or s.get("predicted_hallucinations", []),
            })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"📁 结果已保存至 {out_path}")


def main():
    config = {
        "input": "",
        "output_dir": "",  # 修复结果将保存在此目录
        "model_name": "Claude-HaiKu-4.5",     
        "api_key": "",                # 每次运行前请填写有效密钥
        "api_base": "",
        "temperature": 0.3,
        "max_tokens": 4096,
        "timeout": 600
    }

    with open(config["input"], "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    hallucinated = [s for s in all_samples if s.get("has_hallucination", 0) == 1]
    print(f"📊 共 {len(all_samples)} 个样本，有幻觉样本 {len(hallucinated)} 个")

    if not hallucinated:
        print("✅ 无幻觉样本，退出")
        return

    tool = SimpleLLMRepairTool({
        "api_key": config["api_key"],
        "api_base": config["api_base"],
        "model_name": config["model_name"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "timeout": config["timeout"]
    })

    repaired_samples = tool.repair_all(hallucinated)
    tool.save_results(repaired_samples, config["output_dir"])


if __name__ == "__main__":
    main()
