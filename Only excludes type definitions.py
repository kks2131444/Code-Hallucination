import os
import json
import time
import torch
import numpy as np
import requests
import re
import ast
import traceback
import multiprocessing
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from rank_bm25 import BM25Okapi

# ===================== 全局配置 =====================
device = torch.device("cpu")

# ===================== 数据集 =====================
class LLMRepairTool:
    def __init__(self, llm_config, sample_raw_samples, max_rounds=10):
        self.api_key = llm_config["api_key"]
        self.model_name = llm_config.get("model_name", "")
        self.base_url = llm_config.get("api_base", "")
        self.temperature = llm_config.get("temperature", 0.3)
        self.max_tokens = llm_config.get("max_tokens", 4096)
        self.timeout = llm_config.get("timeout", 400)

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        self.all_case_samples = sample_raw_samples

        self.classifier_model = None
        self.classifier_tokenizer = None
        self.adaptive_thresholds = None

        self.failure_analysis = defaultdict(list)
        self.max_rounds = max_rounds
        self.round_statistics = []

        self.case_documents = []
        self.case_index = []
        self.bm25_index = None
        self.bm25_corpus = []

        self._build_rag_index()

    # ============================================================
    # 1. 完整 RAG：索引构建
    # ============================================================
    def _preprocess_text_for_bm25(self, text):
        if not text:
            return []
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "this", "that", "it", "as", "from"
        }
        return [t for t in tokens if t not in stopwords]

    def _build_case_document(self, sample):
        doc = {
            "question": sample.get("question", ""),
            "original_code": sample.get("original_code", sample.get("src", "")),
            "canonical_solution": sample.get("canonical_solution", ""),
            "flaw_line": sample.get("flaw_line", ""),
            "flaw_line_index": sample.get("flaw_line_index", None),
            "labeling_comments": sample.get("labeling_comments", ""),
            "test_case": sample.get("test_case", None),
        }

        doc["retrieval_text"] = "\n".join([
            f"QUESTION: {doc['question']}",
            f"BUGGY_CODE: {doc['original_code']}",
            f"CANONICAL_SOLUTION: {doc['canonical_solution']}",
            f"FLAW_LINE: {doc['flaw_line']}",
            f"LABELING_COMMENTS: {doc['labeling_comments']}",
        ])
        doc["bm25_tokens"] = self._preprocess_text_for_bm25(doc["retrieval_text"])
        doc["has_enhanced"] = any([
            bool(doc["canonical_solution"]),
            bool(doc["flaw_line"]),
            doc["flaw_line_index"] is not None,
            bool(doc["labeling_comments"]),
        ])
        return doc

    def _build_rag_index(self):
        print("🔍 构建完整 RAG 索引...")
        self.case_documents = []
        self.case_index = []
        self.bm25_corpus = []

        for sample in self.all_case_samples:
            doc = self._build_case_document(sample)
            if not doc["bm25_tokens"]:
                continue

            self.case_documents.append(doc)
            self.case_index.append(sample)
            self.bm25_corpus.append(doc["bm25_tokens"])

        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
            print(f"✅ RAG索引构建完成，共 {len(self.case_documents)} 个有效案例")
        else:
            self.bm25_index = None
            print("⚠️ 无法构建 BM25 索引：无有效案例")

    # ============================================================
    # 2. 完整 RAG：动态 query 构造
    # ============================================================
    def _build_retrieval_query(self, sample, previous_failure_analysis=None, current_code=None):
        question = sample.get("question", "")
        original_code = sample.get("original_code", "")
        current_code = current_code if current_code is not None else original_code

        parts = [
            f"QUESTION: {question}",
            f"ORIGINAL_CODE: {original_code}",
            f"CURRENT_CODE: {current_code}",
        ]

        if previous_failure_analysis and previous_failure_analysis.get("category") not in (None, "initial", "success"):
            parts.extend([
                f"FAILURE_CATEGORY: {previous_failure_analysis.get('category', '')}",
                f"ROOT_CAUSE: {previous_failure_analysis.get('root_cause', '')}",
                f"SUGGESTED_STRATEGY: {previous_failure_analysis.get('suggested_strategy', '')}",
            ])

            failed_assertions = previous_failure_analysis.get("failed_assertions", [])
            for fail in failed_assertions[:3]:
                parts.append(
                    f"ASSERT_FAIL: {fail.get('expression', '')} | ERROR: {fail.get('error', '')}"
                )

        return "\n".join(parts)

    # ============================================================
    # 3. 完整 RAG：初检 + 重排
    # ============================================================
    def _failure_alignment_score(self, previous_failure_analysis, case_doc):
        if not previous_failure_analysis:
            return 0.0

        score = 0.0
        root_cause = str(previous_failure_analysis.get("root_cause", "")).lower()
        strategy = str(previous_failure_analysis.get("suggested_strategy", "")).lower()
        comments = str(case_doc.get("labeling_comments", "")).lower()
        flaw_line = str(case_doc.get("flaw_line", "")).lower()

        for token in self._preprocess_text_for_bm25(root_cause + " " + strategy):
            if token in comments:
                score += 0.05
            if token in flaw_line:
                score += 0.03
        return min(score, 0.4)

    def _rerank_cases(self, retrieved_docs, sample, previous_failure_analysis=None, top_k=3):
        reranked = []

        for item in retrieved_docs:
            doc = item["doc"]
            bm25_score = item["bm25_score"]
            label_score = 0.0   # 固定为0
            enhanced_bonus = 0.15 if doc.get("has_enhanced") else 0.0
            failure_score = self._failure_alignment_score(previous_failure_analysis, doc)
            solution_bonus = 0.08 if doc.get("canonical_solution") else 0.0
            flaw_bonus = 0.05 if doc.get("flaw_line") else 0.0

            final_score = (
                0.55 * bm25_score +
                0.20 * label_score +
                enhanced_bonus +
                failure_score +
                solution_bonus +
                flaw_bonus
            )

            reranked.append({
                "doc": doc,
                "bm25_score": bm25_score,
                "label_score": label_score,
                "failure_score": failure_score,
                "final_score": final_score,
            })

        reranked.sort(key=lambda x: x["final_score"], reverse=True)
        return reranked[:top_k]

    def _fallback_cases(self, top_k=3):
        pool = []
        for doc in self.case_documents:
            if doc.get("has_enhanced"):
                pool.append({
                    "doc": doc,
                    "bm25_score": 0.0,
                    "label_score": 0.0,
                    "failure_score": 0.0,
                    "final_score": 0.2,
                })
        return pool[:top_k]

    def retrieve_relevant_cases(self, sample, previous_failure_analysis=None, current_code=None, initial_top_n=10, top_k=3):
        if not self.bm25_index:
            return self._fallback_cases(top_k)

        query_text = self._build_retrieval_query(sample, previous_failure_analysis, current_code)
        query_tokens = self._preprocess_text_for_bm25(query_text)
        if not query_tokens:
            return self._fallback_cases(top_k)

        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-initial_top_n:][::-1]

        retrieved_docs = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            retrieved_docs.append({
                "doc": self.case_documents[idx],
                "bm25_score": score,
            })

        if not retrieved_docs:
            return self._fallback_cases(top_k)

        return self._rerank_cases(retrieved_docs, sample, previous_failure_analysis, top_k=top_k)

    def _get_relevant_cases_bm25(self, query_text, top_k=3):
        if not self.bm25_index:
            return []

        query_tokens = self._preprocess_text_for_bm25(query_text)
        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            doc = self.case_documents[idx]
            results.append({
                "question": doc.get("question", ""),
                "original_code": doc.get("original_code", ""),
                "BM25相似度": score,
                "canonical_solution": doc.get("canonical_solution", ""),
                "flaw_line": doc.get("flaw_line", ""),
                "flaw_line_index": doc.get("flaw_line_index", None),
                "labeling_comments": doc.get("labeling_comments", ""),
            })
        return results

    # ============================================================
    # 4. API 调用
    # ============================================================
    def _call_api(self, prompt, retries=3, temperature=None, max_tokens=None):
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens

        print(f"📏 Prompt 字符数: {len(prompt)}")
        for attempt in range(retries):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
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

    def clean_repaired_code(self, code):
        if not code:
            return ""
        if "```" in code:
            parts = code.split("```")
            if len(parts) >= 2:
                code = parts[1]
        if code.startswith(("python", "Python")):
            code = "\n".join(code.split("\n")[1:])
        return code.strip()

    # ============================================================
    # 5. 辅助方法：移除顶层函数调用
    # ============================================================
    def _remove_immediate_calls(self, code: str) -> str:
        """移除代码顶层中的独立函数调用（如 solve()、main() 等）"""
        try:
            tree = ast.parse(code)
            new_body = []
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    continue
                new_body.append(node)
            new_tree = ast.Module(body=new_body, type_ignores=[])
            ast.fix_missing_locations(new_tree)
            return ast.unparse(new_tree)
        except Exception:
            return code

    # ============================================================
    # 6. 测试执行
    # ============================================================
    def _normalize_test_case_to_str(self, test_case_raw):
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
                        function_name = None

                    if function_name is None:
                        try:
                            code_tree = ast.parse(code)
                            for node in ast.walk(code_tree):
                                if isinstance(node, ast.FunctionDef):
                                    function_name = node.name
                                    break
                        except Exception:
                            function_name = None

                    if function_name is None:
                        m = re.search(r"def\s+(\w+)\s*\(", code)
                        function_name = m.group(1) if m else "candidate"

                    if function_name not in local_env or not callable(local_env[function_name]):
                        result_queue.put(("function_not_found", f"函数 '{function_name}' 未定义", []))
                        return

                    candidate_func = local_env[function_name]
                    try:
                        num_assertions = len(LLMRepairTool._extract_assertions(test))
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

                assertions = LLMRepairTool._extract_assertions(test)
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

        test_case_raw = sample.get("test_case", None)
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

    # ============================================================
    # 7. Failure analysis
    # ============================================================
    def _llm_analyze_failure(self, failure_reason, sample, repaired_code):
        allowed_categories = [
            "api_error",
            "retrieval_mismatch",
            "test_case_timeout",
            "test_case_assertion_logic",
            "test_case_assertion_edge_case",
            "test_case_assertion_io_mismatch",
            "test_case_general",
            "syntax_error",
            "syntax_indentation",
            "syntax_undefined_name",
            "syntax_import_error",
            "runtime_type_error",
            "runtime_index_error",
            "runtime_attribute_error",
            "runtime_value_error",
            "runtime_general",
            "hallucination_partial",
            "hallucination_persistent",
            "hallucination_wrong_api",
            "hallucination_wrong_algorithm",
            "hallucination_constraint_missed",
            "hallucination_general",
            "unknown"
        ]

        prompt = f'''
你是一名高级代码修复失败分析专家。

请根据以下信息，对本轮修复失败进行精细诊断，并输出 JSON：

[失败原因]
{failure_reason}

[修复后代码]
{repaired_code}

[样本问题]
{sample.get('question', '')}

你的任务：
1. 从 allowed_categories 中选择一个最准确的 category
2. 给出 root_cause（中文，具体、可定位）
3. 给出 suggested_strategy（中文，必须是下一轮可执行修复策略）
4. 如果 failure_reason 或 repaired_code 表明“案例检索方向错了”，可选 retrieval_mismatch

allowed_categories = {allowed_categories}

输出要求：
- 只输出一个 JSON
- 不要输出解释文字
- suggested_strategy 必须可以直接被下一轮 prompt 使用

JSON格式：
{{
  "category": "...",
  "root_cause": "...",
  "suggested_strategy": "..."
}}
'''
        llm_output = self._call_api(prompt, temperature=0.0, max_tokens=800)
        if not llm_output:
            return None

        start = llm_output.find("{")
        end = llm_output.rfind("}")
        if start != -1 and end != -1 and end > start:
            llm_output = llm_output[start:end + 1]

        try:
            data = json.loads(llm_output)
        except Exception as e:
            print(f"⚠️ LLM failure analysis JSON解析失败：{e} | 原始输出：{llm_output}")
            return None

        category = data.get("category", "unknown")
        if category not in allowed_categories:
            category = "unknown"

        return {
            "category": category,
            "root_cause": data.get("root_cause", "LLM 未返回 root_cause").strip() or "LLM 未返回 root_cause",
            "suggested_strategy": data.get("suggested_strategy", "LLM 未返回 suggested_strategy").strip() or "LLM 未返回 suggested_strategy",
        }

    def _analyze_failure_reason(self, sample, round_num, failure_reason, repair_method, repaired_code="", validation_info=None):
        sample_id = sample["sample_id"]
        analysis_result = None
        try:
            analysis_result = self._llm_analyze_failure(failure_reason, sample, repaired_code)
        except Exception as e:
            print(f"⚠️ 样本{sample_id} 第{round_num}轮 failure analysis 异常：{e}")

        if analysis_result is None:
            analysis_result = {
                "category": "unknown",
                "root_cause": failure_reason[:200],
                "suggested_strategy": "请根据失败原因、错误类型和断言失败信息进行针对性修复。"
            }

        if validation_info and "failed_assertions" in validation_info:
            analysis_result["failed_assertions"] = validation_info.get("failed_assertions", [])

        failure_record = {
            "sample_id": sample_id,
            "round": round_num,
            "failure_type": analysis_result["category"],
            "failure_reason": failure_reason,
            "root_cause": analysis_result["root_cause"],
            "suggested_strategy": analysis_result["suggested_strategy"],
            "repair_method": repair_method,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.failure_analysis[analysis_result["category"]].append(failure_record)

        print(f"    📊 深度失败分析（{analysis_result['category']}）")
        print(f"    🔎 root_cause：{analysis_result['root_cause']}")
        print(f"    💡 suggested_strategy：{analysis_result['suggested_strategy']}")
        return analysis_result

    # ============================================================
    # 8. Prompt（修改：允许 input()，禁止顶层调用）
    # ============================================================
    def _format_retrieved_cases_for_prompt(self, reranked_cases):
        if not reranked_cases:
            return "### 三、相关参考案例\n（未检索到可用案例）"

        lines = ["### 三、相关参考案例（完整 RAG 检索 + 重排）"]
        for i, item in enumerate(reranked_cases, 1):
            doc = item["doc"]
            lines.append(f"\n**案例{i}**")
            lines.append(f"- 最终相关度: {item['final_score']:.4f}")
            lines.append(f"- BM25分数: {item['bm25_score']:.4f}")
            lines.append(f"- 标签匹配分: {item['label_score']:.4f}")
            lines.append(f"- 失败对齐分: {item['failure_score']:.4f}")
            lines.append(f"- 需求: {doc.get('question', '')[:180]}")
            lines.append(f"- 错误代码片段:\n```python\n{doc.get('original_code', '')[:400]}\n```")
            if doc.get("canonical_solution"):
                lines.append(f"- 正确代码参考:\n```python\n{doc.get('canonical_solution', '')[:400]}\n```")
            if doc.get("flaw_line"):
                lines.append(f"- 错误位置: 第{doc.get('flaw_line_index', '?')}行 | {doc.get('flaw_line', '')}")
            labeling_comments = doc.get("labeling_comments", "")
            labeling_comments = "" if labeling_comments is None else str(labeling_comments)
            if labeling_comments:
                lines.append(f"- 标签注释: {labeling_comments[:200]}")
        return "\n".join(lines)

    def _generate_failure_guidance(self, failure_analysis, round_num):
        if failure_analysis.get("category") == "initial":
            return "### 五、修复指导（首次修复）\n请先根据参考案例，优先修复最核心错误。"
        if failure_analysis.get("category") == "success":
            return "### 五、修复指导\n上一轮已成功，本轮无需额外指导。"

        lines = [
            f"### 五、智能修复指导（基于第{round_num - 1}轮失败分析）",
            f"失败类别：{failure_analysis.get('category', 'unknown')}",
            f"根本原因：{failure_analysis.get('root_cause', '未知')}",
            f"建议策略：{failure_analysis.get('suggested_strategy', '请根据失败信息修复')}",
        ]
        failed_assertions = failure_analysis.get("failed_assertions", [])
        if failed_assertions:
            lines.append("失败断言：")
            for fail in failed_assertions[:3]:
                lines.append(f"- assert {fail.get('expression', '')[:80]} | error: {fail.get('error', '')[:120]}")
        lines.append("请务必严格执行 suggested_strategy，避免重复之前的失败模式。")
        return "\n".join(lines)

    def generate_repair_prompt(self, sample, round_num, previous_failure_analysis):
        field_definitions = """
    ### 三、参考案例字段说明
    你会看到若干参考案例，每个案例可能包含以下字段：

    - canonical_solution：该案例对应的参考正确实现，可用于学习正确逻辑模式，但不要机械照抄，必须结合当前题目进行迁移。
    - flaw_line：该案例中原错误代码的关键错误行，表示最值得优先修复的局部位置。
    - flaw_line_index：flaw_line 在原代码中的大致行号，可帮助你定位错误位置。
    - labeling_comments：人工标注或分析说明，解释该案例为什么出错、属于什么错误模式，以及修复时应关注什么。

    请注意：
    1. 参考案例是修复依据，不是直接复制对象。
    2. 若 canonical_solution 与当前题目逻辑相近，应学习其正确模式，而不是逐字照搬。
    3. 若 flaw_line / flaw_line_index / labeling_comments 指向了与当前代码相同的错误模式，应优先修复对应位置。
    4. 你必须结合当前题目的 Question、当前代码、失败分析和测试断言综合修复。
    """

        base_round_num = round_num - 1
        current_code_key = f"round_{base_round_num}_code"
        current_code = sample.get(current_code_key, sample.get("original_code", ""))
        sample[current_code_key] = current_code

        reranked_cases = self.retrieve_relevant_cases(
            sample=sample,
            previous_failure_analysis=previous_failure_analysis,
            current_code=current_code,
            initial_top_n=10,
            top_k=3,
        )
        cases_text = self._format_retrieved_cases_for_prompt(reranked_cases)
        failure_guidance_section = self._generate_failure_guidance(previous_failure_analysis, round_num)

        test_case_str = self._normalize_test_case_to_str(sample.get("test_case", ""))
        test_assertions = LLMRepairTool._extract_assertions(test_case_str) if isinstance(test_case_str, str) else []
        test_info = ""
        if test_assertions:
            test_info = "\n\n### 六、测试断言要求（必须全部通过）\n```python\n"
            for assertion in test_assertions:
                test_info += f"assert {assertion}\n"
            test_info += "```"

        return f'''
你需要修复一段存在代码幻觉问题的 Python 代码。

### 一、原始需求（Question）
{sample.get('question', '').strip()}

{field_definitions}

{cases_text}

### 四、当前待修复代码（第{base_round_num}轮版本）
```python
{current_code}
```
{failure_guidance_section}

### 五、修复要求

严格围绕 Question 修复，不要偏题。

必须结合参考案例中的：错误代码、正确代码、错误位置、标签注释。

必须优先修复上一轮 failure analysis 指出的 root_cause。

必须严格执行 suggested_strategy，而不是泛泛修改。

最小化修改，但必须保证通过全部测试。

若案例中的 canonical_solution 可迁移，请优先复用其正确模式。

若案例中的 flaw_line / labeling_comments 指向了同类错误，请针对性修正对应逻辑。

{test_info}

【硬性约束】
- 可以使用 input()，但必须将所有输入读取逻辑放在函数（如 solve 或 main）内部
- 严禁在顶层直接调用函数（不要写 solve() 或 main() 调用）
- 不要添加 if __name__ == "__main__"
- 只输出纯 Python 代码，不要输出 Markdown 代码块
- 代码末尾必须加一行注释：# 修复方法：xxx

当前轮次：第{round_num}轮（共{self.max_rounds}轮）
'''.strip()

    # ============================================================
    # 9. 轮次记录
    # ============================================================
    def _record_round_result(self, sample, round_num, success, reason, repaired_code, repair_method, start_time, failure_analysis, validation_info):
        round_duration = time.time() - start_time
        round_info = {
            "round_number": round_num,
            "success": success,
            "failure_reason": reason if not success else "",
            "failure_analysis": failure_analysis if not success else {},
            "validation_info": validation_info,
            "repaired_code": repaired_code,
            "repair_method": repair_method,
            "duration_seconds": round_duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        sample.setdefault("repair_history", []).append(round_info)

    def _record_round_success(self, round_num, sample_id):
        while len(self.round_statistics) < round_num:
            self.round_statistics.append({
                "round": len(self.round_statistics) + 1,
                "success_count": 0,
                "failure_count": 0,
                "success_samples": [],
                "failure_samples": []
            })
        self.round_statistics[round_num - 1]["success_count"] += 1
        self.round_statistics[round_num - 1]["success_samples"].append(sample_id)

    def _record_round_failure(self, round_num, sample_id):
        while len(self.round_statistics) < round_num:
            self.round_statistics.append({
                "round": len(self.round_statistics) + 1,
                "success_count": 0,
                "failure_count": 0,
                "success_samples": [],
                "failure_samples": []
            })
        self.round_statistics[round_num - 1]["failure_count"] += 1
        self.round_statistics[round_num - 1]["failure_samples"].append(sample_id)

    def print_round_statistics_table(self, total_samples):
        print("\n📊 各轮修复统计：")
        print("=" * 60)
        print(f"总样本数: {total_samples}")
        for stat in self.round_statistics:
            print(
                f"第{stat['round']}轮: "
                f"成功 {stat['success_count']} | "
                f"失败 {stat['failure_count']}"
            )
        print("=" * 60)

    # ============================================================
    # 10. 修复主流程
    # ============================================================
    def repair_single_sample(self, sample):
        sample_id = sample["sample_id"]
        print(f"\n🔧 开始修复样本 {sample_id}（最多{self.max_rounds}轮）")

        sample["repair_history"] = []
        sample["repaired"] = False
        sample["success_round"] = -1
        sample["final_failure_reason"] = ""
        sample.setdefault("round_0_code", sample.get("original_code", ""))

        previous_failure_analysis = {
            "category": "initial",
            "root_cause": "首次修复尝试",
            "suggested_strategy": "结合参考案例、错误位置与标签注释，优先修复最核心逻辑错误。"
        }

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n  📍 样本{sample_id} - 第{round_num}轮修复开始...")
            round_start_time = time.time()

            prompt = self.generate_repair_prompt(sample, round_num, previous_failure_analysis)
            repaired_content = self._call_api(prompt)

            if not repaired_content:
                fail_reason = "API调用失败，无法获取修复代码"
                failure_analysis = self._analyze_failure_reason(sample, round_num, fail_reason, "无", "", validation_info={})
                previous_failure_analysis = failure_analysis
                sample[f"round_{round_num}_code"] = sample.get(f"round_{round_num - 1}_code", sample["round_0_code"])
                self._record_round_result(sample, round_num, False, fail_reason, sample[f"round_{round_num}_code"], "无", round_start_time, failure_analysis, {})
                self._record_round_failure(round_num, sample_id)
                if round_num == self.max_rounds:
                    sample["final_failure_reason"] = fail_reason
                continue

            if "# 修复方法：" in repaired_content:
                code_part, method_part = repaired_content.rsplit("# 修复方法：", 1)
                repaired_code = self.clean_repaired_code(code_part)
                repair_method = method_part.strip()
            else:
                repaired_code = self.clean_repaired_code(repaired_content)
                repair_method = "未明确说明修复方法"

            success, reason, validation_info = self.judge_repair_success_with_timeout(sample, repaired_code, timeout=60)

            if success:
                sample[f"round_{round_num}_code"] = repaired_code
                sample["repair_method"] = repair_method
                self._record_round_result(
                    sample, round_num, True, reason, repaired_code, repair_method, round_start_time,
                    {"category": "success", "root_cause": "修复成功", "suggested_strategy": ""}, validation_info
                )
                sample["repaired"] = True
                sample["success_round"] = round_num
                sample["final_failure_reason"] = ""
                self._record_round_success(round_num, sample_id)
                print(f"  ✅ 样本{sample_id}在第{round_num}轮修复成功！")
                break

            failure_analysis = self._analyze_failure_reason(
                sample, round_num, reason, repair_method, repaired_code, validation_info=validation_info
            )
            previous_failure_analysis = failure_analysis
            print(f"    ❌ 第{round_num}轮修复失败：{reason}")

            sample[f"round_{round_num}_code"] = sample.get(f"round_{round_num - 1}_code", sample["round_0_code"])

            self._record_round_result(
                sample, round_num, False, reason, sample[f"round_{round_num}_code"], repair_method,
                round_start_time, failure_analysis, validation_info
            )
            self._record_round_failure(round_num, sample_id)

            if round_num == self.max_rounds:
                sample["final_failure_reason"] = reason

        return sample

    def repair_all_samples_independently(self, test_classified_samples):
        print(f"\n🚀 开始独立修复 {len(test_classified_samples)} 个样本（最多{self.max_rounds}轮）")
        print("=" * 60)
        self.round_statistics = []
        all_repaired_samples = []

        for i, sample in enumerate(test_classified_samples):
            print(f"\n📋 处理样本 {i+1}/{len(test_classified_samples)} (ID: {sample['sample_id']})")
            print("-" * 50)

            query_text = f"{sample.get('question', '')} {sample.get('original_code', '')}"
            retrieved_cases = self._get_relevant_cases_bm25(query_text, top_k=3)
            if retrieved_cases:
                print(f"🔍 BM25检索到 {len(retrieved_cases)} 个相关案例")
                for j, case in enumerate(retrieved_cases, 1):
                    print(f"  案例{j}: 相似度: {case.get('BM25相似度', 0):.3f}")
            else:
                print("⚠️ BM25未检索到相关案例，使用回退模式")

            repaired_sample = self.repair_single_sample(sample)
            all_repaired_samples.append(repaired_sample)
            time.sleep(2)

        self.print_round_statistics_table(len(test_classified_samples))
        return all_repaired_samples

    def get_failure_analysis_summary(self):
        summary = {}
        for failure_type, records in self.failure_analysis.items():
            summary[failure_type] = {
                "count": len(records),
                "common_repair_methods": defaultdict(int),
                "assertion_failure_counts": defaultdict(int)
            }
            for record in records:
                rm = record["repair_method"]
                if rm and rm != "无":
                    summary[failure_type]["common_repair_methods"][rm] += 1
                if "断言" in record["failure_reason"]:
                    m = re.search(r'(\d+)个断言失败', record["failure_reason"])
                    if m:
                        summary[failure_type]["assertion_failure_counts"][int(m.group(1))] += 1
        return summary

    def save_detailed_repair_results(self, all_repaired_samples, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        detail_path = os.path.join(output_dir, "repair_sample_details_with_history.json")
        sample_details = []

        for sample in all_repaired_samples:
            if sample["repaired"] and sample["success_round"] >= 0:
                final_code = sample.get(f"round_{sample['success_round']}_code", "")
            else:
                final_code = ""
                for round_num in range(self.max_rounds, -1, -1):
                    round_code = sample.get(f"round_{round_num}_code", None)
                    if round_code is not None:
                        final_code = round_code
                        break

            sample_detail = {
                "sample_id": sample["sample_id"],
                "question": sample["question"],
                "original_code": sample["original_code"],
                "repair_final_result": {
                    "repaired": sample["repaired"],
                    "success_round": sample["success_round"],
                    "final_failure_reason": sample.get("final_failure_reason", ""),
                    "final_code": final_code,
                    "final_repair_method": sample.get("repair_method", "无")
                },
                "repair_history": sample.get("repair_history", []),
                "test_case": sample.get("test_case", "无")
            }
            sample_details.append(sample_detail)

        with open(detail_path, 'w', encoding='utf-8') as f:
            json.dump(sample_details, f, ensure_ascii=False, indent=2)

        failure_analysis_path = os.path.join(output_dir, "enhanced_repair_failure_analysis.json")
        failure_summary = self.get_failure_analysis_summary()
        success_distribution_by_round = defaultdict(int)

        for sample in all_repaired_samples:
            if sample["repaired"]:
                success_distribution_by_round[f"round_{sample['success_round']}"] += 1

        failure_analysis = {
            "summary": {
                "total_samples": len(all_repaired_samples),
                "successful_repairs": sum(1 for s in all_repaired_samples if s["repaired"]),
                "failed_repairs": sum(1 for s in all_repaired_samples if not s["repaired"]),
                "success_rate": f"{sum(1 for s in all_repaired_samples if s['repaired']) / len(all_repaired_samples) * 100:.2f}%" if all_repaired_samples else "0%",
                "failure_type_distribution": {ft: data["count"] for ft, data in failure_summary.items()}
            },
            "failure_details": [],
            "intelligent_failure_analysis": failure_summary,
            "success_distribution_by_round": dict(success_distribution_by_round),
            "round_statistics": self.round_statistics
        }

        for sample in all_repaired_samples:
            if not sample["repaired"]:
                failure_info = {
                    "sample_id": sample["sample_id"],
                    "final_failure_reason": sample.get("final_failure_reason", ""),
                    "repair_history_summary": [
                        {
                            "round": hist["round_number"],
                            "failed_reason": hist["failure_reason"],
                            "repair_method": hist["repair_method"]
                        }
                        for hist in sample.get("repair_history", []) if not hist["success"]
                    ],
                }
                failure_analysis["failure_details"].append(failure_info)

        with open(failure_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(failure_analysis, f, ensure_ascii=False, indent=2)

        print(f"\n📊 修复结果统计：")
        print(f"  - 总样本数：{failure_analysis['summary']['total_samples']}")
        print(f"  - 修复成功：{failure_analysis['summary']['successful_repairs']} ({failure_analysis['summary']['success_rate']})")
        print(f"  - 修复失败：{failure_analysis['summary']['failed_repairs']}")
        print(f"\n📁 结果文件已保存：")
        print(f"  - 样本修复详情：{detail_path}")
        print(f"  - 增强失败分析报告：{failure_analysis_path}")

        return failure_analysis['summary']['successful_repairs'], failure_analysis['summary']['failed_repairs']


def normalize_case_samples_for_rag(samples):
    fixed_code_field = 0
    fixed_id_field = 0
    fixed_comments = 0

    for s in samples:
        if "sample_id" not in s and "index" in s:
            s["sample_id"] = s["index"]
            fixed_id_field += 1

        if not s.get("original_code") and s.get("src"):
            s["original_code"] = s["src"]
            fixed_code_field += 1

        lc = s.get("labeling_comments", "")
        if isinstance(lc, dict):
            s["labeling_comments"] = " | ".join([str(v) for v in lc.values() if v])
            fixed_comments += 1
        elif isinstance(lc, list):
            s["labeling_comments"] = " | ".join([str(v) for v in lc if v])
            fixed_comments += 1
        else:
            s["labeling_comments"] = "" if lc is None else str(lc)

    print(
        f"[RAG案例库清洗] sample_id补全: {fixed_id_field}, "
        f"original_code补全: {fixed_code_field}, "
        f"labeling_comments清洗: {fixed_comments}"
    )
    return samples


# ===================== 主函数 =====================
def main():
    config = {
        "input": "",
        "output_dir": "",
        "repair_case_data": "",
        "model_name": "",          # 可修改
        "api_key": "",                # 每次运行前请填写有效密钥
        "api_base": "",
        "temperature": 0.3,
        "max_tokens": 4096,
        "timeout": 600,
        "max_rounds": 10
    }

    os.makedirs(config["output_dir"], exist_ok=True)

    with open(config["input"], 'r', encoding='utf-8') as f:
        classified_samples = json.load(f)

    hallucinated_samples = [s for s in classified_samples if s.get("has_hallucination", 0) == 1]
    print(f"📊 分类结果中共 {len(classified_samples)} 个样本，其中有幻觉样本 {len(hallucinated_samples)} 个")

    if not hallucinated_samples:
        print("✅ 没有需要修复的样本")
        return

    with open(config["repair_case_data"], "r", encoding="utf-8") as f:
        repair_case_raw_samples = json.load(f)

    repair_case_raw_samples = normalize_case_samples_for_rag(repair_case_raw_samples)

    llm_config = {
        "api_key": config["api_key"],
        "api_base": config["api_base"],
        "model_name": config["model_name"],
        "temperature": config["temperature"],
        "max_tokens": config["max_tokens"],
        "timeout": config["timeout"]
    }

    repair_tool = LLMRepairTool(
        llm_config=llm_config,
        sample_raw_samples=repair_case_raw_samples,
        max_rounds=config["max_rounds"]
    )

    all_repaired_samples = repair_tool.repair_all_samples_independently(hallucinated_samples)
    repair_tool.save_detailed_repair_results(all_repaired_samples, config["output_dir"])

    print("\n🎉 修复流程全部完成！")


if __name__ == "__main__":
    main()
