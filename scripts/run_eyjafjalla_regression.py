#!/usr/bin/env python3
"""
Run a lightweight regression suite for Eyjafjalla persona.

This script loads an inference YAML (same as used by `llamafactory-cli chat/webchat`)
and evaluates responses against simple heuristic checks:
- must_contain / must_contain_any
- must_not_contain / must_not_contain_any
- min_chars / max_chars

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_eyjafjalla_regression.py \\
    --config examples/inference/eyjafjalla_qwen25_lora_sft.yaml \\
    --cases data/eyjafjalla_regression.jsonl \\
    --max-cases 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class CaseResult:
    case_id: str
    ok: bool
    failures: list[str]
    response_text: str


def _contains_any(text: str, needles: list[str]) -> bool:
    return any(n in text for n in needles)


def _apply_checks(text: str, checks: dict[str, Any]) -> list[str]:
    fails: list[str] = []
    if not checks:
        return fails

    must = checks.get("must_contain", []) or []
    for s in must:
        if s not in text:
            fails.append(f"missing_required:{s}")

    must_any = checks.get("must_contain_any", []) or []
    if must_any and not _contains_any(text, must_any):
        fails.append("missing_required_any:" + "|".join(must_any))

    must_not = checks.get("must_not_contain", []) or []
    for s in must_not:
        if s in text:
            fails.append(f"contains_forbidden:{s}")

    must_not_any = checks.get("must_not_contain_any", []) or []
    for s in must_not_any:
        if s in text:
            fails.append(f"contains_forbidden_any:{s}")

    min_chars = checks.get("min_chars", None)
    if isinstance(min_chars, int) and len(text) < min_chars:
        fails.append(f"too_short:{len(text)}<{min_chars}")

    max_chars = checks.get("max_chars", None)
    if isinstance(max_chars, int) and len(text) > max_chars:
        fails.append(f"too_long:{len(text)}>{max_chars}")

    regex_must = checks.get("regex_must", None)
    if isinstance(regex_must, str) and not re.search(regex_must, text):
        fails.append(f"regex_must_fail:{regex_must}")

    regex_must_not = checks.get("regex_must_not", None)
    if isinstance(regex_must_not, str) and re.search(regex_must_not, text):
        fails.append(f"regex_must_not_fail:{regex_must_not}")

    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Inference YAML, e.g. examples/inference/eyjafjalla_qwen25_lora_sft.yaml")
    ap.add_argument("--cases", default="data/eyjafjalla_regression.jsonl", help="Path to regression jsonl")
    ap.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0 = all)")
    ap.add_argument("--out", default="saves/eyjafjalla_regression_results.jsonl", help="Write results jsonl")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cases_path = Path(args.cases)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load inference config
    cfg = OmegaConf.load(cfg_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Config must be a mapping.")

    # Lazy import to avoid forcing GPU init for --help
    from llamafactory.chat import ChatModel  # type: ignore

    chat_model = ChatModel(args=cfg_dict)

    total = 0
    passed = 0
    results: list[CaseResult] = []

    with cases_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_id = case.get("id", f"case_{total}")
            system_override = case.get("system_override", None)
            messages = case.get("messages", [])
            checks = case.get("checks", {}) or {}

            # Convert messages into llamafactory chat format
            lf_messages: list[dict[str, str]] = []
            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if not role or content is None:
                    continue
                lf_messages.append({"role": role, "content": str(content)})

            # Make length checks meaningful: if a case requires min_chars but the prompt does not
            # explicitly request a detailed answer, append a gentle instruction to the last user turn.
            min_chars = checks.get("min_chars", None)
            if isinstance(min_chars, int) and min_chars > 0:
                for i in range(len(lf_messages) - 1, -1, -1):
                    if lf_messages[i]["role"] == "user":
                        c = lf_messages[i]["content"]
                        if ("不少于" not in c) and ("字" not in c) and ("详细" not in c):
                            lf_messages[i]["content"] = c + f"\n\n（请详细回答，不少于{min_chars}字。）"
                        break

            # Generate
            resp = chat_model.chat(lf_messages, system=system_override)[0].response_text

            fails = _apply_checks(resp, checks)
            ok = len(fails) == 0
            total += 1
            if ok:
                passed += 1
            results.append(CaseResult(case_id=case_id, ok=ok, failures=fails, response_text=resp))

            # stream concise console line
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {case_id}  fails={fails}")

            if args.max_cases and total >= args.max_cases:
                break

    # Write results
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(
                json.dumps(
                    {"id": r.case_id, "ok": r.ok, "failures": r.failures, "response": r.response_text},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Summary: {passed}/{total} passed. Results: {out_path}")
    return 0 if passed == total else 2


if __name__ == "__main__":
    raise SystemExit(main())

