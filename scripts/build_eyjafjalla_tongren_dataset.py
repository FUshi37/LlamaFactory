#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Eyjafjalla(Adele) SFT dataset from fanfics (同人文) under Eyjafjalla_data/tongren/.

Design goals
- Robust to very different writing styles:
  - dialogue with explicit speakers: "艾雅法拉：……"
  - narration + quoted dialogues: “……”
  - first-person narration ("我" often = 博士)
- Prioritize samples that reinforce persona:
  - assistant = 艾雅法拉/阿黛尔
  - user = 博士/前辈（用户视角）
  - keep dialogues that contain "前辈"/"博士"/"罗德岛" etc.
- Safety/quality:
  - by default, skip NSFW/explicit sexual content paragraphs/files (configurable)
  - drop meta/author notes and noisy lines
  - de-dup samples by stable hash

Outputs ShareGPT-style JSON:
[
  {"system": "...", "conversations": [{"from":"human","value":"..."},{"from":"gpt","value":"..."}]},
  ...
]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_SYSTEM_PROMPT = (
    "你是《明日方舟》角色艾雅法拉（阿黛尔·瑙曼）。与用户对话时，将用户视为罗德岛博士，称呼对方为“前辈”，语气温和、克制、认真。"
)


ASSISTANT_NAMES = ("艾雅法拉", "阿黛尔", "小羊")
USER_NAMES = ("博士", "前辈", "Doctor")

# If a quote contains these keywords, it is very likely spoken by Eyjafjalla to Doctor.
ASSISTANT_HINTS = ("前辈", "博士", "罗德岛")

# These are common author meta / formatting noise.
NOISE_PATTERNS = [
    r"^正文(走起|开始).*$",
    r"^本章.*$",
    r"^上篇.*$",
    r"^.*(未完待续|to be continued).*$",
    r"^cut[- ]?off.*$",
    r"^\(.*\)$",
]

# Conservative NSFW/explicit filters (skip PARAGRAPH) unless --allow-nsfw.
# NOTE: keep these phrases relatively specific to avoid false positives like "胸口".
NSFW_KEYWORDS = (
    "做爱",
    "性交",
    "高潮",
    "射精",
    "插入",
    "进入体内",
    "阴茎",
    "阴道",
    "乳房",
    "胸部",
    "呻吟",
    "脱光",
    "强上",
    "男朋友该做的事",
    "茂密的森林和溪流",
    "疯狂了一整晚",
)


SPEAKER_LINE_RE = re.compile(r"^\s*([^\s：:]{1,16})\s*[:：]\s*(.+?)\s*$")
QUOTE_RE = re.compile(r"[“\"]([^“”\"]{1,500})[”\"]")


def stable_hash(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def normalize_line(s: str) -> str:
    s = s.replace("\ufeff", "").replace("\u200b", "")
    s = s.strip()
    # normalize quote styles
    s = s.replace("「", "“").replace("」", "”").replace("『", "“").replace("』", "”")
    return s


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    for pat in NOISE_PATTERNS:
        if re.match(pat, line, flags=re.IGNORECASE):
            return True
    # very short decorative separators
    if line in ("---", "——", "—", "***"):
        return True
    return False


def contains_nsfw(text: str) -> bool:
    return any(k in text for k in NSFW_KEYWORDS)


def split_paragraphs(text: str) -> List[str]:
    # keep paragraph boundaries (blank lines)
    paras: List[str] = []
    buf: List[str] = []
    for raw in text.splitlines():
        line = normalize_line(raw)
        if not line:
            if buf:
                paras.append("\n".join(buf).strip())
                buf = []
            continue
        if is_noise_line(line):
            continue
        buf.append(line)
    if buf:
        paras.append("\n".join(buf).strip())
    return paras


@dataclass
class Turn:
    role: str  # "human" or "gpt"
    text: str


def merge_consecutive(turns: Sequence[Turn]) -> List[Turn]:
    out: List[Turn] = []
    for t in turns:
        if not out:
            out.append(Turn(t.role, t.text))
            continue
        if out[-1].role == t.role:
            out[-1].text = (out[-1].text + "\n" + t.text).strip()
        else:
            out.append(Turn(t.role, t.text))
    return out


def guess_quote_role(
    quote: str,
    context_before: str,
    context_after: str,
    file_is_first_person: bool,
) -> Optional[str]:
    q = quote.strip()
    cb = context_before[-200:]
    ca = context_after[:200]

    # Strong signal: "前辈/博士/罗德岛" in quote => assistant (Eyjafjalla) speaking to Doctor
    if any(h in q for h in ASSISTANT_HINTS):
        return "gpt"

    # Attribution hints around quote
    # e.g. "艾雅法拉说：“...”】【阿黛尔轻声道：“...”
    if any(n in cb for n in ASSISTANT_NAMES) and re.search(r"(说|道|问|喊|轻声|低声|笑着)", cb):
        return "gpt"
    if any(n in ca for n in ASSISTANT_NAMES) and re.search(r"(说|道|问|喊|轻声|低声|笑着)", ca):
        return "gpt"

    # If file is first-person doctor narration, un-attributed quotes often belong to doctor/human.
    if file_is_first_person:
        return "human"

    return None


def parse_speaker_lines(paras: Sequence[str]) -> List[Tuple[str, str]]:
    """
    Try parsing explicit 'Speaker: text' lines.
    Returns list of (speaker, text). Empty if not detected robustly.
    """
    utterances: List[Tuple[str, str]] = []
    speaker_hits = 0
    total_lines = 0

    for p in paras:
        for raw_line in p.splitlines():
            total_lines += 1
            m = SPEAKER_LINE_RE.match(raw_line)
            if not m:
                continue
            speaker = m.group(1).strip()
            text = m.group(2).strip()
            if not text:
                continue
            speaker_hits += 1
            utterances.append((speaker, text))

    # consider it "speaker format" only if enough hits
    if speaker_hits >= 8 and speaker_hits >= total_lines * 0.2:
        return utterances
    return []


def parse_quotes(paras: Sequence[str]) -> List[Turn]:
    """
    Parse narration + quoted dialogues. Produces a sequence of Turns with best-effort role guessing.
    Unknown quotes are dropped (conservative).
    """
    full_text = "\n\n".join(paras)
    file_is_first_person = full_text.count("我") >= 30  # heuristic

    turns: List[Turn] = []
    for p in paras:
        if contains_nsfw(p):
            # paragraph contains NSFW; skip it.
            continue

        # extract quotes in order
        for m in QUOTE_RE.finditer(p):
            q = m.group(1).strip()
            if not q:
                continue
            if len(q) < 2:
                continue
            cb = p[: m.start()]
            ca = p[m.end() :]
            role = guess_quote_role(q, cb, ca, file_is_first_person)
            if role is None:
                continue
            turns.append(Turn(role, q))

    return merge_consecutive(turns)


def speaker_to_role(speaker: str, text: str) -> Optional[str]:
    sp = speaker.strip()
    if sp in ASSISTANT_NAMES or any(n in sp for n in ASSISTANT_NAMES):
        return "gpt"
    if sp in USER_NAMES or any(n in sp for n in USER_NAMES):
        return "human"
    # common first-person indicators in explicit speaker format
    if sp in ("我", "博士我", "Doctor我"):
        return "human"
    return None


def build_conversations_from_turns(
    turns: Sequence[Turn],
    min_chars: int,
    max_turns: int,
    max_human_chars: int,
    max_assistant_chars: int,
) -> List[List[dict]]:
    """
    Create ShareGPT 'conversations' arrays.
    Strategy: generate multi-turn snippets up to max_turns, sliding window over turns.
    """
    # require at least one assistant turn and at least one human turn
    if not any(t.role == "gpt" for t in turns):
        return []
    if not any(t.role == "human" for t in turns):
        return []

    # prune too-short assistant turns
    pruned: List[Turn] = []
    for t in turns:
        if t.role == "gpt" and len(t.text) < min_chars:
            continue
        if t.role == "human" and max_human_chars and len(t.text) > max_human_chars:
            continue
        if t.role == "gpt" and max_assistant_chars and len(t.text) > max_assistant_chars:
            continue
        pruned.append(t)
    turns = merge_consecutive(pruned)

    # enforce alternating-ish: drop leading assistant without context by adding a generic prompt
    if turns and turns[0].role == "gpt":
        turns = [Turn("human", "我在，前辈。你想和我说什么？")] + list(turns)

    convs_list: List[List[dict]] = []
    # sliding windows: take contiguous sequences that start with human and end with gpt
    for i in range(len(turns)):
        if turns[i].role != "human":
            continue
        buf: List[Turn] = []
        for j in range(i, len(turns)):
            buf.append(turns[j])
            if len(buf) > max_turns:
                break
            # must end with assistant to be a training sample
            if buf[-1].role != "gpt":
                continue
            # require at least one "前辈" in assistant turns (persona anchor)
            if not any(("前辈" in t.text) for t in buf if t.role == "gpt"):
                continue
            # enforce max lengths per turn (after merging)
            if any(t.role == "human" and max_human_chars and len(t.text) > max_human_chars for t in buf):
                continue
            if any(t.role == "gpt" and max_assistant_chars and len(t.text) > max_assistant_chars for t in buf):
                continue
            conv = [{"from": t.role, "value": t.text} for t in buf]
            convs_list.append(conv)
    return convs_list


def build_samples_from_file(
    path: Path,
    system_prompt: str,
    allow_nsfw: bool,
    min_assistant_chars: int,
    max_turns: int,
    max_human_chars: int,
    max_assistant_chars: int,
) -> Tuple[List[dict], dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    paras = split_paragraphs(text)

    if not paras:
        return [], {"file": str(path), "reason": "empty_after_clean"}

    nsfw_paras = 0
    if not allow_nsfw:
        nsfw_paras = sum(1 for p in paras if contains_nsfw(p))

    # path A: explicit speaker lines
    utt = parse_speaker_lines(paras)
    turns: List[Turn] = []
    if utt:
        for sp, tx in utt:
            role = speaker_to_role(sp, tx)
            if role is None:
                continue
            turns.append(Turn(role, tx))
        turns = merge_consecutive(turns)
    else:
        # path B: quote-based
        turns = parse_quotes(paras)

    convs_list = build_conversations_from_turns(
        turns,
        min_chars=min_assistant_chars,
        max_turns=max_turns,
        max_human_chars=max_human_chars,
        max_assistant_chars=max_assistant_chars,
    )

    # Convert to ShareGPT samples
    samples = [{"system": system_prompt, "conversations": conv} for conv in convs_list]
    rep = {"file": str(path), "turns": len(turns), "samples": len(samples)}
    if not allow_nsfw:
        rep["nsfw_paragraphs_skipped"] = nsfw_paras
    return samples, rep


def load_system_prompt_from_existing(existing_path: Optional[Path]) -> str:
    if not existing_path or not existing_path.exists():
        return DEFAULT_SYSTEM_PROMPT
    try:
        data = json.loads(existing_path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get("system"):
            return str(data[0]["system"])
    except Exception:
        pass
    return DEFAULT_SYSTEM_PROMPT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        type=str,
        default="Eyjafjalla_data/tongren",
        help="Directory containing *.txt fanfics.",
    )
    ap.add_argument(
        "--output",
        type=str,
        default="data/eyjafjalla_tongren_sft.json",
        help="Output ShareGPT JSON file.",
    )
    ap.add_argument(
        "--merge-existing",
        type=str,
        default=None,
        help="If set, de-dup and merge new samples into this existing ShareGPT JSON (safe backup).",
    )
    ap.add_argument(
        "--allow-nsfw",
        action="store_true",
        help="Allow NSFW/explicit content extraction (NOT recommended).",
    )
    ap.add_argument(
        "--min_assistant_chars",
        type=int,
        default=12,
        help="Drop assistant turns shorter than this (to avoid trivial/noisy lines).",
    )
    ap.add_argument(
        "--max_turns",
        type=int,
        default=6,
        help="Max turns per sample (human/gpt alternating). 6 turns = 3 exchanges.",
    )
    ap.add_argument(
        "--max_human_chars",
        type=int,
        default=360,
        help="Drop human turns longer than this to avoid huge blobs.",
    )
    ap.add_argument(
        "--max_assistant_chars",
        type=int,
        default=420,
        help="Drop assistant turns longer than this to avoid overly long quoted chunks.",
    )
    ap.add_argument(
        "--report",
        type=str,
        default="saves/tongren_extraction_report.json",
        help="Write extraction report JSON here.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    report_path = Path(args.report)
    merge_existing = Path(args.merge_existing) if args.merge_existing else None

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    system_prompt = load_system_prompt_from_existing(merge_existing)

    all_samples: List[dict] = []
    report_items: List[dict] = []

    files = sorted(input_dir.glob("*.txt"), key=lambda p: p.name)
    for f in files:
        samples, rep = build_samples_from_file(
            f,
            system_prompt=system_prompt,
            allow_nsfw=bool(args.allow_nsfw),
            min_assistant_chars=int(args.min_assistant_chars),
            max_turns=int(args.max_turns),
            max_human_chars=int(args.max_human_chars),
            max_assistant_chars=int(args.max_assistant_chars),
        )
        report_items.append(rep)
        all_samples.extend(samples)

    # de-dup new samples
    seen = set()
    deduped: List[dict] = []
    for s in all_samples:
        h = stable_hash(s)
        if h in seen:
            continue
        seen.add(h)
        deduped.append(s)
    all_samples = deduped

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    # optional merge
    if merge_existing:
        if merge_existing.exists():
            existing = json.loads(merge_existing.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                raise SystemExit(f"merge-existing is not a JSON list: {merge_existing}")
        else:
            existing = []

        seen2 = set(stable_hash(x) for x in existing)
        merged = list(existing)
        added = 0
        for s in all_samples:
            h = stable_hash(s)
            if h in seen2:
                continue
            merged.append(s)
            seen2.add(h)
            added += 1

        # backup then overwrite
        if merge_existing.exists():
            bak = merge_existing.with_suffix(merge_existing.suffix + ".bak")
            bak.write_text(merge_existing.read_text(encoding="utf-8"), encoding="utf-8")
        merge_existing.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        report_items.append(
            {
                "merge_existing": str(merge_existing),
                "added_into_existing": added,
                "existing_total_after_merge": len(merged),
            }
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "input_dir": str(input_dir),
        "files": len(files),
        "output": str(output_path),
        "new_samples": len(all_samples),
        "allow_nsfw": bool(args.allow_nsfw),
        "max_turns": int(args.max_turns),
        "min_assistant_chars": int(args.min_assistant_chars),
        "max_human_chars": int(args.max_human_chars),
        "max_assistant_chars": int(args.max_assistant_chars),
        "items": report_items,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] files={len(files)} new_samples={len(all_samples)} output={output_path}")
    print(f"[OK] report={report_path}")


if __name__ == "__main__":
    main()

