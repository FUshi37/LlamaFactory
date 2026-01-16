#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "Eyjafjalla_data"
OUTPUT_FILE = ROOT / "data" / "eyjafjalla_sft.json"
ADDITIONAL_TEXT_FILE = SOURCE_DIR / "additional_text.txt"

ASSISTANT_SPEAKERS = {
    "艾雅法拉",
    "阿黛尔",
}

USER_SPEAKERS = {
    "博士",
    "前辈",
}

ADDRESS_KEYWORDS = ("前辈", "博士")
INTERACTIVE_KEYWORDS = ("前辈", "博士", "你", "您")

SYSTEM_PROMPT = (
    "你是《明日方舟》角色艾雅法拉（阿黛尔·瑙曼）。"
    "与用户对话时，将用户视为罗德岛博士，称呼对方为“前辈”，"
    "语气温和、克制、认真。"
)

DEFAULT_USER_TEXT = "我在，请继续。"

HEADER_PROMPTS = {
    "任命助理": "今天想听你说说近况。",
    "交谈": "和我聊聊吧。",
    "晋升后交谈": "你晋升后有什么想法？",
    "信赖提升后交谈": "最近过得怎么样？",
    "精英化晋升": "这次晋升对感觉怎么样？",
    "干员报到": "阿黛尔，欢迎加入罗德岛，打个招呼吧。",
    "行动出发": "准备出发前，有什么要交代的吗？",
    "行动开始": "行动开始了，提醒一下大家吧。",
    "作战中": "战斗中有什么发现？",
    "完成高难行动": "行动结束，你有什么想说的？",
    "3星结束行动": "任务完成得不错，说说你的感受。",
    "非3星结束行动": "这次行动还有改进空间，你怎么看？",
    "行动失败": "先调整状态，我们再来。",
    "进驻设施": "对这个房间满意吗？",
    "戳一下": "阿黛尔，你在吗？",
    "信赖触摸": "谢谢你一直在身边。",
    "标题": "你愿意说一句“明日方舟”吗？",
    "新年祝福": "新年快乐",
    "问候": "阿黛尔，早上好。",
    "生日": "阿黛尔，今天是我的生日。",
    "周年庆典": "这段时间多亏你了。",
    "观看作战记录": "刚看了作战记录，你有什么想法？",
    "录音内容": "能和我分享一下这段录音吗？",
    "闲置": "我休息一会儿。",
    "编入队伍": "这次行动就拜托你了。",
    "任命队长": "你来当队长，能行吗？",
    "部署": "准备部署。",
    "选中干员": "我在。",
}

SPEAKER_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff·]+")

VOICE_HEADERS = (
    "语音记录",
    "交谈",
    "晋升后交谈",
    "信赖提升后交谈",
    "精英化晋升",
    "干员报到",
    "行动出发",
    "行动开始",
    "作战中",
    "完成高难行动",
    "3星结束行动",
    "非3星结束行动",
    "行动失败",
    "进驻设施",
    "戳一下",
    "信赖触摸",
    "标题",
    "新年祝福",
    "问候",
    "生日",
    "周年庆典",
    "任命队长",
    "编入队伍",
    "选中干员",
    "部署",
    "观看作战记录",
    "闲置",
)


def is_speaker_line(text: str) -> bool:
    if not text:
        return False
    if len(text) > 12:
        return False
    if any(ch in text for ch in "【】：:，。？！“”\"'()（）[]"):
        return False
    return SPEAKER_RE.fullmatch(text) is not None


def is_voice_header(text: str) -> bool:
    if not text:
        return False
    return text.startswith(VOICE_HEADERS)


def parse_utterances(path: Path):
    utterances = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    voice_mode = False
    voice_texts = []
    current_header = None
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("【录音内容"):
            block = [line]
            i += 1
            while i < len(lines):
                cur = lines[i].strip()
                if not cur:
                    break
                if cur.startswith("【录音内容"):
                    break
                block.append(cur)
                i += 1
            utterances.append(("艾雅法拉", "\n".join(block), "录音内容"))
            continue
        if is_voice_header(line):
            if voice_texts:
                utterances.append(("艾雅法拉", "\n".join(voice_texts), current_header))
                voice_texts = []
            voice_mode = True
            current_header = line
            i += 1
            continue
        if not is_speaker_line(line):
            if voice_mode:
                if line == "[播放]":
                    if voice_texts:
                        utterances.append(("艾雅法拉", "\n".join(voice_texts), current_header))
                        voice_texts = []
                    voice_mode = False
                else:
                    voice_texts.append(line)
                i += 1
                continue
            i += 1
            continue
        speaker = line
        i += 1
        texts = []
        while i < len(lines):
            cur = lines[i].strip()
            if not cur:
                i += 1
                # allow blank lines inside a block
                continue
            if is_voice_header(cur):
                break
            if is_speaker_line(cur):
                break
            if cur == "[播放]":
                i += 1
                continue
            texts.append(cur)
            i += 1
        if texts:
            utterances.append((speaker, "\n".join(texts), None))
        else:
            i += 1
        if voice_mode and voice_texts:
            utterances.append(("艾雅法拉", "\n".join(voice_texts), current_header))
            voice_texts = []
            voice_mode = False
    if voice_mode and voice_texts:
        utterances.append(("艾雅法拉", "\n".join(voice_texts), current_header))
    return utterances


def normalize_user_text(text: str) -> str:
    if not text:
        return ""
    if any(key in text for key in ADDRESS_KEYWORDS):
        return text
    if text.startswith("博士") or text.startswith("前辈"):
        return text
    return text


def prompt_from_header(header: str) -> str:
    if not header:
        return DEFAULT_USER_TEXT
    for key, prompt in HEADER_PROMPTS.items():
        if header.startswith(key):
            return prompt
    return DEFAULT_USER_TEXT


def load_existing_samples(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse existing dataset: {path}")
        return []
    if not isinstance(data, list):
        print(f"Warning: Existing dataset is not a list: {path}")
        return []
    return data


def dedup_samples(samples):
    deduped = []
    seen = set()
    for sample in samples:
        conv = sample.get("conversations", [])
        if len(conv) < 2:
            continue
        human = conv[0].get("value", "").strip()
        gpt = conv[1].get("value", "").strip()
        key = (sample.get("system", ""), human, gpt)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sample)
    return deduped


def build_narrative_samples(path: Path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    samples = []

    def add_sample(user_text: str, assistant_text: str):
        if not assistant_text:
            return
        samples.append(
            {
                "system": SYSTEM_PROMPT,
                "conversations": [
                    {"from": "human", "value": user_text},
                    {"from": "gpt", "value": assistant_text},
                ],
            }
        )

    letter_start = "亲爱的孩子："
    letter_split = "——在一叠厚厚的科研文件中"
    text_after = text
    if letter_start in text:
        letter_section = text[text.index(letter_start) :]
        if letter_split in letter_section:
            letter_body, rest = letter_section.split(letter_split, 1)
            letter_body = letter_body.strip()
            if letter_body:
                add_sample("前辈，我想读一封家书给你听。", letter_body)
            text_after = rest.strip()
        else:
            add_sample("前辈，我想读一封家书给你听。", letter_section.strip())
            text_after = ""

    if text_after:
        text_after = re.sub(
            r"【录音内容[^】]*】.*?(?=【录音内容|\Z)",
            "",
            text_after,
            flags=re.S,
        )

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text_after) if b.strip()]
    buffer = []
    buffer_len = 0
    max_chars = 1200
    for block in blocks:
        if block.startswith("——在一叠厚厚的科研文件中"):
            if buffer:
                add_sample("前辈，我想和你分享一段回忆。", "\n".join(buffer).strip())
                buffer = []
                buffer_len = 0
            continue
        if len(block) < 8:
            continue
        if buffer and buffer_len + len(block) + 1 > max_chars:
            add_sample("前辈，我想和你分享一段回忆。", "\n".join(buffer).strip())
            buffer = [block]
            buffer_len = len(block)
        else:
            buffer.append(block)
            buffer_len += len(block) + 1
    if buffer:
        add_sample("前辈，我想和你分享一段回忆。", "\n".join(buffer).strip())

    return samples


def build_samples():
    samples = []
    for path in sorted(SOURCE_DIR.glob("*.txt")):
        utterances = parse_utterances(path)
        last_non_assistant = None
        last_non_assistant_speaker = None
        for speaker, text, header in utterances:
            if speaker in ASSISTANT_SPEAKERS:
                has_address = any(key in text for key in ADDRESS_KEYWORDS)
                has_interaction = any(key in text for key in INTERACTIVE_KEYWORDS)
                has_user_ctx = (
                    last_non_assistant and last_non_assistant_speaker in USER_SPEAKERS
                )
                if not (has_interaction or has_user_ctx or header):
                    continue
                if has_user_ctx:
                    user_text = normalize_user_text(last_non_assistant)
                else:
                    user_text = prompt_from_header(header)
                if user_text:
                    samples.append(
                        {
                            "system": SYSTEM_PROMPT,
                            "conversations": [
                                {"from": "human", "value": user_text},
                                {"from": "gpt", "value": text},
                            ]
                        }
                    )
                last_non_assistant = None
                last_non_assistant_speaker = None
            else:
                last_non_assistant = text
                last_non_assistant_speaker = speaker
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Build Eyjafjalla SFT dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Output dataset path.",
    )
    parser.add_argument(
        "--merge-existing",
        action="store_true",
        help="Merge with existing output to preserve manual edits.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not SOURCE_DIR.exists():
        raise SystemExit(f"Missing source dir: {SOURCE_DIR}")
    samples = build_samples()
    samples.extend(build_narrative_samples(ADDITIONAL_TEXT_FILE))
    if args.merge_existing and args.output.exists():
        samples = load_existing_samples(args.output) + samples
    samples = dedup_samples(samples)
    if not samples:
        raise SystemExit("No samples generated. Check speaker parsing rules.")
    args.output.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
