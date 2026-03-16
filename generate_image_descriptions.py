import argparse
import asyncio
import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_PROMPT_VERSION = "2026-03-16-v1"

# 모델별 API 가격 ($/1M tokens). 최신 가격: https://openai.com/pricing
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4.1":      {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o":       {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15, "output":  0.60},
}


@dataclass(frozen=True)
class ImageTask:
    category: str
    image_path: Path


@dataclass
class Stats:
    pending_total: int
    started_at: datetime
    input_price_per_1m: float
    output_price_per_1m: float
    success_count: int = 0
    failed_count: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0

    def completed(self) -> int:
        return self.success_count + self.failed_count

    def total_cost(self) -> float:
        return (
            self.input_tokens_total / 1_000_000 * self.input_price_per_1m
            + self.output_tokens_total / 1_000_000 * self.output_price_per_1m
        )

    def progress_str(self) -> str:
        done = self.completed()
        pct = done / self.pending_total * 100 if self.pending_total else 0.0
        cost = self.total_cost()
        elapsed = (datetime.now() - self.started_at).total_seconds()
        if done > 0 and elapsed > 0:
            remaining_secs = (self.pending_total - done) / (done / elapsed)
            eta = (datetime.now() + timedelta(seconds=remaining_secs)).strftime("%H:%M:%S")
            est_total_cost = cost / done * self.pending_total
            est_str = f"${est_total_cost:.4f}"
        else:
            eta = "--:--:--"
            est_str = "-.----"
        return (
            f"{done}/{self.pending_total} ({pct:.1f}%) | "
            f"누적 ${cost:.4f} | 예상총비용 {est_str} | ETA {eta}"
        )


def load_dotenv_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if value and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Extract detailed image descriptions for every image under images/ using gpt-4o-mini."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=root_dir / "images",
        help="Top-level images directory containing category subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "descriptions",
        help="Directory where category JSONL outputs will be written.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI vision-capable model name.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per image on transient failures.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay after each successful request.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=None,
        help="Only process the specified category. Repeat to include multiple categories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N discovered images after filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run descriptions even if an image already exists in the output JSONL.",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help="Version tag stored in the output records.",
    )
    return parser.parse_args()


def discover_image_tasks(images_dir: Path, categories: set[str] | None) -> list[ImageTask]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    tasks: list[ImageTask] = []
    for category_dir in sorted(path for path in images_dir.iterdir() if path.is_dir()):
        if categories and category_dir.name not in categories:
            continue

        for image_path in sorted(category_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                tasks.append(ImageTask(category=category_dir.name, image_path=image_path))
    return tasks


def build_processed_index(output_dir: Path) -> set[str]:
    processed: set[str] = set()
    if not output_dir.exists():
        return processed

    for jsonl_path in output_dir.glob("*.jsonl"):
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                relative_path = row.get("relative_path")
                if isinstance(relative_path, str):
                    processed.add(relative_path)
    return processed


def encode_image(image_path: Path) -> tuple[str, str]:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        suffix = image_path.suffix.lower()
        mime_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "application/octet-stream")

    with image_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return mime_type, encoded


def parse_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def build_messages(category: str, image_name: str) -> list[dict[str, Any]]:
    system_prompt = (
        "You are a visual archivist describing Korean cultural and everyday-life images. "
        "Return only valid JSON. Do not add markdown fences or extra commentary."
    )
    user_prompt = f"""
다음 이미지를 매우 자세하게 설명하세요.

카테고리: {category}
파일명: {image_name}

반드시 아래 JSON 스키마만 반환하세요.
{{
  "summary_ko": "한두 문장 요약",
  "detailed_description_ko": "장면, 인물, 사물, 행동, 배경, 분위기, 색감, 구도, 텍스트 정보를 포함한 4~8문장 상세 설명",
  "main_subjects": ["핵심 대상 1", "핵심 대상 2"],
  "actions_or_events": ["보이는 행동 또는 상황"],
  "background_and_setting": "장소와 배경 설명",
  "style_and_mood": "사진/일러스트/포스터 여부와 분위기 설명",
  "visible_text": ["이미지 안에서 읽히는 텍스트가 있으면 그대로"],
  "cultural_context": ["한국 문화나 맥락상 중요한 요소"],
  "notable_details": ["작지만 중요한 디테일"]
}}

규칙:
- 설명은 한국어로 작성합니다.
- 보이는 사실 중심으로 작성하고, 불확실한 내용은 추측임을 드러내지 말고 과도한 단정도 피합니다.
- 텍스트가 잘 보이지 않으면 visible_text는 빈 배열로 둡니다.
- JSON 외 다른 텍스트는 절대 출력하지 않습니다.
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


async def request_description(
    client: AsyncOpenAI,
    model: str,
    task: ImageTask,
    request_timeout: float,
) -> tuple[dict[str, Any], int, int]:
    mime_type, encoded_image = encode_image(task.image_path)
    messages = build_messages(task.category, task.image_path.name)
    messages[1]["content"].append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"},
        }
    )

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        ),
        timeout=request_timeout,
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise TypeError(f"Unexpected response type: {type(content)!r}")
    usage = response.usage
    return parse_json_object(content), usage.prompt_tokens, usage.completion_tokens


def append_jsonl_row(output_path: Path, row: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def log(message: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


async def process_one_image(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    task: ImageTask,
    root_dir: Path,
    output_dir: Path,
    model: str,
    prompt_version: str,
    max_retries: int,
    request_timeout: float,
    delay_seconds: float,
    stats: Stats,
) -> tuple[str, str, str | None]:
    relative_path = task.image_path.relative_to(root_dir).as_posix()
    output_path = output_dir / f"{task.category}.jsonl"

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                description, in_tok, out_tok = await request_description(
                    client=client,
                    model=model,
                    task=task,
                    request_timeout=request_timeout,
                )
                row = {
                    "category": task.category,
                    "file_name": task.image_path.name,
                    "relative_path": relative_path,
                    "prompt_version": prompt_version,
                    "model": model,
                    "description": description,
                }
                append_jsonl_row(output_path, row)
                stats.input_tokens_total += in_tok
                stats.output_tokens_total += out_tok
                stats.success_count += 1
                log(
                    f"Completed: {relative_path} | "
                    f"in={in_tok:,} out={out_tok:,} | {stats.progress_str()}",
                    level="OK",
                )
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
                return ("success", relative_path, None)
            except Exception as exc:
                if attempt == max_retries:
                    error_row = {
                        "category": task.category,
                        "file_name": task.image_path.name,
                        "relative_path": relative_path,
                        "prompt_version": prompt_version,
                        "model": model,
                        "error": str(exc),
                    }
                    append_jsonl_row(output_dir / "_errors.jsonl", error_row)
                    stats.failed_count += 1
                    log(
                        f"Failed: {relative_path} | error={exc} | {stats.progress_str()}",
                        level="FAIL",
                    )
                    return ("failed", relative_path, str(exc))
                await asyncio.sleep(min(2 ** (attempt - 1), 8))

    return ("failed", relative_path, "Unreachable state")


async def async_main(args: argparse.Namespace) -> int:
    root_dir = Path(__file__).resolve().parent
    load_dotenv_file(root_dir / ".env")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()

    categories = set(args.category) if args.category else None
    tasks = discover_image_tasks(args.images_dir, categories)

    if args.limit is not None:
        tasks = tasks[: args.limit]

    if not tasks:
        log("No images found to process.")
        return 0

    if args.overwrite:
        pending_tasks = tasks
    else:
        processed = build_processed_index(output_dir)
        pending_tasks = [
            task
            for task in tasks
            if task.image_path.relative_to(root_dir).as_posix() not in processed
        ]

    if not pending_tasks:
        log("All matching images are already processed.")
        return 0

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is required.")

    pricing = MODEL_PRICING.get(args.model, {"input": 0.0, "output": 0.0})
    stats = Stats(
        pending_total=len(pending_tasks),
        started_at=started_at,
        input_price_per_1m=pricing["input"],
        output_price_per_1m=pricing["output"],
    )
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))
    log(
        (
            "Description extraction started. "
            f"started_at={started_at.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"total={len(tasks)}, pending={len(pending_tasks)}, skipped={len(tasks) - len(pending_tasks)}, "
            f"concurrency={max(1, args.max_concurrency)}, model={args.model}"
        )
    )
    jobs = [
        process_one_image(
            semaphore=semaphore,
            client=client,
            task=task,
            root_dir=root_dir,
            output_dir=output_dir,
            model=args.model,
            prompt_version=args.prompt_version,
            max_retries=max(1, args.max_retries),
            request_timeout=args.request_timeout,
            delay_seconds=max(0.0, args.delay_seconds),
            stats=stats,
        )
        for task in pending_tasks
    ]

    for future in asyncio.as_completed(jobs):
        await future

    finished_at = datetime.now()
    log(
        (
            "Description extraction finished. "
            f"started_at={started_at.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"finished_at={finished_at.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"elapsed={format_elapsed((finished_at - started_at).total_seconds())}, "
            f"success={stats.success_count}, failed={stats.failed_count}, "
            f"skipped={len(tasks) - len(pending_tasks)}, "
            f"total_input_tokens={stats.input_tokens_total:,}, "
            f"total_output_tokens={stats.output_tokens_total:,}, "
            f"total_cost=${stats.total_cost():.4f}"
        )
    )
    return 0 if stats.failed_count == 0 else 1


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())