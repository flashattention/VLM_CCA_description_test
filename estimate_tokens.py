"""
estimate_tokens.py

generate_image_descriptions.py 실행 시 소요되는 토큰 수 및 비용을 추정합니다.
샘플 이미지에서 실제 completions 호출을 통해 입력·출력 토큰을 동시에 측정하고
전체 이미지 수로 외삽합니다.

  - usage.prompt_tokens  : system 메시지 + user 텍스트 프롬프트 + 이미지 토큰 합산
  - usage.completion_tokens: 모델이 생성한 JSON 출력 토큰
  - 최소/최대 기준 외삽은 합계 토큰 기준으로 최솟값·최댓값에 해당하는
    실제 이미지의 입출력 쌍을 사용합니다.

사용법:
    python estimate_tokens.py                    # 기본값: 샘플 10장
    python estimate_tokens.py --sample-size 20   # 샘플 20장
"""

import argparse
import asyncio
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from generate_image_descriptions import (
    ImageTask,
    build_messages,
    discover_image_tasks,
    encode_image,
    load_dotenv_file,
)

# ─── 모델별 토큰 가격 ($/1M tokens) ────────────────────────────────────────────
# 최신 가격은 https://openai.com/pricing 에서 확인하세요.
# { 모델명: {"input": 입력가격, "output": 출력가격} }
MODELS: dict[str, dict[str, float]] = {
    "gpt-4.1":      {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}



async def count_tokens(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    task: ImageTask,
) -> tuple[str, int, int]:
    """
    실제 completions 호출을 통해 (파일명, 입력토큰, 출력토큰) 반환.
    실패 시 토큰값은 -1.
    입력·출력 토큰을 한 번의 API 호출로 동시에 측정합니다.
    """
    async with semaphore:
        try:
            mime_type, encoded = encode_image(task.image_path)
            messages = build_messages(task.category, task.image_path.name)
            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            usage = response.usage
            return (task.image_path.name, usage.prompt_tokens, usage.completion_tokens)
        except Exception as exc:
            print(f"  [WARN] {task.image_path.name}: {exc}")
            return (task.image_path.name, -1, -1)


def sample_tasks(all_tasks: list[ImageTask], n: int, seed: int) -> list[ImageTask]:
    """카테고리별로 균등하게 n장을 샘플링합니다."""
    rng = random.Random(seed)
    by_cat: dict[str, list[ImageTask]] = defaultdict(list)
    for t in all_tasks:
        by_cat[t.category].append(t)

    # 카테고리별 셔플
    for lst in by_cat.values():
        rng.shuffle(lst)

    sampled: list[ImageTask] = []
    categories = list(by_cat.keys())
    i = 0
    while len(sampled) < n and any(by_cat[c] for c in categories):
        cat = categories[i % len(categories)]
        if by_cat[cat]:
            sampled.append(by_cat[cat].pop())
        i += 1
    return sampled[:n]


def separator(char: str = "─", width: int = 64) -> str:
    return char * width


async def estimate_for_model(
    client: AsyncOpenAI,
    model: str,
    prices: dict[str, float],
    sampled_tasks: list[ImageTask],
    total_count: int,
    concurrency: int,
) -> None:
    price_in  = prices["input"]
    price_out = prices["output"]

    print(f"\n{separator('━')}")
    print(f"  모델: {model}  (입력 ${price_in:.2f} / 출력 ${price_out:.2f}  per 1M tokens)")
    print(separator('━'))

    semaphore = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *[count_tokens(semaphore, client, model, t) for t in sampled_tasks]
    )

    valid = [(name, inp, out) for name, inp, out in results if inp >= 0]
    if not valid:
        print("  유효한 결과 없음 — API 키 또는 모델명을 확인하세요.")
        return

    in_list  = [inp for _, inp, _  in valid]
    out_list = [out for _, _,  out in valid]

    avg_in  = sum(in_list)  / len(in_list)
    avg_out = sum(out_list) / len(out_list)

    # 합계 토큰 기준으로 최솟값·최댓값 이미지를 찾아 그 쌍을 사용
    # (입력/출력을 독립적으로 min/max하면 실제 존재하지 않는 조합이 됨)
    by_total = sorted(valid, key=lambda x: x[1] + x[2])
    mn_name, mn_in, mn_out = by_total[0]
    mx_name, mx_in, mx_out = by_total[-1]

    print(f"\n  이미지별 토큰 수 ({len(valid)}장 샘플):")
    print(f"  {'파일명':<45} {'입력':>8} {'출력':>8} {'합계':>8}")
    print(f"  {separator('-', 72)}")
    for name, inp, out in sorted(valid, key=lambda x: -(x[1] + x[2])):
        print(f"  {name:<45} {inp:>8,} {out:>8,} {inp+out:>8,}")

    print(f"\n  샘플 통계:")
    print(f"  {'':10} {'입력 토큰':>12} {'출력 토큰':>12} {'합계':>12}  {'기준 이미지'}")
    print(f"  {separator('-', 80)}")
    print(f"  {'평균':<10} {avg_in:>12,.1f} {avg_out:>12,.1f} {avg_in+avg_out:>12,.1f}")
    print(f"  {'최소':<10} {mn_in:>12,} {mn_out:>12,} {mn_in+mn_out:>12,}  {mn_name}")
    print(f"  {'최대':<10} {mx_in:>12,} {mx_out:>12,} {mx_in+mx_out:>12,}  {mx_name}")

    print(f"\n  전체 {total_count:,}장 외삽:")
    hdr = f"  {'':10} {'입력 토큰':>14} {'출력 토큰':>14} {'합계 토큰':>14}  {'입력 비용':>10} {'출력 비용':>10} {'총 비용':>10}"
    print(hdr)
    print(f"  {separator('-', len(hdr) - 2)}")
    for label, bi, bo in [
        ("평균 기준", avg_in, avg_out),
        ("최소 기준", mn_in,  mn_out),
        ("최대 기준", mx_in,  mx_out),
    ]:
        ti = int(bi * total_count)
        to = int(bo * total_count)
        tt = ti + to
        ci = (ti / 1_000_000) * price_in
        co = (to / 1_000_000) * price_out
        ct = ci + co
        print(f"  {label:<10} {ti:>14,} {to:>14,} {tt:>14,}  ${ci:>9,.2f} ${co:>9,.2f} ${ct:>9,.2f}")


async def async_main(args: argparse.Namespace) -> None:
    load_dotenv_file(_ROOT / ".env")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY 환경변수가 필요합니다.")

    all_tasks = discover_image_tasks(args.images_dir, categories=None)
    total = len(all_tasks)

    sampled = sample_tasks(all_tasks, min(args.sample_size, total), args.seed)

    print(separator('═'))
    print("  토큰 사용량 추정기")
    print(separator('═'))
    print(f"  전체 이미지 수   : {total:,}")
    print(f"  샘플 수          : {len(sampled)}")
    print(f"  동시 요청 수     : {args.concurrency}")
    print(f"  랜덤 시드        : {args.seed}")
    print(f"  가격 출처        : https://openai.com/pricing (수동 확인 필요)")

    client = AsyncOpenAI(api_key=api_key)

    for model, prices in MODELS.items():
        await estimate_for_model(
            client=client,
            model=model,
            prices=prices,
            sampled_tasks=sampled,
            total_count=total,
            concurrency=args.concurrency,
        )

    print(f"\n{separator('═')}")
    print("  완료.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="generate_image_descriptions.py 실행 시 토큰 수 및 비용을 추정합니다."
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="토큰 측정에 사용할 샘플 이미지 수 (기본값: 10)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="동시 API 요청 수 (기본값: 5)",
    )
    parser.add_argument(
        "--images-dir", type=Path, default=_ROOT / "images",
        help="카테고리 하위 디렉터리가 있는 최상위 images 폴더",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="재현 가능한 샘플링을 위한 랜덤 시드 (기본값: 42)",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
