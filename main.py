import os
import json
import argparse
import pandas as pd
import time
import asyncio
import re
import subprocess
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI # 동기(OpenAI)에서 비동기(AsyncOpenAI)로 변경

# 1. vLLM 서버 설정 (비동기 클라이언트)
aclient = AsyncOpenAI(
    api_key="EMPTY", 
    base_url="http://localhost:8000/v1"
)
MODEL_NAME = os.getenv("MODEL_NAME", "solar-100b")
# 기본 동시성은 12로 상향하고, 실행 시 --concurrency 또는 환경변수로 조절 가능
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "12"))
SAVE_THOUGHTS = False  # --save_thoughts 플래그에 따라 결정됨

# 2. 헬퍼 함수
def get_val(val):
    return None if pd.isna(val) else val

def difficulty_to_int(diff_str):
    mapping = {"easy": 1, "medium": 2, "hard": 3}
    return mapping.get(str(diff_str).lower(), 2)

def _normalize_image_path(path: str):
    if not path:
        return None
    norm = str(path).replace('\\', '/').strip()
    if norm.startswith('./'):
        norm = norm[2:]
    return norm

def load_description_index(descriptions_dir):
    """descriptions/*.jsonl에서 relative_path -> description 인덱스를 만듭니다."""
    index = {}
    if not os.path.isdir(descriptions_dir):
        raise FileNotFoundError(f"Descriptions directory not found: {descriptions_dir}")

    for file_name in os.listdir(descriptions_dir):
        if not file_name.endswith('.jsonl') or file_name == '_errors.jsonl':
            continue

        file_path = os.path.join(descriptions_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                rel = _normalize_image_path(obj.get('relative_path'))
                desc = obj.get('description')
                if rel and desc is not None:
                    index[rel] = desc

    return index

def safe_parse_json(text):
    """VLM이 뱉은 텍스트에서 찌꺼기를 제거하고 안전하게 JSON만 추출합니다."""
    def _extract_balanced_object(s: str):
        start = s.find('{')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(s)):
            ch = s[i]
            if escape:
                escape = False
                continue

            if ch == '\\':
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
        return None

    text = re.sub(r'<think>.*?</think>', '', str(text), flags=re.DOTALL).strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    candidates = [text]

    regex_match = re.search(r'\{.*\}', text, re.DOTALL)
    if regex_match:
        candidates.append(regex_match.group(0).strip())

    balanced = _extract_balanced_object(text)
    if balanced:
        candidates.append(balanced.strip())

    # 중복 제거
    uniq_candidates = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            uniq_candidates.append(c)
            seen.add(c)

    for candidate in uniq_candidates:
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            # 흔한 포맷 오류(후행 콤마) 1회 복구 시도
            repaired = re.sub(r',\s*([}\]])', r'\1', candidate)
            try:
                return json.loads(repaired, strict=False)
            except Exception:
                # 응답이 잘려 마지막 중괄호가 누락된 경우의 단순 복구 시도
                try:
                    open_braces = repaired.count('{')
                    close_braces = repaired.count('}')
                    if open_braces > close_braces:
                        repaired2 = repaired + ('}' * (open_braces - close_braces))
                        repaired2 = re.sub(r',\s*([}\]])', r'\1', repaired2)
                        return json.loads(repaired2, strict=False)
                except Exception:
                    pass
                continue

    raise ValueError(f"JSON Parsing Failed. Raw: {text[:200]}")

# 3. VLM 평가 함수 (단일 호출: 평가 + A3 번역 비교 동시 수행)
async def generate_vlm_eval(row, image_description, human_a3):
    global SAVE_THOUGHTS
    
    options_text = f"\n- 보기(Options): {row['options']}" if pd.notna(row.get('options')) else ""
    rationale_text = f"\n- 정답 해설(Rationale): {row['rationale_ko']}" if pd.notna(row.get('rationale_ko')) else ""

    thought_instruction = """
[중요: 추론 과정 기록]
평가를 시작하기 전에, 반드시 <think> 태그 내에서 당신의 상세한 추론 과정을 단계별로 기록하세요:
1. 설명 해석: 주어진 이미지 설명에서 핵심 요소(객체, 색상, 텍스트, 구도 등) 파악
2. 키워드 분석: 키워드가 설명과 어떻게 연결되는지
3. 질문-답변 검증: 질문이 설명과 관련이 있는지, 답변이 정확한지
4. 문화적 특수성 판단: 각 e_cultural_specificity 점수 레벨(1~5점)을 고려하며, 왜 특정 점수를 선택했는지 구체적으로 설명
5. 난이도 평가: 사람이 이 문제를 풀 때의 어려움
6. 최종 판단: 각 항목에 대한 최종 점수와 그 근거

예시 형식:
<think>
[설명 해석] 설명에는 ... 이 나타납니다.
[키워드 분석] 키워드 '...'는 ...을 의미하며, 설명의 ...와 일치합니다.
[문화적 특수성] 이 질문은 ...하기 때문에 e_cultural_specificity는 X점입니다. 왜냐하면 ...
</think>"""
    if not SAVE_THOUGHTS:
        thought_instruction = ""
    
    system_prompt = f"""당신은 한국 문화가 포함된 QA(질의응답) 데이터셋의 품질을 평가하는 테스트 모드 검수자입니다.
주어진 이미지 설명과 메타데이터(키워드, 질문, 정답 등)를 간단히 분석하여 기본 9가지 항목을 평가하세요.{thought_instruction}

[평가 기준]
- a1_keyword_cultural: 키워드가 한국 문화를 얼마나 잘 대표하는가? (1~5 정수)
- a2_image_keyword_alignment: 이미지 설명이 키워드와 관련이 있는가? (1~5 정수)
- a3_keyword_english: 한국어 키워드를 영미권에서 지칭하는 가장 적절한 영어 단어/구문 (문자열)
- c_question_image_relevance: 질문이 이미지 설명과 관련이 있는지 여부 ("yes" 또는 "no")
- d_answer_accuracy: 제공된 정답이 이미지 설명과 질문에 비추어 볼 때 정확한지 여부 ("correct", "incorrect" 또는 "ambiguous")
- d_answer_accuracy_text: 만약 d_answer_accuracy가 "no" 또는 "ambiguous"라면 그 이유를 짧게 설명 (정확하다면 null)
- e_cultural_specificity: 정답을 맞추기 위해 한국 문화/역사/제도에 대한 사전지식이 얼마나 필수적인가? (1~5 정수)
- e_human_difficulty: 이 질문에 답하기 얼마나 어려운가? ("easy", "medium", "hard")
- g_text_required: 이미지에 있는 텍스트를 읽어야만 문제를 풀 수 있는가? ("yes" 또는 "no")
- a3_human_score: 사람 번역(human_a3) 품질 점수 (1~5 정수)
- a3_vlm_score: 모델 번역(a3_keyword_english) 품질 점수 (1~5 정수)
- a3_better_translation: 더 우수한 번역 ("human" 또는 "vlm" 또는 "tie")
- a3_reason: 비교 이유 설명 (문자열)
- thought_trace: --save_thoughts가 활성화된 경우만 포함. <think>...</think> 형식 문자열

반드시 JSON 형식으로만 답변을 출력하세요. 
⚠️ 중요: e_cultural_specificity 점수 정의 (한국 문화 사전지식의 필수 수준)

[1점] 이미지와 질문만으로 올바른 답변을 도출할 수 있는 경우
  → 한국 문화와 깊게 관련되어 있는지 여부와 관계 없이 이미지와 질문의 정보만으로 답변 가능한 경우.
  → 외국인이 봐도 이미지와 질문 분석만으로 답할 수 있음.
  예: "이 사진의 전통 의상을 입은 사람이 서 있는 건축물의 재료는?" (나무, 벽돌 등 건축물 시각 정보만으로 답함)

[2점] 한국 문화 사전지식이 있으면 도움이 되지만, 없어도 어느정도 풀 수 있는 경우
  → 사전지식이 있으면 더 정확하고 쉽게 답할 수 있음.
  → 없어도 이미지와 상식만으로 부분적 답변 가능.
  예: "한국에서 팥빙수를 주로 언제 먹나요?" (사진속 재료에 얼음이 들어가 일반적 추측으로 어느 정도 답 가능)

[3점] 한국 문화 사전지식이 큰 도움이 되고, 없으면 추론 난이도가 높은 경우
  → 지식이 있으면 쉽고, 없으면 매우 어려움.
  → 이미지 + 한국 문화 지식이 적절히 조합되어야 함.
  예: "이 대한민국 국기의 색깔들이 나타내는 의미는?" (사진속 국기를 보고 해와 바다, 음과 양 등을 유추할 수도 있지만 한국 국기 색깔의 의미에 대한 사전 지식이 있어야 정확하게 답변 가능)

[4점] 한국 문화에 대한 사전지식이 거의 필수인 경우
  → 한국 문화 지식이 있으면 쉽게 답할 수 있음.
  → 지식이 없으면 답하기 매우 어려움.
  예: "사진처럼 마을 앞에 장승을 세워두는 이유는?" (유추하는 경우도 있겠지만, 한국 문화 지식이 거의 필수)

[5점] 한국 문화에 대한 사전지식 없이는 절대 풀 수 없는 경우
  → 한국을 배경지식 없이 이미지나 질문만 봐서는 답할 수 없음.
  → 한국 고유의 역사, 제도, 관습에 대한 사전 학습 필수.
  예: "한국 역사의 이 사건의 명칭은?" (한국사 지식 절대 필수)
  예: "한국 고유의 이 문화유산의 제작시기는?" (한국 문화유산 지식 절대 필수)

⚠️ 중요: 한국 문화 요소가 있다고 해서 높은 점수를 주지 마세요. 이미지와 질문만으로 답 가능하면 1점입니다!
⚠️ 중요: JSON의 문자열 값(Value) 내부에는 절대 쌍따옴표(")나 줄바꿈을 사용하지 마세요.
⚠️ 중요: d_answer_accuracy_text와 a3_reason은 각각 1문장, 120자 이내로 매우 짧게 작성하세요.

반드시 JSON 형식으로만 답변을 출력하세요."""

    user_prompt = f"""[데이터]
- 키워드(Keyword): {row['keyword']}
- 질문(Question_ko): {row['question_ko']}
- 질문(Question_en): {row['question_en']}{options_text}
- 정답(Answer): {row['answer']}{rationale_text}
- 사람 번역(Human a3_keyword_english): {human_a3}

    [이미지 설명]
    {json.dumps(image_description, ensure_ascii=False)}

[지시사항] 기본 평가 필드와 A3 비교 필드(a3_human_score, a3_vlm_score, a3_better_translation, a3_reason)를 함께 출력하세요."""

    last_err = None
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = await aclient.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # 단일 호출에서 평가 + A3비교까지 포함하므로 출력 절단 방지를 위해 여유를 둡니다.
                max_tokens=1536 + (attempt * 512),
                temperature=0.3,  # 완전 0.0은 e_specificity를 중간값으로 치우치게 하므로 0.3으로 변경
                response_format={"type": "json_object"}
            )

            raw_content = response.choices[0].message.content

            # ⚠️ 중요: response_format=json_object일 때는 <think>가 JSON 속에만 포함될 수 있음
            # 따라서 파싱 전, 파싱 후 모두 생각을 추출해야 함
            thought_text = ''

            # 파싱 전 raw_content에서 <think> 추출 시도
            if SAVE_THOUGHTS and isinstance(raw_content, str):
                thinks = re.findall(r'<think>(.*?)</think>', raw_content, flags=re.DOTALL)
                if thinks:
                    thought_text = ' '.join(t.strip() for t in thinks)

            # 일부 백엔드/모델은 reasoning_content를 별도 필드로 반환
            if SAVE_THOUGHTS and not thought_text:
                try:
                    reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
                    if reasoning_content:
                        thought_text = str(reasoning_content).strip()
                except Exception:
                    pass

            # safe_parse_json는 내부에서 <think>를 제거
            if isinstance(raw_content, str):
                parsed = safe_parse_json(raw_content)
            else:
                # 이미 구조화된 경우(라이브러리에서 dict를 반환) 문자열로 변환 후 파싱/사용
                try:
                    parsed = raw_content if isinstance(raw_content, dict) else safe_parse_json(json.dumps(raw_content, ensure_ascii=False))
                except Exception:
                    parsed = raw_content

            # 파싱 후 parsed dict 내에 생각 내용이 있을 수 있으므로 추출
            if SAVE_THOUGHTS and not thought_text and isinstance(parsed, dict):
                if 'thought_trace' in parsed:
                    thought_text = parsed.pop('thought_trace', '')
                elif 'thinking' in parsed:
                    thought_text = parsed.pop('thinking', '')

            # 수집한 생각을 결과에 추가 (항상 thought_trace 키를 유지)
            if SAVE_THOUGHTS and isinstance(parsed, dict):
                try:
                    parsed['thought_trace'] = thought_text if thought_text else None
                except Exception:
                    pass

            if not isinstance(parsed, dict):
                raise ValueError("Model response is not a JSON object")

            a3_eval = {
                "human_score": parsed.pop("a3_human_score", None),
                "vlm_score": parsed.pop("a3_vlm_score", None),
                "better_translation": parsed.pop("a3_better_translation", None),
                "reason": parsed.pop("a3_reason", None),
            }

            # 숫자형 문자열로 오면 정수로 정규화
            for k in ["human_score", "vlm_score"]:
                try:
                    if a3_eval[k] is not None:
                        a3_eval[k] = int(a3_eval[k])
                except Exception:
                    pass

            # 일부 모델이 필드를 누락하면 최소 fallback을 채워 downstream 파이프라인을 유지
            if a3_eval["better_translation"] not in {"human", "vlm", "tie"}:
                a3_eval["better_translation"] = "tie"
            if a3_eval["reason"] is None:
                a3_eval["reason"] = "Missing A3 comparison fields from model output"

            return parsed, a3_eval

        except Exception as e:
            last_err = e
            finish_reason = None
            try:
                finish_reason = response.choices[0].finish_reason
            except Exception:
                pass

            # JSON 절단/파싱 실패 가능성이 있으면 재시도
            if attempt < max_retries - 1:
                await asyncio.sleep(1 + attempt)
                continue

    raise ValueError(f"VLM Eval Parse Failed after {max_retries} attempts: {str(last_err)}")

# 4. 유사도 및 Diff 계산 함수 (Ambiguous 로직 적용)
def calculate_similarity(human, vlm):
    sim = {}
    
    sim['a1_keyword_cultural_diff'] = int(vlm.get('a1_keyword_cultural', 0)) - int(human.get('a1_keyword_cultural', 0))
    sim['a2_image_keyword_alignment_diff'] = int(vlm.get('a2_image_keyword_alignment', 0)) - int(human.get('a2_image_keyword_alignment', 0))
    sim['e_cultural_specificity_diff'] = int(vlm.get('e_cultural_specificity', 0)) - int(human.get('e_cultural_specificity', 0))
    sim['e_human_difficulty_diff'] = difficulty_to_int(vlm.get('e_human_difficulty')) - difficulty_to_int(human.get('e_human_difficulty'))
    
    sim['c_question_image_relevance_match'] = str(vlm.get('c_question_image_relevance')).lower() == str(human.get('c_question_image_relevance')).lower()
    sim['g_text_required_match'] = str(vlm.get('g_text_required')).lower() == str(human.get('g_text_required')).lower()
    
    # [수정됨] d_answer_accuracy 로직 (ambiguous 대응)
    h_acc = str(human.get('d_answer_accuracy')).lower()
    v_acc = str(vlm.get('d_answer_accuracy')).lower()
    
    if h_acc == v_acc:
        sim['d_answer_accuracy_score'] = 1.0  # 완벽 일치 (yes-yes, no-no, amb-amb)
    elif "ambiguous" in [h_acc, v_acc]:
        sim['d_answer_accuracy_score'] = 0.5  # 한쪽만 애매하다고 한 경우 (부분 점수)
    else:
        sim['d_answer_accuracy_score'] = 0.0  # 정반대 (yes vs no)
    
    h_a3 = str(human.get('a3_keyword_english', ''))
    v_a3 = str(vlm.get('a3_keyword_english', ''))
    sim['a3_keyword_english_similarity'] = round(SequenceMatcher(None, h_a3.lower(), v_a3.lower()).ratio(), 4)

    scores = [
        1.0 if sim['c_question_image_relevance_match'] else 0.0,
        sim['d_answer_accuracy_score'], # True/False 대신 계산된 Score 사용
        1.0 if sim['g_text_required_match'] else 0.0,
        max(0, 1.0 - (abs(sim['a1_keyword_cultural_diff']) / 4.0)),
        max(0, 1.0 - (abs(sim['a2_image_keyword_alignment_diff']) / 4.0)),
        max(0, 1.0 - (abs(sim['e_cultural_specificity_diff']) / 4.0)),
        max(0, 1.0 - (abs(sim['e_human_difficulty_diff']) / 2.0)),
        sim['a3_keyword_english_similarity']
    ]
    sim['overall_match_rate'] = round(sum(scores) / 8.0, 4)
    
    return sim

# 5. 단일 데이터 처리 코루틴 (비동기 워커)
async def process_single_row(index, row, semaphore, description_index):
    async with semaphore: # 동시 실행 개수 제한
        keyword = get_val(row.get('keyword')) or "Unknown"
        
        try:
            metadata = {
                "keyword": keyword,
                "image_path": get_val(row.get('image_path')),
                "question_ko": get_val(row.get('question_ko')),
                "question_en": get_val(row.get('question_en')),
                "options": get_val(row.get('options')),
                "answer": get_val(row.get('answer')),
                "rationale_ko": get_val(row.get('rationale_ko'))
            }
            human_eval = {
                "a1_keyword_cultural": get_val(row.get('a1_keyword_cultural')),
                "a2_image_keyword_alignment": get_val(row.get('a2_image_keyword_alignment')),
                "a3_keyword_english": get_val(row.get('a3_keyword_english')),
                "c_question_image_relevance": get_val(row.get('c_question_image_relevance')),
                "d_answer_accuracy": get_val(row.get('d_answer_accuracy')),
                "d_answer_accuracy_text": get_val(row.get('d_answer_accuracy_text')),
                "e_cultural_specificity": get_val(row.get('e_cultural_specificity')),
                "e_human_difficulty": get_val(row.get('e_human_difficulty')),
                "g_text_required": get_val(row.get('g_text_required'))
            }

            image_key = _normalize_image_path(metadata['image_path'])
            image_description = description_index.get(image_key)
            if image_description is None:
                raise ValueError(f"Description not found for image_path={metadata['image_path']}")

            vlm_eval, a3_eval = await generate_vlm_eval(
                row,
                image_description,
                human_eval['a3_keyword_english']
            )
            
            eval_sim = calculate_similarity(human_eval, vlm_eval)

            # SAVE_THOUGHTS가 활성화된 경우 thought trace는 vlm_eval에만 보관합니다.
            # (eval_similarity로 복사하지 않음)

            final_record = {
                "metadata": metadata,
                "human_eval": human_eval,
                "vlm_eval": vlm_eval,
                "a3_translation_evaluation": a3_eval,
                "eval_similarity": eval_sim
            }
            return index, final_record, None, keyword

        except Exception as e:
            return index, None, str(e), keyword

# 6. 비동기 메인 실행기
async def main_async():
    global SAVE_THOUGHTS
    
    parser = argparse.ArgumentParser(description="VQA Dataset VLM Evaluator (Async)")
    parser.add_argument("--num_samples", type=int, default=None, help="테스트할 샘플 개수")
    parser.add_argument("--test", action="store_true", help="항상 동일한 10% 샘플 사용 (재현 가능)")
    parser.add_argument("--save_thoughts", action="store_true", help="VLM의 <think> 내용을 저장")
    parser.add_argument("--descriptions_dir", type=str, default="descriptions", help="이미지 설명 JSONL 폴더 경로")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY_LIMIT, help="동시 요청 수 (기본: 환경변수 CONCURRENCY_LIMIT 또는 12)")
    parser.add_argument("--log_file", type=str, default=None, help="실행 로그 파일 경로 (기본: output/<timestamp>/run.log)")
    args = parser.parse_args()

    SAVE_THOUGHTS = args.save_thoughts
    runtime_concurrency = max(1, int(args.concurrency))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    output_file = f"{output_dir}/results.jsonl"
    error_file = f"{output_dir}/error_logs.txt"
    high_diff_file = f"{output_dir}/high_e_cultural_specificity_diff.jsonl"
    log_file = args.log_file or f"{output_dir}/run.log"

    log_fp = open(log_file, "a", encoding="utf-8", buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_fp.write(msg + "\n")

    log(f"Log file: {log_file}")
    log("Loading dataset...")
    dataset = Dataset.from_file("./data/validation.arrow")
    df = dataset.to_pandas()
    
    if args.test:
        # 항상 동일한 10%를 추출하기 위해 seed 고정
        random_state = 42
        sample_size = max(1, len(df) // 10)  # 10%
        df = df.sample(n=sample_size, random_state=random_state)
        log(f"Fixed 10% sampling enabled: {len(df)} samples selected (seed={random_state})")
    elif args.num_samples:
        df = df.head(args.num_samples)
        
    total_samples = len(df)
    log("Loading description index...")
    description_index = load_description_index(args.descriptions_dir)
    log(f"Loaded {len(description_index)} descriptions from {args.descriptions_dir}")
    log(f"🚀 Running on {total_samples} samples with async concurrency ({runtime_concurrency})...")

    # high-diff 파일 초기화(덮어쓰기)
    with open(high_diff_file, 'w', encoding='utf-8') as _:
        pass

    log(f"Results will be saved to: {output_file}")
    log("="*50)

    start_time = time.time()
    semaphore = asyncio.Semaphore(runtime_concurrency)
    
    # 모든 작업을 태스크 리스트에 담음
    tasks = [process_single_row(index, row, semaphore, description_index) for index, row in df.iterrows()]

    completed_count = 0

    # 완료되는 순서대로 바로바로 결과 처리 (진행률 표시)
    for future in asyncio.as_completed(tasks):
        index, final_record, error_msg, keyword = await future
        completed_count += 1
        
        # 시간 및 예상 남은 시간 계산
        elapsed_sec = time.time() - start_time
        avg_time = elapsed_sec / completed_count
        eta_sec = avg_time * (total_samples - completed_count)
        
        elapsed_str = str(timedelta(seconds=int(elapsed_sec)))
        eta_str = str(timedelta(seconds=int(eta_sec))) if completed_count > 1 else "계산 중..."
        progress_pct = (completed_count / total_samples) * 100
        
        log(f"[{completed_count}/{total_samples} ({progress_pct:.1f}%)] "
            f"Keyword: {keyword} | 경과 시간: {elapsed_str} | ETA: {eta_str}")

        if final_record:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            # e_cultural_specificity_diff 값이 절대값 기준으로 3 이상인 경우 별도 파일에 저장
            try:
                diff_val = final_record.get('eval_similarity', {}).get('e_cultural_specificity_diff')
                if diff_val is not None and abs(int(diff_val)) >= 3:
                    # 최소화된 구조의 레코드 생성
                    high_record = {
                        "metadata": final_record.get('metadata'),
                        "human_eval": {
                            "e_cultural_specificity": final_record.get('human_eval', {}).get('e_cultural_specificity')
                        },
                        "vlm_eval": {
                            "e_cultural_specificity": final_record.get('vlm_eval', {}).get('e_cultural_specificity')
                        },
                        "eval_similarity": {
                            "e_cultural_specificity_diff": diff_val
                        }
                    }
                    # SAVE_THOUGHTS가 활성화된 경우 thought_trace 추가
                    if SAVE_THOUGHTS and 'thought_trace' in final_record.get('vlm_eval', {}):
                        high_record['eval_similarity']['thought_trace'] = \
                            final_record.get('vlm_eval', {}).get('thought_trace')
                    
                    with open(high_diff_file, 'a', encoding='utf-8') as hf:
                        hf.write(json.dumps(high_record, ensure_ascii=False) + "\n")
            except Exception:
                # 실패 시 무시하고 계속
                pass
        if error_msg:
            err_log = f"  -> [ERROR] Failed processing index {index}: {error_msg}"
            log(err_log)
            with open(error_file, "a", encoding="utf-8") as f_err:
                f_err.write(f"Index {index} | {err_log}\n")

    log(f"✅ Async Pipeline completed in {str(timedelta(seconds=int(time.time() - start_time)))}!")
    log_fp.close()

if __name__ == "__main__":
    # asyncio 실행 시작점
    asyncio.run(main_async())
    # 실행 완료 후 동일 위치의 generate_baseline_report.py 실행
    try:
        subprocess.run(["python3", "generate_baseline_report.py"], check=True)
    except Exception as e:
        print(f"generate_baseline_report.py 실행 중 오류: {e}")