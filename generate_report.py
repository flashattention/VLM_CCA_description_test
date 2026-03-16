import os
import json
import re
import glob
import shutil
import pandas as pd
import logging
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import openai
from openai import OpenAI

# 1. 스타일 및 로그 설정
logging.getLogger('matplotlib').setLevel(logging.ERROR)
plt.rcdefaults()
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid", palette="viridis")

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) 
MODEL_NAME = "gpt-4.1" 

def extract_report_content(text):
    report_match = re.search(r'<report>(.*?)</report>', text, flags=re.DOTALL | re.IGNORECASE)
    return report_match.group(1).strip() if report_match else text.strip()

def normalize_d_accuracy(val):
    val = str(val).lower().strip()
    mapping = {
        'yes': 'correct', 'correct': 'correct',
        'no': 'incorrect', 'incorrect': 'incorrect',
        'ambiguous': 'ambiguous'
    }
    return mapping.get(val, 'incorrect')

def is_multiple_choice(options_str):
    """Parse metadata.options string to check if it's a multiple choice question"""
    try:
        if pd.isna(options_str):
            return False
        options_str = str(options_str).strip()
        if options_str == '[]' or options_str == '':
            return False
        options = json.loads(options_str)
        return len(options) > 0
    except:
        return False

def run_analysis(df, target_dir, subset_label="all"):
    """Run full analysis pipeline for a given dataframe subset"""
    charts_dir = f"{target_dir}/charts"
    os.makedirs(charts_dir, exist_ok=True)
    
    # --- [데이터 정규화 및 수치화] ---
    if 'human_eval.d_answer_accuracy' in df.columns:
        df['human_eval.d_answer_accuracy'] = df['human_eval.d_answer_accuracy'].apply(normalize_d_accuracy)
    if 'vlm_eval.d_answer_accuracy' in df.columns:
        df['vlm_eval.d_answer_accuracy'] = df['vlm_eval.d_answer_accuracy'].apply(normalize_d_accuracy)

    df['calc_d_match'] = (df['human_eval.d_answer_accuracy'] == df['vlm_eval.d_answer_accuracy']).astype(float)

    metrics = ['a1_keyword_cultural', 'a2_image_keyword_alignment', 'e_cultural_specificity']
    for col in metrics:
        df[f'human_eval.{col}'] = pd.to_numeric(df[f'human_eval.{col}'], errors='coerce')
        df[f'vlm_eval.{col}'] = pd.to_numeric(df[f'vlm_eval.{col}'], errors='coerce')

    df['calc_c_match'] = df['eval_similarity.c_question_image_relevance_match'].map({True: 1.0, False: 0.0}).fillna(0)
    df['calc_g_match'] = df['eval_similarity.g_text_required_match'].map({True: 1.0, False: 0.0}).fillna(0)
    
    # --- [시각화 1: 평균 지표 비교] ---
    comparison_df = pd.DataFrame([
        {'Metric': m, 'Type': t, 'Score': df[f'{t.lower()}_eval.{m}'].mean()}
        for m in metrics for t in ['Human', 'VLM']
    ])
    plt.figure(figsize=(10, 5))
    sns.barplot(data=comparison_df, x='Metric', y='Score', hue='Type')
    plt.title(f'Mean Score Comparison ({subset_label})')
    plt.savefig(f"{charts_dir}/01_{subset_label}_quant_metrics.png"); plt.close()

    # --- [시각화 2: 일치율] ---
    binary_matches = {
        'c_relevance_match': df['calc_c_match'].mean(),
        'd_accuracy_match': df['calc_d_match'].mean(),
        'g_text_required_match': df['calc_g_match'].mean(),
        'overall_match_rate': pd.to_numeric(df['eval_similarity.overall_match_rate'], errors='coerce').mean()
    }
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(binary_matches.values()), y=list(binary_matches.keys()), hue=list(binary_matches.keys()), palette="magma", legend=False)
    plt.title(f'Agreement Rates ({subset_label})')
    plt.savefig(f"{charts_dir}/02_{subset_label}_categorical_agreement.png"); plt.close()

    # --- [시각화 3: 핵심 지표별 점수 분포 (Histogram/Distribution)] ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, m in enumerate(metrics):
        dist_data = pd.melt(df[[f'human_eval.{m}', f'vlm_eval.{m}']], var_name='Evaluator', value_name='Score')
        dist_data['Evaluator'] = dist_data['Evaluator'].map({f'human_eval.{m}': 'Human', f'vlm_eval.{m}': 'VLM'})
        
        sns.countplot(data=dist_data, x='Score', hue='Evaluator', ax=axes[i], palette="Set2")
        axes[i].set_title(f'Score Distribution: {m}')
        axes[i].set_xlabel('Score (1-5)')
        axes[i].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(f"{charts_dir}/03_{subset_label}_score_distribution.png"); plt.close()

    # --- [통계 요약] ---
    stats_summary = {
        "Sample_Size": len(df),
        "Key_Agreement_Rates": binary_matches,
        "Mean_Scores": comparison_df.to_dict(orient='records'),
        "D_Accuracy_Distribution": {
            "Human": df['human_eval.d_answer_accuracy'].value_counts().to_dict(),
            "VLM": df['vlm_eval.d_answer_accuracy'].value_counts().to_dict()
        },
        "Score_Distributions": {
            m: {
                "Human": df[f'human_eval.{m}'].value_counts().sort_index().to_dict(),
                "VLM": df[f'vlm_eval.{m}'].value_counts().sort_index().to_dict()
            } for m in metrics
        }
    }
    
    return stats_summary, comparison_df, binary_matches

def main():
    target_dirs = sorted([d for d in glob.glob("output/*") if os.path.isdir(d)])
    if not target_dirs: return
    target_dir = target_dirs[-1]
    charts_dir = f"{target_dir}/charts"
    os.makedirs(charts_dir, exist_ok=True)

    with open(f"{target_dir}/results.jsonl", "r", encoding="utf-8") as f:
        df = pd.json_normalize([json.loads(line) for line in f])
    
    # --- [MC vs Non-MC 분류] ---
    df['is_mc'] = df['metadata.options'].apply(is_multiple_choice)
    
    # 세 가지 데이터셋 생성
    datasets = {
        'all': df,
        'mc': df[df['is_mc'] == True],
        'non_mc': df[df['is_mc'] == False]
    }
    
    # 각 데이터셋별 분석 실행
    all_stats = {}
    for subset_name, subset_df in datasets.items():
        if len(subset_df) == 0:
            continue
        print(f"Analyzing {subset_name}: {len(subset_df)} samples")
        all_stats[subset_name], _, _ = run_analysis(subset_df, target_dir, subset_name)

    # --- [OpenAI o1 프롬프트 구성 및 보고서 생성] ---
    all_stats_str = json.dumps(all_stats.get('all', {}), indent=2, ensure_ascii=False)
    mc_stats_str = json.dumps(all_stats.get('mc', {}), indent=2, ensure_ascii=False)
    non_mc_stats_str = json.dumps(all_stats.get('non_mc', {}), indent=2, ensure_ascii=False)
    
    instruction = f"""당신은 한국 문화 AI 평가 수석 연구원입니다.
VLM_CCA 데이터셋에 대한 사람과 VLM의 평가 분포 데이터를 분석하여 마크다운 보고서를 작성하십시오.

### [지표 정의]
1. a1_keyword_cultural: 키워드-문화 관련성
2. a2_image_keyword_alignment: 이미지-키워드 정합성
3. d_answer_accuracy: 정답 정확성 (correct, incorrect, ambiguous)
4. e_cultural_specificity: 문화 특이성 (1-5)

[분석용 통계 데이터]
전체 데이터: {all_stats_str}
객관식 문제만: {mc_stats_str}
주관식 문제만: {non_mc_stats_str}

[요구사항]
1. <report> 태그 내부에 한국어 마크다운으로 작성.
2. 전체 데이터 분석, 객관식 문제 분석, 주관식 문제 분석을 각각 섹션으로 구분하여 제시할 것.
3. 객관식과 주관식 문제 간의 평가 패턴 차이를 중점적으로 분석할 것.
4. 특히 '점수 분포(Score Distribution)' 데이터를 통해 사람이 고점을 준 문항에 대해 VLM이 어떤 경향성을 보이는지 심층 분석할 것.
5. 시각적 정합성(a2)과 문화 특이성(e)의 분포 차이가 시사하는 바를 논할 것.
"""

    # 시도 순서: 1) chat.completions (기본) 2) completions (text completion 폴백) 3) responses API 폴백
    try:
        response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": instruction}])
        report_raw = response.choices[0].message.content
    except Exception as e:
        try:
            response = client.completions.create(model=MODEL_NAME, prompt=instruction, max_tokens=1024)
            report_raw = response.choices[0].text
        except Exception:
            # 마지막으로 Responses API 시도
            try:
                response = client.responses.create(model=MODEL_NAME, input=instruction)
                if hasattr(response, "output_text"):
                    report_raw = response.output_text
                else:
                    report_raw = ""
                    try:
                        for out in getattr(response, "output", []):
                            for c in out.get("content", []):
                                if isinstance(c, dict) and "text" in c:
                                    report_raw += c["text"]
                                elif isinstance(c, str):
                                    report_raw += c
                    except Exception:
                        report_raw = str(response)
            except Exception as e2:
                raise

    report_text = extract_report_content(report_raw)

    final_md = f"""# 🏛️ 한국 문화 VQA: 인간-AI 메타 평가 리포트 (객관식/주관식 분류)

{report_text}

---
## 📊 시각 데이터 분석

### 전체 데이터 (All)
#### 1. 지표별 점수 분포 (Score Distribution)
> **설명**: 각 지표별로 사람과 VLM이 부여한 점수(1~5)의 빈도수를 비교합니다. AI의 평가 편향성을 확인할 수 있습니다.
![all_dist](charts/03_all_score_distribution.png)

#### 2. 정량 지표 평균 비교
![all_metrics](charts/01_all_quant_metrics.png)

#### 3. 지표별 일치율
![all_agreement](charts/02_all_categorical_agreement.png)

---

### 객관식 문제만 (Multiple Choice)
#### 1. 지표별 점수 분포
![mc_dist](charts/03_mc_score_distribution.png)

#### 2. 정량 지표 평균 비교
![mc_metrics](charts/01_mc_quant_metrics.png)

#### 3. 지표별 일치율
![mc_agreement](charts/02_mc_categorical_agreement.png)

---

### 주관식 문제만 (Non-Multiple Choice)
#### 1. 지표별 점수 분포
![non_mc_dist](charts/03_non_mc_score_distribution.png)

#### 2. 정량 지표 평균 비교
![non_mc_metrics](charts/01_non_mc_quant_metrics.png)

#### 3. 지표별 일치율
![non_mc_agreement](charts/02_non_mc_categorical_agreement.png)
"""
    
    with open(f"{target_dir}/final_report.md", "w", encoding="utf-8") as f:
        f.write(final_md)
    print(f"✅ MC/Non-MC 분류 분석 리포트 생성 완료: {target_dir}/final_report.md")
    print(f"   전체: {len(datasets['all'])} | 객관식: {len(datasets['mc'])} | 주관식: {len(datasets['non_mc'])}")
    
    # --- [최상단 디렉터리로 복사] ---
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 기존 파일들 제거 (선택사항)
    root_final_report = os.path.join(project_root, "final_report.md")
    root_charts_dir = os.path.join(project_root, "charts")
    
    if os.path.exists(root_final_report):
        os.remove(root_final_report)
    if os.path.exists(root_charts_dir):
        shutil.rmtree(root_charts_dir)
    
    # 보고서와 charts 폴더 복사
    shutil.copy(f"{target_dir}/final_report.md", root_final_report)
    shutil.copytree(f"{target_dir}/charts", root_charts_dir)
    
    print(f"✅ 보고서 및 charts 폴더를 최상단 디렉터리로 복사 완료:")
    print(f"   - {root_final_report}")
    print(f"   - {root_charts_dir}/")

if __name__ == "__main__":
    main()