import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import platform
from datetime import datetime

# =========================
# 🔧 사용자 설정
# =========================
NEAR_TOL = 0.03       # 최저가 근처 (±3%)
RANGE_DAYS = 10       # 횡보 판정 구간
SIDEWAYS_TOL = 0.05   # 횡보 기준 (5%)

# =========================
# 파일 업로드
# =========================
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="코스피 6개월 종가 파일 업로드",
    filetypes=[("CSV files", "*.csv")]
)

df = pd.read_csv(file_path)

# =========================
# 날짜 컬럼 자동 추출
# =========================
date_cols = [col for col in df.columns if col.startswith("2025-")]

results = []

for idx, row in df.iterrows():
    # 날짜별 종가만 추출 (문자열 → 숫자 변환)
    prices = row[date_cols].astype(str).str.replace(",", "").str.strip()
    prices = pd.to_numeric(prices, errors="coerce").dropna().values

    if len(prices) == 0:
        continue
    
    last_close = prices[-1]               # 최근 종가
    real_min = np.min(prices)             # 실제 최저가
    
    # ⚠️ 최저가 검증: 실제 배열에 없는 값은 제외
    if not np.isin(real_min, prices):
        print(f"⚠️ {row['종목']} → 계산된 최저가 {real_min} 값이 배열에 없음 → 제외")
        continue

    # 최저가 근처 여부
    if abs(last_close - real_min) / real_min <= NEAR_TOL:
        # 최근 RANGE_DAYS 기준 횡보 여부
        if len(prices) >= RANGE_DAYS:
            recent = prices[-RANGE_DAYS:]
            max_p = np.max(recent)
            min_p = np.min(recent)
            avg_p = np.mean(recent)
            sideways = (max_p - min_p) / avg_p
        else:
            sideways = 1.0  # 데이터 부족 시 횡보 아님 처리

        if sideways <= SIDEWAYS_TOL:
            results.append({
                "종목": row["종목"],
                "최근종가": int(last_close),
                "최저가": int(real_min),
                "괴리율(%)": round((last_close - real_min) / real_min * 100, 2),
                "횡보율(%)": round(sideways * 100, 2),
                "시가총액": row["시가총액"],
                "섹터": row["섹터"]
            })

# =========================
# 결과 출력 및 저장
# =========================
result_df = pd.DataFrame(results)
print("🔹 최근 종가가 최저가 3% 이내에서 횡보 중인 종목")
print(result_df.sort_values("괴리율(%)"))

# 저장 파일 이름에 timestamp 추가
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(os.path.dirname(file_path), f"최저가에서횡보중인종목_{timestamp}.csv")
result_df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"✅ 결과가 '{save_path}' 파일로 저장되었습니다.")

# =========================
# 저장된 파일 자동 열기
# =========================
if platform.system() == "Windows":
    os.startfile(save_path)
elif platform.system() == "Darwin":  # macOS
    os.system(f"open '{save_path}'")
else:  # Linux
    os.system(f"xdg-open '{save_path}'")
