import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykrx import stock
from datetime import datetime, timedelta

# ==============================
# 🔧 사용자 설정값
# ==============================
TICKER = "298690"         # 종목코드 (삼성전자)
PERIOD_WEEKS = 26         # 조회 기간 (주 단위) → 26주 = 6개월
PIVOT_WINDOW = 3          # Pivot 탐지 윈도우
CLUSTER_TOL = 0.01        # 지지선 클러스터링 허용오차 (±1%)
NEAR_TOL = 0.005          # 터치 판정 허용오차 (±0.5%)
MIN_TOUCHES = 3           # 최소 터치 횟수
VOL_PERCENTILE = 0.7      # 거래량 상위 % 기준 (0.7 = 상위 30%)
REBOUND_DAYS = 5          # 반등 관찰 기간 (일)
REBOUND_PCT = 0.02        # 반등 성공 기준 (+2%)
STOP_PCT = 0.02           # 실패 기준 (-2%)
# ==============================

# ====== 데이터 수집 ======
today = datetime.today()
start_date = today - timedelta(weeks=PERIOD_WEEKS)

df = stock.get_market_ohlcv_by_date(
    fromdate=start_date.strftime("%Y%m%d"),
    todate=today.strftime("%Y%m%d"),
    ticker=TICKER
).reset_index()

df.rename(columns={"날짜":"Date","시가":"Open","고가":"High","저가":"Low","종가":"Close","거래량":"Volume"}, inplace=True)
df["Date"] = pd.to_datetime(df["Date"])

# ====== Pivot Low 탐지 ======
def find_pivot_lows(data, window=3):
    lows = []
    for i in range(window, len(data)-window):
        if data["Low"].iloc[i] == min(data["Low"].iloc[i-window:i+window+1]):
            lows.append((data["Date"].iloc[i], data["Low"].iloc[i]))
    return pd.DataFrame(lows, columns=["Date","PivotLow"])

# ====== 지지선 후보군 탐색 ======
def cluster_levels(levels, tol=0.01):
    clustered = []
    levels = sorted(levels)
    cluster = [levels[0]]
    for lv in levels[1:]:
        if abs(lv - np.mean(cluster)) / lv < tol:
            cluster.append(lv)
        else:
            clustered.append(np.mean(cluster))
            cluster = [lv]
    clustered.append(np.mean(cluster))
    return clustered

# ====== 강한 지지선 필터링 ======
def detect_support_levels(data):
    pivots = find_pivot_lows(data, window=PIVOT_WINDOW)
    if pivots.empty:
        return pd.DataFrame(columns=["SupportLevel","Touches","HighVolTouches"])
    
    levels = cluster_levels(pivots["PivotLow"].values, tol=CLUSTER_TOL)
    support_levels = []
    vol_threshold = data["Volume"].quantile(VOL_PERCENTILE)

    for lv in levels:
        touches = data[np.isclose(data["Low"], lv, rtol=NEAR_TOL)]
        touch_count = len(touches)
        high_volume_touches = (touches["Volume"] > vol_threshold).sum()
        if touch_count >= MIN_TOUCHES and high_volume_touches >= 1:
            support_levels.append((lv, touch_count, high_volume_touches))

    return pd.DataFrame(support_levels, columns=["SupportLevel","Touches","HighVolTouches"])

# ====== 지지선 반등 성공률 평가 ======
def evaluate_support_accuracy(data, support_df):
    results = []
    for s in support_df["SupportLevel"]:
        touches = []
        for i in range(len(data) - REBOUND_DAYS):
            price = data["Close"].iloc[i]
            if abs(price - s) / s < NEAR_TOL:  # 터치
                future = data.iloc[i+1:i+1+REBOUND_DAYS]
                if (future["Close"] >= price * (1 + REBOUND_PCT)).any():
                    touches.append("Success")
                elif (future["Close"] <= price * (1 - STOP_PCT)).any():
                    touches.append("Fail")
                else:
                    touches.append("Neutral")
        if touches:
            success_count = touches.count("Success")
            fail_count = touches.count("Fail")
            total = len(touches)
            success_rate = success_count / total * 100
            results.append({
                "SupportLevel": round(s,2),
                "Touches": total,
                "Success": success_count,
                "Fail": fail_count,
                "SuccessRate(%)": round(success_rate,2)
            })
    return pd.DataFrame(results)

# ====== 실행 ======
support_df = detect_support_levels(df)
print("🔹 강한 지지선 후보")
print(support_df)

accuracy_df = evaluate_support_accuracy(df, support_df)
print("\n🔹 지지선 반등 성공률")
print(accuracy_df)

# ====== 시각화 (지지선 + 가격 라벨) ======
plt.figure(figsize=(14,7))
plt.plot(df["Date"], df["Close"], label="Close", color="black")

for s in support_df["SupportLevel"]:
    plt.axhline(s, color="green", linestyle="--", alpha=0.7)
    # 가격 라벨 표시 (차트 오른쪽 끝에 숫자 출력)
    plt.text(df["Date"].iloc[-1], s, f"{int(s)}",
             va='center', ha='left', color="green", fontsize=9)

plt.title(f"{TICKER} Strong Support Levels (최근 {PERIOD_WEEKS//4}개월)")
plt.legend()
plt.show()
