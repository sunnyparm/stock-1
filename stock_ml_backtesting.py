# -*- coding: utf-8 -*-
"""
머신러닝 기반 주가 예측 백테스팅 시스템.
Random Forest vs XGBoost vs LightGBM 성능 비교.
"""

import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb

# ====== 설정 ======
TICKER = "323410"  # 카카오뱅크
START_DATE = "2020-01-01"
END_DATE = "2024-06-30"
TRAIN_RATIO = 0.7
INIT_CASH = 100.0
FEE = 0.001
PREDICTION_DAYS = 5  # N일 후 상승/하락 예측

print("📊 머신러닝 기반 주가 예측 백테스팅 시작")
print(f"종목: {TICKER} (카카오뱅크)")
print(f"기간: {START_DATE} ~ {END_DATE}")
print("="*50)

# ====== 데이터 수집 및 전처리 ======
def create_features(df):
    """기술적 지표 기반 피처 생성"""
    # 가격 관련 피처
    df['Returns'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # 볼린저 밴드
    df['BB_upper'] = df['MA20'] + (df['Close'].rolling(20).std() * 2)
    df['BB_lower'] = df['MA20'] - (df['Close'].rolling(20).std() * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['MA20']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12).mean()
    exp2 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # 변동성
    df['Volatility'] = df['Returns'].rolling(20).std()
    
    # 가격 위치 지표
    df['Price_position'] = (df['Close'] - df['Close'].rolling(20).min()) / (df['Close'].rolling(20).max() - df['Close'].rolling(20).min())
    
    return df

def create_target(df, days=PREDICTION_DAYS):
    """타겟 변수 생성: N일 후 상승(1) vs 하락(0)"""
    df['Future_Close'] = df['Close'].shift(-days)
    df['Target'] = (df['Future_Close'] > df['Close']).astype(int)
    return df

# 데이터 로드
print("📈 데이터 수집 중...")
df = fdr.DataReader(TICKER, START_DATE, END_DATE)

# 피처 생성
print("🔧 피처 엔지니어링 중...")
df = create_features(df)
df = create_target(df, PREDICTION_DAYS)

# 피처 선택
feature_columns = ['Returns', 'MA5', 'MA10', 'MA20', 'MA60',
                  'BB_width', 'BB_position', 'RSI', 'MACD', 'MACD_signal', 
                  'MACD_hist', 'Volatility', 'Price_position']

# 결측값 제거
df = df.dropna()
X = df[feature_columns]
y = df['Target']

print(f"데이터 크기: {len(df)}행, 피처 수: {len(feature_columns)}개")
print(f"타겟 분포: 상승 {y.sum()}회, 하락 {len(y)-y.sum()}회")

# ====== 학습/테스트 분할 ======
split_idx = int(len(df) * TRAIN_RATIO)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
df_test = df[split_idx:]

print(f"훈련 데이터: {len(X_train)}개")
print(f"테스트 데이터: {len(X_test)}개")

# ====== 모델 정의 및 학습 ======
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
}

# 모델 학습 및 예측
print("\n🤖 모델 학습 및 예측 중...")
model_results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # 학습
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 상승 확률
    
    # 정확도
    accuracy = accuracy_score(y_test, y_pred)
    print(f"정확도: {accuracy:.4f}")
    
    # 예측 결과 저장
    model_results[name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy
    }

# ====== 백테스팅 함수 ======
def backtest_ml_strategy(df_test, predictions, probabilities, threshold=0.5, name="Model"):
    """머신러닝 예측 기반 백테스팅"""
    cash = INIT_CASH
    shares = 0
    position = 0  # 0: 현금, 1: 주식 보유
    
    equity_curve = []
    trades = []
    
    for i, (idx, row) in enumerate(df_test.iterrows()):
        price = row['Close']
        
        # 예측 확률이 임계값보다 높으면 매수, 낮으면 매도
        if probabilities[i] > threshold and position == 0:
            # 매수
            shares = (cash * (1 - FEE)) / price
            cash = 0
            position = 1
            trades.append({
                'Date': idx,
                'Action': 'BUY',
                'Price': price,
                'Probability': probabilities[i]
            })
        elif probabilities[i] < (1 - threshold) and position == 1:
            # 매도
            cash = shares * price * (1 - FEE)
            shares = 0
            position = 0
            trades.append({
                'Date': idx,
                'Action': 'SELL',
                'Price': price,
                'Probability': probabilities[i]
            })
        
        # 자산 가치 계산
        total_value = cash + shares * price
        equity_curve.append(total_value)
    
    # 최종 수익률 계산
    final_return = (equity_curve[-1] - INIT_CASH) / INIT_CASH * 100
    
    # Buy & Hold 수익률
    buy_hold_return = (df_test['Close'].iloc[-1] - df_test['Close'].iloc[0]) / df_test['Close'].iloc[0] * 100
    
    # 최대 낙폭 계산
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    return {
        'name': name,
        'final_return': final_return,
        'buy_hold_return': buy_hold_return,
        'max_drawdown': max_drawdown,
        'total_trades': len(trades),
        'equity_curve': equity_curve,
        'trades': trades
    }

# ====== 백테스팅 실행 ======
print("\n💰 백테스팅 실행 중...")
backtest_results = {}

for name, result in model_results.items():
    bt_result = backtest_ml_strategy(
        df_test, 
        result['predictions'], 
        result['probabilities'], 
        threshold=0.6,  # 60% 이상 확신할 때만 거래
        name=name
    )
    backtest_results[name] = bt_result

# ====== 결과 출력 ======
print("\n" + "="*60)
print("📊 머신러닝 모델 백테스팅 결과 비교")
print("="*60)

results_df = pd.DataFrame([
    {
        'Model': name,
        'Accuracy': f"{model_results[name]['accuracy']:.4f}",
        'Final Return (%)': f"{result['final_return']:.2f}",
        'Buy&Hold (%)': f"{result['buy_hold_return']:.2f}",
        'Max Drawdown (%)': f"{result['max_drawdown']:.2f}",
        'Total Trades': result['total_trades']
    }
    for name, result in backtest_results.items()
])

print(results_df.to_string(index=False))

# ====== 시각화 ======
plt.figure(figsize=(15, 10))

# 1. 자산 곡선 비교
plt.subplot(2, 2, 1)
for name, result in backtest_results.items():
    plt.plot(result['equity_curve'], label=f"{name}: {result['final_return']:.2f}%")

# Buy & Hold 라인 추가
buy_hold_curve = INIT_CASH * (df_test['Close'] / df_test['Close'].iloc[0])
plt.plot(buy_hold_curve.values, label=f"Buy&Hold: {backtest_results[list(backtest_results.keys())[0]]['buy_hold_return']:.2f}%", linestyle='--')

plt.title('자산 곡선 비교')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)

# 2. 예측 정확도 비교
plt.subplot(2, 2, 2)
accuracies = [model_results[name]['accuracy'] for name in models.keys()]
plt.bar(models.keys(), accuracies)
plt.title('예측 정확도 비교')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# 3. 수익률 vs MDD
plt.subplot(2, 2, 3)
returns = [backtest_results[name]['final_return'] for name in models.keys()]
mdds = [abs(backtest_results[name]['max_drawdown']) for name in models.keys()]
plt.scatter(mdds, returns)
for i, name in enumerate(models.keys()):
    plt.annotate(name, (mdds[i], returns[i]), xytext=(5, 5), textcoords='offset points')
plt.xlabel('Max Drawdown (%)')
plt.ylabel('Final Return (%)')
plt.title('수익률 vs 최대낙폭')
plt.grid(True, alpha=0.3)

# 4. 피처 중요도 (Random Forest)
plt.subplot(2, 2, 4)
rf_model = model_results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)

plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('피처 중요도 (Random Forest)')
plt.xlabel('Importance')

plt.tight_layout()
plt.show()

# ====== 최고 성능 모델 분석 ======
best_model = max(backtest_results.items(), key=lambda x: x[1]['final_return'])
print(f"\n🏆 최고 성능 모델: {best_model[0]}")
print(f"   수익률: {best_model[1]['final_return']:.2f}%")
print(f"   거래 횟수: {best_model[1]['total_trades']}회")
print(f"   최대 낙폭: {best_model[1]['max_drawdown']:.2f}%")

print("\n✅ 분석 완료!")