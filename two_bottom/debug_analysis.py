# -*- coding: utf-8 -*-
"""
쌍바닥 분석 디버깅 코드
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def debug_data_structure():
    """데이터 구조 디버깅"""
    print("🔍 데이터 구조 디버깅")
    
    # 원본 데이터 로드
    df = pd.read_csv('코스피6개월종가_20250908_035947.csv')
    print(f"원본 데이터 형태: {df.shape}")
    print(f"컬럼: {df.columns[:5].tolist()}...")
    
    # 첫 번째 종목 데이터 확인
    first_stock = df.iloc[0, 0]  # 종목명
    print(f"\n첫 번째 종목: {first_stock}")
    
    # 가로 -> 세로 변환
    df_long = df.set_index('종목').reset_index().melt(
        id_vars=['종목'], 
        var_name='Date', 
        value_name='Close'
    )
    
    df_long['Date'] = pd.to_datetime(df_long['Date'])
    df_long = df_long.dropna().sort_values(['종목', 'Date']).reset_index(drop=True)
    
    print(f"변환된 데이터 형태: {df_long.shape}")
    
    # 첫 번째 종목의 데이터 확인
    first_stock_data = df_long[df_long['종목'] == first_stock].copy()
    print(f"\n{first_stock} 데이터:")
    print(f"  - 데이터 개수: {len(first_stock_data)}")
    print(f"  - 기간: {first_stock_data['Date'].min()} ~ {first_stock_data['Date'].max()}")
    print(f"  - 가격 범위: {first_stock_data['Close'].min():.0f} ~ {first_stock_data['Close'].max():.0f}")
    
    # 최근 30일 데이터 확인
    recent_data = first_stock_data.tail(30)
    print(f"\n최근 30일 데이터:")
    print(recent_data[['Date', 'Close']].head(10))
    
    return first_stock_data

def plot_stock_data(stock_data, symbol):
    """주가 데이터 시각화"""
    plt.figure(figsize=(15, 6))
    plt.plot(stock_data['Date'], stock_data['Close'], linewidth=2, color='black')
    plt.title(f'{symbol} - 주가 차트', fontsize=14, fontweight='bold')
    plt.xlabel('날짜', fontsize=12)
    plt.ylabel('주가 (원)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def find_simple_patterns(stock_data):
    """간단한 패턴 찾기"""
    print("\n🔍 간단한 패턴 분석")
    
    # 최근 60일 데이터
    recent_data = stock_data.tail(60).reset_index(drop=True)
    
    # 최저점과 최고점 찾기
    min_idx = recent_data['Close'].idxmin()
    max_idx = recent_data['Close'].idxmax()
    
    min_price = recent_data.loc[min_idx, 'Close']
    max_price = recent_data.loc[max_idx, 'Close']
    
    print(f"최근 60일 최저점: {min_price:,.0f}원 (인덱스: {min_idx})")
    print(f"최근 60일 최고점: {max_price:,.0f}원 (인덱스: {max_idx})")
    
    # 로컬 최소값 찾기 (간단한 방법)
    window = 3
    local_mins = []
    
    for i in range(window, len(recent_data) - window):
        current_price = recent_data.loc[i, 'Close']
        left_min = recent_data.loc[i-window:i-1, 'Close'].min()
        right_min = recent_data.loc[i+1:i+window, 'Close'].min()
        
        if current_price < left_min and current_price < right_min:
            local_mins.append((i, current_price))
    
    print(f"\n로컬 최소값 {len(local_mins)}개 발견:")
    for i, (idx, price) in enumerate(local_mins[:5]):  # 처음 5개만
        print(f"  {i+1}. 인덱스 {idx}: {price:,.0f}원")
    
    return local_mins

def main():
    """메인 함수"""
    print("🚀 쌍바닥 분석 디버깅 시작")
    print("="*50)
    
    # 데이터 구조 확인
    stock_data = debug_data_structure()
    
    # 첫 번째 종목 시각화
    first_stock = stock_data['종목'].iloc[0]
    print(f"\n📊 {first_stock} 차트 생성 중...")
    plot_stock_data(stock_data, first_stock)
    
    # 간단한 패턴 분석
    local_mins = find_simple_patterns(stock_data)
    
    print("\n✅ 디버깅 완료!")

if __name__ == '__main__':
    main()
