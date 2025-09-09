# -*- coding: utf-8 -*-
"""
개별 종목 상세 쌍바닥 패턴 분석 차트
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def create_detailed_chart(symbol='쿠쿠홀딩스'):
    """특정 종목의 상세 쌍바닥 패턴 차트 생성"""
    
    # 결과 데이터 로드
    results_df = pd.read_csv('updated_valid_double_bottom_results.csv')
    
    # 원본 데이터 로드
    df = pd.read_csv('코스피6개월종가_20250908_035947.csv')
    
    # 해당 종목 정보 가져오기
    stock_info = results_df[results_df['종목'] == symbol].iloc[0]
    
    # 해당 종목 데이터 추출
    stock_row = df[df['종목'] == symbol].iloc[0]
    
    # 날짜와 가격 데이터 분리
    dates = []
    prices = []
    
    for col in df.columns[1:]:  # 첫 번째 컬럼(종목) 제외
        if pd.notna(stock_row[col]):
            dates.append(pd.to_datetime(col))
            prices.append(float(stock_row[col]))
    
    # DataFrame 생성
    stock_data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    }).sort_values('Date').reset_index(drop=True)
    
    # 차트 생성 (2개 서브플롯)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 1. 전체 기간 차트
    ax1.plot(stock_data['Date'], stock_data['Close'], 
             linewidth=2, color='black', label=f'{symbol} 주가')
    
    # 쌍바닥 패턴 지점 표시
    b1_price = stock_info['첫번째바닥']
    b2_price = stock_info['두번째바닥']
    peak_price = stock_info['넥라인']
    current_price = stock_info['현재가']
    
    # 바닥과 넥라인 위치 찾기
    b1_dates = stock_data[stock_data['Close'] == b1_price]['Date']
    b2_dates = stock_data[stock_data['Close'] == b2_price]['Date']
    peak_dates = stock_data[stock_data['Close'] == peak_price]['Date']
    
    # 첫 번째 바닥 표시
    if len(b1_dates) > 0:
        ax1.scatter(b1_dates.iloc[0], b1_price, 
                   color='red', s=120, marker='v', 
                   label=f'첫 번째 바닥 ({b1_price:,.0f}원)', zorder=5)
    
    # 두 번째 바닥 표시
    if len(b2_dates) > 0:
        ax1.scatter(b2_dates.iloc[-1], b2_price, 
                   color='red', s=120, marker='v', 
                   label=f'두 번째 바닥 ({b2_price:,.0f}원)', zorder=5)
    
    # 넥라인 표시
    if len(peak_dates) > 0:
        ax1.scatter(peak_dates.iloc[0], peak_price, 
                   color='blue', s=120, marker='^', 
                   label=f'넥라인 ({peak_price:,.0f}원)', zorder=5)
    
    # 현재가 표시
    ax1.scatter(stock_data['Date'].iloc[-1], current_price, 
               color='green', s=120, marker='o', 
               label=f'현재가 ({current_price:,.0f}원)', zorder=5)
    
    # 넥라인 수평선
    ax1.axhline(y=peak_price, color='blue', linestyle='--', alpha=0.7, label='넥라인 저항선')
    
    # 바닥 수평선
    ax1.axhline(y=b1_price, color='red', linestyle=':', alpha=0.5, label='바닥 지지선')
    ax1.axhline(y=b2_price, color='red', linestyle=':', alpha=0.5)
    
    ax1.set_title(f'{symbol} - 쌍바닥 패턴 분석 (전체 기간)\n'
                  f'종합점수: {stock_info["종합점수"]:.1f} | 바닥차이: {stock_info["바닥차이(%)"]:.1f}% | '
                  f'반등률: {stock_info["반등률(%)"]:.1f}% | 돌파률: {stock_info["돌파률(%)"]:.1f}%', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('주가 (원)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # 2. 최근 60일 확대 차트
    recent_data = stock_data.tail(60)
    ax2.plot(recent_data['Date'], recent_data['Close'], 
             linewidth=2, color='black', label=f'{symbol} 주가')
    
    # 최근 데이터에서 패턴 지점 표시
    recent_b1 = recent_data[recent_data['Close'] == b1_price]
    recent_b2 = recent_data[recent_data['Close'] == b2_price]
    recent_peak = recent_data[recent_data['Close'] == peak_price]
    
    if len(recent_b1) > 0:
        ax2.scatter(recent_b1['Date'], recent_b1['Close'], 
                   color='red', s=120, marker='v', 
                   label='첫 번째 바닥', zorder=5)
    
    if len(recent_b2) > 0:
        ax2.scatter(recent_b2['Date'], recent_b2['Close'], 
                   color='red', s=120, marker='v', 
                   label='두 번째 바닥', zorder=5)
    
    if len(recent_peak) > 0:
        ax2.scatter(recent_peak['Date'], recent_peak['Close'], 
                   color='blue', s=120, marker='^', 
                   label='넥라인', zorder=5)
    
    # 현재가 표시
    ax2.scatter(recent_data['Date'].iloc[-1], recent_data['Close'].iloc[-1], 
               color='green', s=120, marker='o', 
               label='현재가', zorder=5)
    
    # 넥라인 수평선
    ax2.axhline(y=peak_price, color='blue', linestyle='--', alpha=0.7)
    
    # 바닥 수평선
    ax2.axhline(y=b1_price, color='red', linestyle=':', alpha=0.5)
    ax2.axhline(y=b2_price, color='red', linestyle=':', alpha=0.5)
    
    ax2.set_title(f'{symbol} - 쌍바닥 패턴 차트 (최근 60일)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('날짜', fontsize=12)
    ax2.set_ylabel('주가 (원)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='plain', axis='y')
    
    # 날짜 축 포맷팅
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # 상세 정보 출력
    print(f"🔍 {symbol} 상세 쌍바닥 패턴 분석")
    print("="*60)
    print(f"📊 종합 점수: {stock_info['종합점수']:.1f}/100")
    print(f"💰 첫 번째 바닥: {stock_info['첫번째바닥']:,.0f}원")
    print(f"💰 두 번째 바닥: {stock_info['두번째바닥']:,.0f}원")
    print(f"📈 넥라인: {stock_info['넥라인']:,.0f}원")
    print(f"💎 현재가: {stock_info['현재가']:,.0f}원")
    print(f"📉 바닥 차이: {stock_info['바닥차이(%)']:.2f}%")
    print(f"📈 반등률: {stock_info['반등률(%)']:.2f}%")
    print(f"🚀 돌파률: {stock_info['돌파률(%)']:.2f}%")
    print(f"📊 최근 바닥 이후 횡보: {stock_info['최근바닥횡보']}")
    
    if pd.notna(stock_info['이전최저바닥']) and stock_info['이전최저바닥'] != 'N/A':
        print(f"📊 이전 최저 바닥: {stock_info['이전최저바닥']:,.0f}원")
        print(f"📈 바닥 개선률: {stock_info['바닥개선률(%)']:.2f}%")
    
    print(f"✅ 유효성 검증: {stock_info['유효성검증']}")
    print(f"📊 검증 점수: {stock_info['검증점수']:.1f}/100")
    
    if stock_info['문제점수'] > 0:
        print(f"⚠️ 문제점: {stock_info['문제점']}")
    else:
        print("✅ 문제점: 없음")

def create_comparison_chart():
    """상위 3개 종목 비교 차트"""
    
    # 결과 데이터 로드
    results_df = pd.read_csv('updated_valid_double_bottom_results.csv')
    
    # 원본 데이터 로드
    df = pd.read_csv('코스피6개월종가_20250908_035947.csv')
    
    # 상위 3개 종목 선택
    top3 = results_df.head(3)
    
    # 차트 생성
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, (_, stock) in enumerate(top3.iterrows()):
        symbol = stock['종목']
        
        # 해당 종목 데이터 추출
        stock_row = df[df['종목'] == symbol].iloc[0]
        
        # 날짜와 가격 데이터 분리
        dates = []
        prices = []
        
        for col in df.columns[1:]:
            if pd.notna(stock_row[col]):
                dates.append(pd.to_datetime(col))
                prices.append(float(stock_row[col]))
        
        # DataFrame 생성
        stock_data = pd.DataFrame({
            'Date': dates,
            'Close': prices
        }).sort_values('Date').reset_index(drop=True)
        
        # 최근 60일 데이터만 사용
        recent_data = stock_data.tail(60)
        
        # 차트 그리기
        axes[i].plot(recent_data['Date'], recent_data['Close'], 
                    linewidth=2, color='black', label='종가')
        
        # 쌍바닥 패턴 지점 표시
        b1_price = stock['첫번째바닥']
        b2_price = stock['두번째바닥']
        peak_price = stock['넥라인']
        current_price = stock['현재가']
        
        # 바닥과 넥라인 위치 찾기
        recent_b1 = recent_data[recent_data['Close'] == b1_price]
        recent_b2 = recent_data[recent_data['Close'] == b2_price]
        recent_peak = recent_data[recent_data['Close'] == peak_price]
        
        # 첫 번째 바닥 표시
        if len(recent_b1) > 0:
            axes[i].scatter(recent_b1['Date'], recent_b1['Close'], 
                          color='red', s=100, marker='v', 
                          label='첫 번째 바닥', zorder=5)
        
        # 두 번째 바닥 표시
        if len(recent_b2) > 0:
            axes[i].scatter(recent_b2['Date'], recent_b2['Close'], 
                          color='red', s=100, marker='v', 
                          label='두 번째 바닥', zorder=5)
        
        # 넥라인 표시
        if len(recent_peak) > 0:
            axes[i].scatter(recent_peak['Date'], recent_peak['Close'], 
                          color='blue', s=100, marker='^', 
                          label='넥라인', zorder=5)
        
        # 현재가 표시
        axes[i].scatter(recent_data['Date'].iloc[-1], current_price, 
                       color='green', s=100, marker='o', 
                       label='현재가', zorder=5)
        
        # 넥라인 수평선
        axes[i].axhline(y=peak_price, color='blue', linestyle='--', alpha=0.7)
        
        # 제목 설정
        axes[i].set_title(f'{symbol}\n점수: {stock["종합점수"]:.1f} | 바닥차이: {stock["바닥차이(%)"]:.1f}%', 
                         fontsize=12, fontweight='bold')
        axes[i].set_ylabel('주가 (원)', fontsize=10)
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
        
        # 날짜 축 포맷팅
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        axes[i].tick_params(axis='x', rotation=45)
        
        # Y축 포맷팅
        axes[i].ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    plt.suptitle('🏆 상위 3개 쌍바닥 패턴 종목 비교 (최근 60일)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.show()

if __name__ == '__main__':
    # 1. 개별 종목 상세 차트
    create_detailed_chart('쿠쿠홀딩스')
    
    # 2. 상위 3개 종목 비교 차트
    create_comparison_chart()

