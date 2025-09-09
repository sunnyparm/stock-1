# -*- coding: utf-8 -*-
"""
업데이트된 쌍바닥 분석 결과 시각화
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def create_visualization():
    """업데이트된 쌍바닥 분석 결과 시각화"""
    
    # 결과 데이터 로드
    results_df = pd.read_csv('updated_valid_double_bottom_results.csv')
    
    # 원본 데이터 로드
    df = pd.read_csv('코스피6개월종가_20250908_035947.csv')
    
    # 상위 6개 종목 선택
    top_stocks = results_df.head(6)
    
    # 차트 생성
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (_, stock) in enumerate(top_stocks.iterrows()):
        symbol = stock['종목']
        
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
        
        # 최근 120일 데이터만 사용
        recent_data = stock_data.tail(120)
        
        # 차트 그리기
        axes[i].plot(recent_data['Date'], recent_data['Close'], 
                    linewidth=2, color='black', label='종가')
        
        # 쌍바닥 패턴 지점 표시
        b1_price = stock['첫번째바닥']
        b2_price = stock['두번째바닥']
        peak_price = stock['넥라인']
        current_price = stock['현재가']
        
        # 바닥과 넥라인 위치 찾기
        b1_dates = recent_data[recent_data['Close'] == b1_price]['Date']
        b2_dates = recent_data[recent_data['Close'] == b2_price]['Date']
        peak_dates = recent_data[recent_data['Close'] == peak_price]['Date']
        
        # 첫 번째 바닥 표시
        if len(b1_dates) > 0:
            axes[i].scatter(b1_dates.iloc[0], b1_price, 
                          color='red', s=100, marker='v', 
                          label=f'첫 번째 바닥 ({b1_price:,.0f}원)', zorder=5)
        
        # 두 번째 바닥 표시
        if len(b2_dates) > 0:
            axes[i].scatter(b2_dates.iloc[-1], b2_price, 
                          color='red', s=100, marker='v', 
                          label=f'두 번째 바닥 ({b2_price:,.0f}원)', zorder=5)
        
        # 넥라인 표시
        if len(peak_dates) > 0:
            axes[i].scatter(peak_dates.iloc[0], peak_price, 
                          color='blue', s=100, marker='^', 
                          label=f'넥라인 ({peak_price:,.0f}원)', zorder=5)
        
        # 현재가 표시
        axes[i].scatter(recent_data['Date'].iloc[-1], current_price, 
                       color='green', s=100, marker='o', 
                       label=f'현재가 ({current_price:,.0f}원)', zorder=5)
        
        # 넥라인 수평선
        axes[i].axhline(y=peak_price, color='blue', linestyle='--', alpha=0.7)
        
        # 제목 설정
        axes[i].set_title(f'{symbol}\n점수: {stock["종합점수"]:.1f} | 바닥차이: {stock["바닥차이(%)"]:.1f}% | 횡보: {stock["최근바닥횡보"]}', 
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
    plt.suptitle('🏆 업데이트된 쌍바닥 패턴 분석 - 상위 6개 종목\n(최근 바닥 이후 3일간 횡보 조건 포함)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    # 통계 정보 출력
    print("📊 업데이트된 쌍바닥 분석 통계")
    print("="*50)
    print(f"총 분석 종목 수: {len(results_df)}개")
    print(f"평균 종합 점수: {results_df['종합점수'].mean():.1f}점")
    print(f"평균 바닥 차이: {results_df['바닥차이(%)'].mean():.2f}%")
    print(f"평균 반등률: {results_df['반등률(%)'].mean():.2f}%")
    print(f"평균 돌파률: {results_df['돌파률(%)'].mean():.2f}%")
    
    # 횡보 패턴 통계
    sideways_count = len(results_df[results_df['최근바닥횡보'] == '✅ 횡보'])
    print(f"횡보 패턴 종목: {sideways_count}개 ({sideways_count/len(results_df)*100:.1f}%)")
    
    # 상위 10개 종목 정보
    print(f"\n🏆 상위 10개 쌍바닥 패턴 종목:")
    print("-" * 80)
    for i, (_, stock) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"{i:2d}. {stock['종목']:15s} | 점수: {stock['종합점수']:5.1f} | "
              f"바닥차이: {stock['바닥차이(%)']:4.1f}% | "
              f"반등률: {stock['반등률(%)']:5.1f}% | "
              f"돌파률: {stock['돌파률(%)']:6.1f}% | "
              f"횡보: {stock['최근바닥횡보']}")

if __name__ == '__main__':
    create_visualization()

