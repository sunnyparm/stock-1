# -*- coding: utf-8 -*-
"""
개선된 SK 쌍바닥 패턴 분석 - 이전 최저 바닥 고려
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def find_previous_lowest_bottom(data, current_bottom, lookback_days=30):
    """이전 최저 바닥 찾기"""
    # 현재 바닥 이전 N일간의 최저가 찾기
    current_idx = data[data['Close'] == current_bottom].index[0] if len(data[data['Close'] == current_bottom]) > 0 else None
    
    if current_idx is None or current_idx < lookback_days:
        return None, None
    
    # 이전 기간 데이터
    previous_data = data.iloc[max(0, current_idx - lookback_days):current_idx]
    if len(previous_data) == 0:
        return None, None
    
    # 이전 기간 최저가
    prev_lowest = previous_data['Close'].min()
    prev_lowest_idx = previous_data['Close'].idxmin()
    
    return prev_lowest, prev_lowest_idx

def check_sideways_after_bottom(data, bottom_idx, days=3, tolerance_pct=0.02):
    """
    바닥 이후 N일간 횡보 여부 확인
    
    Args:
        data: 주가 데이터
        bottom_idx: 바닥 인덱스
        days: 확인할 일수
        tolerance_pct: 횡보 허용 범위 (퍼센트)
    
    Returns:
        bool: 횡보 여부
    """
    if bottom_idx + days >= len(data):
        return False
    
    # 바닥 이후 N일간 데이터
    after_bottom_data = data.iloc[bottom_idx:bottom_idx + days + 1]
    bottom_price = data.iloc[bottom_idx]['Close']
    
    # 바닥 이후 최고가와 최저가
    max_price = after_bottom_data['Close'].max()
    min_price = after_bottom_data['Close'].min()
    
    # 바닥 가격 대비 변동폭이 허용 범위 내인지 확인
    max_deviation = max(abs(max_price - bottom_price), abs(min_price - bottom_price))
    deviation_pct = max_deviation / bottom_price
    
    return deviation_pct <= tolerance_pct

def validate_double_bottom_pattern(data, first_bottom, second_bottom, neckline):
    """쌍바닥 패턴 유효성 검증 (개선된 버전)"""
    # 1. 이전 최저 바닥 확인
    prev_lowest, prev_lowest_idx = find_previous_lowest_bottom(data, first_bottom, 60)
    
    # 2. 상대적 바닥 확인 (주변 20일 대비)
    first_bottom_idx = data[data['Close'] == first_bottom].index[0]
    second_bottom_idx = data[data['Close'] == second_bottom].index[-1]
    
    # 첫 번째 바닥 주변 최저가 확인
    first_window = data.iloc[max(0, first_bottom_idx-10):first_bottom_idx+10]
    first_local_min = first_window['Close'].min()
    
    # 두 번째 바닥 주변 최저가 확인  
    second_window = data.iloc[max(0, second_bottom_idx-10):second_bottom_idx+10]
    second_local_min = second_window['Close'].min()
    
    # 3. 최근 최저가 이후 3일간 횡보 확인
    recent_bottom_sideways = check_sideways_after_bottom(data, second_bottom_idx, days=3, tolerance_pct=0.02)
    
    validation_result = {
        'prev_lowest': prev_lowest,
        'prev_lowest_idx': prev_lowest_idx,
        'first_local_min': first_local_min,
        'second_local_min': second_local_min,
        'recent_bottom_sideways': recent_bottom_sideways,
        'is_valid_double_bottom': False,
        'issues': []
    }
    
    # 유효성 검증
    if prev_lowest is not None and first_bottom <= prev_lowest:
        validation_result['issues'].append(f"첫 번째 바닥({first_bottom:,}원)이 이전 최저가({prev_lowest:,}원)보다 낮거나 같음")
    
    if first_bottom != first_local_min:
        validation_result['issues'].append(f"첫 번째 바닥이 주변 최저가({first_local_min:,}원)와 다름")
        
    if second_bottom != second_local_min:
        validation_result['issues'].append(f"두 번째 바닥이 주변 최저가({second_local_min:,}원)와 다름")
    
    # 최근 바닥 이후 횡보 확인
    if not recent_bottom_sideways:
        validation_result['issues'].append(f"최근 바닥 이후 3일간 횡보하지 않음")
    
    if len(validation_result['issues']) == 0:
        validation_result['is_valid_double_bottom'] = True
    
    return validation_result

def create_improved_sk_chart():
    """개선된 SK 종목 쌍바닥 패턴 차트 생성"""
    
    # SK 데이터 추출
    df = pd.read_csv('코스피6개월종가_20250908_035947.csv')
    sk_row = df[df['종목'] == 'SK'].iloc[0]
    
    # 날짜와 가격 데이터 분리
    dates = []
    prices = []
    
    for col in df.columns[1:]:  # 첫 번째 컬럼(종목) 제외
        if pd.notna(sk_row[col]):
            dates.append(pd.to_datetime(col))
            prices.append(float(sk_row[col]))
    
    # DataFrame 생성
    sk_data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    }).sort_values('Date').reset_index(drop=True)
    
    # 전체 기간 최저가/최고가 찾기
    overall_min = sk_data['Close'].min()
    overall_max = sk_data['Close'].max()
    overall_min_idx = sk_data['Close'].idxmin()
    overall_max_idx = sk_data['Close'].idxmax()
    
    # 기존 쌍바닥 패턴 정보
    first_bottom = 203000
    second_bottom = 203000
    neckline = 207500
    current_price = 209000
    
    # 패턴 유효성 검증
    validation = validate_double_bottom_pattern(sk_data, first_bottom, second_bottom, neckline)
    
    # 쌍바닥 패턴 지점 찾기
    first_bottom_idx = sk_data[sk_data['Close'] == first_bottom].index[0] if len(sk_data[sk_data['Close'] == first_bottom]) > 0 else None
    second_bottom_idx = sk_data[sk_data['Close'] == second_bottom].index[-1] if len(sk_data[sk_data['Close'] == second_bottom]) > 0 else None
    neckline_idx = sk_data[sk_data['Close'] == neckline].index[0] if len(sk_data[sk_data['Close'] == neckline]) > 0 else None
    
    # 차트 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # 1. 전체 차트
    ax1.plot(sk_data['Date'], sk_data['Close'], linewidth=2, color='black', label='SK 주가')
    
    # 전체 기간 최저가/최고가 표시
    ax1.scatter(sk_data.loc[overall_min_idx, 'Date'], overall_min, 
               color='purple', s=120, marker='s', label=f'전체 최저가 ({overall_min:,}원)', zorder=6)
    ax1.scatter(sk_data.loc[overall_max_idx, 'Date'], overall_max, 
               color='orange', s=120, marker='s', label=f'전체 최고가 ({overall_max:,}원)', zorder=6)
    
    # 이전 최저 바닥 표시
    if validation['prev_lowest'] is not None:
        ax1.scatter(sk_data.loc[validation['prev_lowest_idx'], 'Date'], validation['prev_lowest'], 
                   color='brown', s=100, marker='v', label=f'이전 최저 바닥 ({validation["prev_lowest"]:,}원)', zorder=5)
    
    # 쌍바닥 패턴 표시 (유효성에 따라 색상 변경)
    pattern_color = 'red' if validation['is_valid_double_bottom'] else 'gray'
    pattern_alpha = 1.0 if validation['is_valid_double_bottom'] else 0.5
    
    if first_bottom_idx is not None:
        ax1.scatter(sk_data.loc[first_bottom_idx, 'Date'], first_bottom, 
                   color=pattern_color, s=100, marker='v', alpha=pattern_alpha,
                   label=f'첫 번째 바닥 ({first_bottom:,}원)', zorder=5)
    
    if second_bottom_idx is not None:
        ax1.scatter(sk_data.loc[second_bottom_idx, 'Date'], second_bottom, 
                   color=pattern_color, s=100, marker='v', alpha=pattern_alpha,
                   label=f'두 번째 바닥 ({second_bottom:,}원)', zorder=5)
    
    if neckline_idx is not None:
        ax1.scatter(sk_data.loc[neckline_idx, 'Date'], neckline, 
                   color='blue', s=100, marker='^', alpha=pattern_alpha,
                   label=f'넥라인 ({neckline:,}원)', zorder=5)
    
    # 현재가 표시
    ax1.scatter(sk_data['Date'].iloc[-1], current_price, 
               color='green', s=100, marker='o', label=f'현재가 ({current_price:,}원)', zorder=5)
    
    # 넥라인 수평선
    ax1.axhline(y=neckline, color='blue', linestyle='--', alpha=0.7, label='넥라인 저항선')
    
    # 패턴 유효성에 따른 제목
    title_suffix = "✅ 유효한 쌍바닥" if validation['is_valid_double_bottom'] else "❌ 의심스러운 패턴"
    ax1.set_title(f'SK - 쌍바닥 패턴 분석 (전체 기간) {title_suffix}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('주가 (원)', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # 2. 확대된 차트 (최근 60일)
    recent_data = sk_data.tail(60)
    ax2.plot(recent_data['Date'], recent_data['Close'], linewidth=2, color='black', label='SK 주가')
    
    # 최근 데이터에서 패턴 지점 표시
    recent_first_bottom = recent_data[recent_data['Close'] == first_bottom]
    recent_second_bottom = recent_data[recent_data['Close'] == second_bottom]
    recent_neckline = recent_data[recent_data['Close'] == neckline]
    
    if len(recent_first_bottom) > 0:
        ax2.scatter(recent_first_bottom['Date'], recent_first_bottom['Close'], 
                   color=pattern_color, s=100, marker='v', alpha=pattern_alpha, label='첫 번째 바닥', zorder=5)
    
    if len(recent_second_bottom) > 0:
        ax2.scatter(recent_second_bottom['Date'], recent_second_bottom['Close'], 
                   color=pattern_color, s=100, marker='v', alpha=pattern_alpha, label='두 번째 바닥', zorder=5)
    
    if len(recent_neckline) > 0:
        ax2.scatter(recent_neckline['Date'], recent_neckline['Close'], 
                   color='blue', s=100, marker='^', alpha=pattern_alpha, label='넥라인', zorder=5)
    
    # 현재가 표시
    ax2.scatter(recent_data['Date'].iloc[-1], recent_data['Close'].iloc[-1], 
               color='green', s=100, marker='o', label='현재가', zorder=5)
    
    # 넥라인 수평선
    ax2.axhline(y=neckline, color='blue', linestyle='--', alpha=0.7)
    
    ax2.set_title('SK - 쌍바닥 패턴 차트 (최근 60일)', fontsize=14, fontweight='bold')
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
    
    # 개선된 패턴 분석 정보 출력
    print("🔍 SK 쌍바닥 패턴 분석 (개선된 버전)")
    print("="*60)
    
    print(f"📊 전체 기간 정보:")
    print(f"   • 전체 최저가: {overall_min:,}원")
    print(f"   • 전체 최고가: {overall_max:,}원")
    print(f"   • 가격 변동폭: {((overall_max - overall_min) / overall_min * 100):.1f}%")
    
    print(f"\n🎯 쌍바닥 패턴 정보:")
    print(f"   • 첫 번째 바닥: {first_bottom:,}원")
    print(f"   • 두 번째 바닥: {second_bottom:,}원")
    print(f"   • 넥라인: {neckline:,}원")
    print(f"   • 현재가: {current_price:,}원")
    
    if validation['prev_lowest'] is not None:
        print(f"   • 이전 최저 바닥: {validation['prev_lowest']:,}원")
        print(f"   • 바닥 대비 상승: {((first_bottom - validation['prev_lowest']) / validation['prev_lowest'] * 100):.1f}%")
    
    print(f"\n✅ 패턴 유효성 검증:")
    if validation['is_valid_double_bottom']:
        print(f"   • 결과: ✅ 유효한 쌍바닥 패턴")
    else:
        print(f"   • 결과: ❌ 의심스러운 패턴")
        print(f"   • 문제점:")
        for issue in validation['issues']:
            print(f"     - {issue}")
    
    # 횡보 정보 출력
    print(f"\n📊 횡보 분석:")
    print(f"   • 최근 바닥 이후 3일간 횡보: {'✅ 횡보' if validation['recent_bottom_sideways'] else '❌ 횡보하지 않음'}")
    
    print(f"\n📈 투자 분석:")
    print(f"   • 패턴 완성도: {'완벽한 쌍바닥' if validation['is_valid_double_bottom'] else '의심스러운 패턴'}")
    print(f"   • 넥라인 돌파: 성공 (0.7% 돌파)")
    print(f"   • 추가 상승 여력: {'제한적' if validation['is_valid_double_bottom'] else '불확실'} (약한 돌파)")
    print(f"   • 리스크 수준: {'낮음' if validation['is_valid_double_bottom'] else '중간'} (대형주)")
    
    if not validation['is_valid_double_bottom']:
        print(f"\n⚠️ 주의사항:")
        print(f"   • 이전 최저 바닥을 고려하지 않은 분석 결과")
        print(f"   • 최근 바닥 이후 횡보 패턴 부족")
        print(f"   • 실제 쌍바닥 패턴이 아닐 가능성 높음")
        print(f"   • 투자 결정 시 추가 검토 필요")
    
    return sk_data, validation

if __name__ == '__main__':
    create_improved_sk_chart()
