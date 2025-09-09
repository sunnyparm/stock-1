# -*- coding: utf-8 -*-
"""
두 번째 바닥 분석 및 시각화
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_second_bottom():
    """두 번째 바닥 분석"""
    df = pd.read_csv('top15_double_bottom_results.csv')
    
    print("🔍 두 번째 바닥 분석 결과")
    print("="*80)
    
    # 두 번째 바닥 정보 정리
    second_bottom_data = df[['종목', '첫번째바닥', '두번째바닥', '바닥차이(%)', '넥라인', '현재가']].copy()
    
    # 가격을 천원 단위로 변환
    second_bottom_data['첫번째바닥(천원)'] = (second_bottom_data['첫번째바닥'] / 1000).round(1)
    second_bottom_data['두번째바닥(천원)'] = (second_bottom_data['두번째바닥'] / 1000).round(1)
    second_bottom_data['넥라인(천원)'] = (second_bottom_data['넥라인'] / 1000).round(1)
    second_bottom_data['현재가(천원)'] = (second_bottom_data['현재가'] / 1000).round(1)
    
    # 바닥차이를 소수점 1자리로 반올림
    second_bottom_data['바닥차이(%)'] = second_bottom_data['바닥차이(%)'].round(1)
    
    # 최종 테이블
    final_table = second_bottom_data[['종목', '첫번째바닥(천원)', '두번째바닥(천원)', 
                                     '바닥차이(%)', '넥라인(천원)', '현재가(천원)']]
    
    print("📊 두 번째 바닥 상세 정보:")
    print(final_table.to_string(index=False))
    
    # 통계 분석
    print(f"\n📈 두 번째 바닥 통계 분석:")
    print(f"   • 평균 첫 번째 바닥: {second_bottom_data['첫번째바닥(천원)'].mean():.1f}천원")
    print(f"   • 평균 두 번째 바닥: {second_bottom_data['두번째바닥(천원)'].mean():.1f}천원")
    print(f"   • 평균 바닥 차이: {second_bottom_data['바닥차이(%)'].mean():.1f}%")
    print(f"   • 최대 바닥 차이: {second_bottom_data['바닥차이(%)'].max():.1f}%")
    print(f"   • 최소 바닥 차이: {second_bottom_data['바닥차이(%)'].min():.1f}%")
    
    # 완벽한 쌍바닥 (차이가 0%인 종목들)
    perfect_double_bottom = second_bottom_data[second_bottom_data['바닥차이(%)'] == 0.0]
    print(f"   • 완벽한 쌍바닥 종목: {len(perfect_double_bottom)}개")
    print(f"     - {', '.join(perfect_double_bottom['종목'].tolist())}")
    
    return second_bottom_data

def create_second_bottom_visualization():
    """두 번째 바닥 시각화"""
    df = pd.read_csv('top15_double_bottom_results.csv')
    
    # 전체 레이아웃 설정
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('두 번째 바닥 분석 시각화', fontsize=16, fontweight='bold')
    
    # 1. 첫 번째 vs 두 번째 바닥 비교
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['첫번째바닥']/1000, width, label='첫 번째 바닥', 
                   color='lightblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, df['두번째바닥']/1000, width, label='두 번째 바닥', 
                   color='lightcoral', alpha=0.7)
    
    ax1.set_title('첫 번째 vs 두 번째 바닥 비교', fontsize=12, fontweight='bold')
    ax1.set_xlabel('종목')
    ax1.set_ylabel('바닥 가격 (천원)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['종목'], rotation=45, ha='right')
    ax1.legend()
    
    # 2. 바닥 차이 분포
    ax2 = axes[0, 1]
    colors = ['green' if x == 0 else 'orange' if x < 5 else 'red' for x in df['바닥차이(%)']]
    bars2 = ax2.bar(range(len(df)), df['바닥차이(%)'], color=colors, alpha=0.7)
    ax2.set_title('바닥 차이 분포 (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('종목 순위')
    ax2.set_ylabel('바닥 차이 (%)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['종목'], rotation=45, ha='right')
    
    # 막대 위에 차이 표시
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. 바닥 차이 히스토그램
    ax3 = axes[0, 2]
    ax3.hist(df['바닥차이(%)'], bins=8, color='lightgreen', alpha=0.7, edgecolor='black')
    ax3.set_title('바닥 차이 분포 히스토그램', fontsize=12, fontweight='bold')
    ax3.set_xlabel('바닥 차이 (%)')
    ax3.set_ylabel('종목 수')
    ax3.axvline(df['바닥차이(%)'].mean(), color='red', linestyle='--', 
                label=f'평균: {df["바닥차이(%)"].mean():.1f}%')
    ax3.legend()
    
    # 4. 바닥에서 넥라인까지의 상승률
    ax4 = axes[1, 0]
    first_bottom_to_neck = ((df['넥라인'] - df['첫번째바닥']) / df['첫번째바닥'] * 100).round(1)
    second_bottom_to_neck = ((df['넥라인'] - df['두번째바닥']) / df['두번째바닥'] * 100).round(1)
    
    x = np.arange(len(df))
    bars1 = ax4.bar(x - width/2, first_bottom_to_neck, width, label='첫 번째 바닥→넥라인', 
                   color='lightblue', alpha=0.7)
    bars2 = ax4.bar(x + width/2, second_bottom_to_neck, width, label='두 번째 바닥→넥라인', 
                   color='lightcoral', alpha=0.7)
    
    ax4.set_title('바닥에서 넥라인까지 상승률', fontsize=12, fontweight='bold')
    ax4.set_xlabel('종목')
    ax4.set_ylabel('상승률 (%)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(df['종목'], rotation=45, ha='right')
    ax4.legend()
    
    # 5. 바닥 유사성 점수
    ax5 = axes[1, 1]
    # 바닥 차이가 작을수록 높은 점수 (100점 만점)
    similarity_scores = (100 - df['바닥차이(%)']).clip(0, 100)
    colors = ['green' if x == 100 else 'orange' if x > 95 else 'red' for x in similarity_scores]
    
    bars5 = ax5.bar(range(len(df)), similarity_scores, color=colors, alpha=0.7)
    ax5.set_title('바닥 유사성 점수', fontsize=12, fontweight='bold')
    ax5.set_xlabel('종목 순위')
    ax5.set_ylabel('유사성 점수')
    ax5.set_xticks(range(len(df)))
    ax5.set_xticklabels(df['종목'], rotation=45, ha='right')
    ax5.set_ylim(95, 101)
    
    # 막대 위에 점수 표시
    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 6. 바닥 차이 vs 현재 상승률 관계
    ax6 = axes[1, 2]
    current_rise = ((df['현재가'] - df['두번째바닥']) / df['두번째바닥'] * 100).round(1)
    
    scatter = ax6.scatter(df['바닥차이(%)'], current_rise, 
                         c=df['종합점수'], cmap='viridis', s=100, alpha=0.7)
    ax6.set_title('바닥 차이 vs 현재 상승률', fontsize=12, fontweight='bold')
    ax6.set_xlabel('바닥 차이 (%)')
    ax6.set_ylabel('두 번째 바닥 대비 상승률 (%)')
    
    # 각 점에 종목명 표시
    for i, txt in enumerate(df['종목']):
        ax6.annotate(txt, (df['바닥차이(%)'].iloc[i], current_rise.iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 컬러바 추가
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('종합 점수')
    
    plt.tight_layout()
    plt.show()
    
    return df

def main():
    """메인 실행 함수"""
    print("🔍 두 번째 바닥 분석 시작")
    print("="*50)
    
    try:
        # 두 번째 바닥 분석
        second_bottom_data = analyze_second_bottom()
        
        # 시각화 생성
        print("\n📊 두 번째 바닥 시각화 생성 중...")
        create_second_bottom_visualization()
        
        print("\n✅ 두 번째 바닥 분석 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
