# -*- coding: utf-8 -*-
"""
상위 15개 쌍바닥 분석 결과 시각화
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """데이터 로드 및 전처리"""
    df = pd.read_csv('top15_double_bottom_results.csv')
    
    # 백분율 컬럼을 소수점 1자리로 반올림
    df['바닥차이(%)'] = df['바닥차이(%)'].round(1)
    df['반등률(%)'] = df['반등률(%)'].round(1)
    df['돌파률(%)'] = df['돌파률(%)'].round(1)
    
    return df

def create_comprehensive_visualization():
    """종합 시각화 생성"""
    df = load_and_prepare_data()
    
    # 전체 레이아웃 설정
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 상위 15개 종목 점수 막대 차트
    ax1 = plt.subplot(3, 3, 1)
    bars = ax1.bar(range(len(df)), df['종합점수'], color='skyblue', alpha=0.7)
    ax1.set_title('상위 15개 종목 종합 점수', fontsize=14, fontweight='bold')
    ax1.set_xlabel('종목 순위')
    ax1.set_ylabel('종합 점수')
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels(df['종목'], rotation=45, ha='right')
    ax1.set_ylim(95, 101)
    
    # 막대 위에 점수 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 돌파률 분포
    ax2 = plt.subplot(3, 3, 2)
    colors = ['red' if x > 50 else 'orange' if x > 10 else 'green' for x in df['돌파률(%)']]
    bars2 = ax2.bar(range(len(df)), df['돌파률(%)'], color=colors, alpha=0.7)
    ax2.set_title('종목별 돌파률 (%)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('종목 순위')
    ax2.set_ylabel('돌파률 (%)')
    ax2.set_xticks(range(len(df)))
    ax2.set_xticklabels(df['종목'], rotation=45, ha='right')
    
    # 막대 위에 돌파률 표시
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(df['돌파률(%)'])*0.01,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 3. 반등률 vs 돌파률 산점도
    ax3 = plt.subplot(3, 3, 3)
    scatter = ax3.scatter(df['반등률(%)'], df['돌파률(%)'], 
                         c=df['종합점수'], cmap='viridis', s=100, alpha=0.7)
    ax3.set_title('반등률 vs 돌파률 관계', fontsize=14, fontweight='bold')
    ax3.set_xlabel('반등률 (%)')
    ax3.set_ylabel('돌파률 (%)')
    
    # 각 점에 종목명 표시
    for i, txt in enumerate(df['종목']):
        ax3.annotate(txt, (df['반등률(%)'].iloc[i], df['돌파률(%)'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 컬러바 추가
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('종합 점수')
    
    # 4. 바닥 가격 분포
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(df['첫번째바닥'], bins=8, color='lightcoral', alpha=0.7, edgecolor='black')
    ax4.set_title('첫 번째 바닥 가격 분포', fontsize=14, fontweight='bold')
    ax4.set_xlabel('바닥 가격 (원)')
    ax4.set_ylabel('종목 수')
    ax4.ticklabel_format(style='plain', axis='x')
    
    # 5. 현재가 vs 바닥가 비교
    ax5 = plt.subplot(3, 3, 5)
    x = np.arange(len(df))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, df['첫번째바닥'], width, label='첫 번째 바닥', 
                   color='lightblue', alpha=0.7)
    bars2 = ax5.bar(x + width/2, df['현재가'], width, label='현재가', 
                   color='lightgreen', alpha=0.7)
    
    ax5.set_title('바닥가 vs 현재가 비교', fontsize=14, fontweight='bold')
    ax5.set_xlabel('종목')
    ax5.set_ylabel('가격 (원)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(df['종목'], rotation=45, ha='right')
    ax5.legend()
    ax5.ticklabel_format(style='plain', axis='y')
    
    # 6. 상승률 계산 및 표시
    df['상승률(%)'] = ((df['현재가'] - df['첫번째바닥']) / df['첫번째바닥'] * 100).round(1)
    
    ax6 = plt.subplot(3, 3, 6)
    colors = ['red' if x > 100 else 'orange' if x > 50 else 'green' for x in df['상승률(%)']]
    bars6 = ax6.bar(range(len(df)), df['상승률(%)'], color=colors, alpha=0.7)
    ax6.set_title('바닥 대비 상승률 (%)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('종목 순위')
    ax6.set_ylabel('상승률 (%)')
    ax6.set_xticks(range(len(df)))
    ax6.set_xticklabels(df['종목'], rotation=45, ha='right')
    
    # 막대 위에 상승률 표시
    for i, bar in enumerate(bars6):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + max(df['상승률(%)'])*0.01,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 7. 넥라인 돌파 성공률
    ax7 = plt.subplot(3, 3, 7)
    breakout_success = (df['돌파률(%)'] > 0).sum()
    breakout_fail = (df['돌파률(%)'] <= 0).sum()
    
    labels = ['돌파 성공', '돌파 실패']
    sizes = [breakout_success, breakout_fail]
    colors = ['lightgreen', 'lightcoral']
    
    ax7.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax7.set_title('넥라인 돌파 성공률', fontsize=14, fontweight='bold')
    
    # 8. 가격대별 분포
    ax8 = plt.subplot(3, 3, 8)
    price_ranges = ['1만원 미만', '1-5만원', '5-10만원', '10-20만원', '20만원 이상']
    price_counts = [
        (df['현재가'] < 10000).sum(),
        ((df['현재가'] >= 10000) & (df['현재가'] < 50000)).sum(),
        ((df['현재가'] >= 50000) & (df['현재가'] < 100000)).sum(),
        ((df['현재가'] >= 100000) & (df['현재가'] < 200000)).sum(),
        (df['현재가'] >= 200000).sum()
    ]
    
    bars8 = ax8.bar(price_ranges, price_counts, color='lightsteelblue', alpha=0.7)
    ax8.set_title('현재가 가격대별 분포', fontsize=14, fontweight='bold')
    ax8.set_ylabel('종목 수')
    ax8.tick_params(axis='x', rotation=45)
    
    # 막대 위에 개수 표시
    for bar in bars8:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # 9. 종합 성과 지표
    ax9 = plt.subplot(3, 3, 9)
    
    # 평균값 계산
    avg_metrics = {
        '평균 점수': df['종합점수'].mean(),
        '평균 반등률': df['반등률(%)'].mean(),
        '평균 돌파률': df['돌파률(%)'].mean(),
        '평균 상승률': df['상승률(%)'].mean()
    }
    
    metrics = list(avg_metrics.keys())
    values = list(avg_metrics.values())
    
    bars9 = ax9.bar(metrics, values, color=['gold', 'lightblue', 'lightgreen', 'lightcoral'], alpha=0.7)
    ax9.set_title('평균 성과 지표', fontsize=14, fontweight='bold')
    ax9.set_ylabel('값')
    ax9.tick_params(axis='x', rotation=45)
    
    # 막대 위에 값 표시
    for bar in bars9:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return df

def create_detailed_table():
    """상세 정보 테이블 생성"""
    df = load_and_prepare_data()
    
    # 상승률 계산
    df['상승률(%)'] = ((df['현재가'] - df['첫번째바닥']) / df['첫번째바닥'] * 100).round(1)
    
    # 테이블용 데이터 준비
    table_data = df[['종목', '종합점수', '첫번째바닥', '현재가', '상승률(%)', 
                    '반등률(%)', '돌파률(%)']].copy()
    
    # 가격을 천원 단위로 변환
    table_data['첫번째바닥(천원)'] = (table_data['첫번째바닥'] / 1000).round(1)
    table_data['현재가(천원)'] = (table_data['현재가'] / 1000).round(1)
    
    # 최종 테이블
    final_table = table_data[['종목', '종합점수', '첫번째바닥(천원)', '현재가(천원)', 
                             '상승률(%)', '반등률(%)', '돌파률(%)']]
    
    print("\n" + "="*100)
    print("🏆 상위 15개 쌍바닥 종목 상세 분석 결과")
    print("="*100)
    print(final_table.to_string(index=False))
    
    # 통계 요약
    print(f"\n📊 분석 요약:")
    print(f"   • 총 분석 종목: {len(df)}개")
    print(f"   • 평균 종합 점수: {df['종합점수'].mean():.1f}점")
    print(f"   • 평균 상승률: {df['상승률(%)'].mean():.1f}%")
    print(f"   • 평균 반등률: {df['반등률(%)'].mean():.1f}%")
    print(f"   • 평균 돌파률: {df['돌파률(%)'].mean():.1f}%")
    print(f"   • 최고 상승률: {df['상승률(%)'].max():.1f}% ({df.loc[df['상승률(%)'].idxmax(), '종목']})")
    print(f"   • 최고 돌파률: {df['돌파률(%)'].max():.1f}% ({df.loc[df['돌파률(%)'].idxmax(), '종목']})")
    
    return final_table

def main():
    """메인 실행 함수"""
    print("📊 상위 15개 쌍바닥 분석 결과 시각화")
    print("="*50)
    
    try:
        # 종합 시각화 생성
        print("📈 종합 시각화 생성 중...")
        df = create_comprehensive_visualization()
        
        # 상세 테이블 생성
        print("\n📋 상세 분석 테이블 생성 중...")
        table = create_detailed_table()
        
        print("\n✅ 시각화 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
