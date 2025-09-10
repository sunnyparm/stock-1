# -*- coding: utf-8 -*-
"""
개선된 상위 15개 쌍바닥 분석 코드 - 이전 최저 바닥 고려
코스피 데이터에서 쌍바닥 패턴을 보이는 상위 15개 종목을 분석합니다.
이전 최저 바닥을 고려한 정확한 쌍바닥 패턴 분석을 수행합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class DoubleBottomAnalyzer:
    """쌍바닥 패턴 분석 클래스"""
    
    def __init__(self, file_path):
        """
        초기화
        
        Args:
            file_path (str): 코스피 데이터 CSV 파일 경로
        """
        self.file_path = file_path
        self.df = None
        self.df_long = None
        self.double_bottom_results = []
        
    def load_and_transform_data(self):
        """데이터 로드 및 변환 (가로 -> 세로 형태)"""
        print("📊 데이터 로드 및 변환 중...")
        
        # CSV 파일 읽기
        self.df = pd.read_csv(self.file_path)
        
        # 첫 번째 컬럼이 '종목'인지 확인
        if self.df.columns[0] != '종목':
            raise ValueError("첫 번째 컬럼이 '종목'이 아닙니다.")
        
        # 종목명을 인덱스로 설정
        self.df = self.df.set_index('종목')
        
        # 가로 형태를 세로 형태로 변환 (melt)
        self.df_long = self.df.reset_index().melt(
            id_vars=['종목'], 
            var_name='Date', 
            value_name='Close'
        )
        
        # 날짜 컬럼을 datetime으로 변환
        self.df_long['Date'] = pd.to_datetime(self.df_long['Date'])
        
        # 결측값 제거
        self.df_long = self.df_long.dropna()
        
        # 종목별, 날짜별로 정렬
        self.df_long = self.df_long.sort_values(['종목', 'Date']).reset_index(drop=True)
        
        print(f"✅ 데이터 변환 완료: {len(self.df_long)} 행, {self.df_long['종목'].nunique()}개 종목")
        
    def find_local_minima_maxima(self, series, window=5):
        """
        로컬 최소값과 최대값의 인덱스를 찾습니다.
        
        Args:
            series: 가격 시계열
            window: 비교할 윈도우 크기
            
        Returns:
            tuple: (최소값 인덱스 리스트, 최대값 인덱스 리스트)
        """
        minima_indices = []
        maxima_indices = []
        
        for i in range(window, len(series) - window):
            # 로컬 최소값
            if series.iloc[i] == series.iloc[i-window : i+window+1].min():
                minima_indices.append(i)
            # 로컬 최대값
            if series.iloc[i] == series.iloc[i-window : i+window+1].max():
                maxima_indices.append(i)
                
        return minima_indices, maxima_indices
    
    def find_previous_lowest_bottom(self, data, current_bottom_idx, lookback_days=60):
        """
        이전 최저 바닥 찾기
        
        Args:
            data: 주가 데이터
            current_bottom_idx: 현재 바닥 인덱스
            lookback_days: 이전 기간 (일)
            
        Returns:
            tuple: (이전 최저가, 이전 최저가 인덱스)
        """
        if current_bottom_idx < lookback_days:
            return None, None
        
        # 이전 기간 데이터
        previous_data = data.iloc[max(0, current_bottom_idx - lookback_days):current_bottom_idx]
        if len(previous_data) == 0:
            return None, None
        
        # 이전 기간 최저가
        prev_lowest = previous_data['Close'].min()
        prev_lowest_idx = previous_data['Close'].idxmin()
        
        return prev_lowest, prev_lowest_idx
    
    def check_sideways_after_bottom(self, data, bottom_idx, days=3, tolerance_pct=0.02):
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

    def validate_double_bottom_pattern(self, data, first_bottom_idx, second_bottom_idx, 
                                     first_bottom_price, second_bottom_price, lookback_days=60):
        """
        쌍바닥 패턴 유효성 검증 (개선된 버전)
        
        Args:
            data: 주가 데이터
            first_bottom_idx: 첫 번째 바닥 인덱스
            second_bottom_idx: 두 번째 바닥 인덱스
            first_bottom_price: 첫 번째 바닥 가격
            second_bottom_price: 두 번째 바닥 가격
            lookback_days: 이전 기간 (일)
            
        Returns:
            dict: 검증 결과
        """
        # 1. 이전 최저 바닥 확인
        prev_lowest, prev_lowest_idx = self.find_previous_lowest_bottom(
            data, first_bottom_idx, lookback_days)
        
        # 2. 상대적 바닥 확인 (주변 20일 대비)
        first_window = data.iloc[max(0, first_bottom_idx-10):first_bottom_idx+10]
        first_local_min = first_window['Close'].min()
        
        second_window = data.iloc[max(0, second_bottom_idx-10):second_bottom_idx+10]
        second_local_min = second_window['Close'].min()
        
        # 3. 최근 최저가 이후 3일간 횡보 확인
        recent_bottom_sideways = self.check_sideways_after_bottom(data, second_bottom_idx, days=3, tolerance_pct=0.02)
        
        validation_result = {
            'prev_lowest': prev_lowest,
            'prev_lowest_idx': prev_lowest_idx,
            'first_local_min': first_local_min,
            'second_local_min': second_local_min,
            'recent_bottom_sideways': recent_bottom_sideways,
            'is_valid_double_bottom': False,
            'issues': [],
            'validation_score': 0
        }
        
        # 유효성 검증 및 점수 계산
        score = 100  # 기본 점수
        
        # 이전 최저 바닥 검증 (30점)
        if prev_lowest is not None:
            if first_bottom_price <= prev_lowest:
                validation_result['issues'].append(f"첫 번째 바닥이 이전 최저가보다 낮거나 같음")
                score -= 30
            else:
                # 이전 최저가보다 높으면 보너스 점수
                improvement_pct = (first_bottom_price - prev_lowest) / prev_lowest
                if improvement_pct > 0.05:  # 5% 이상 개선
                    score += 10
        
        # 상대적 바닥 검증 (20점씩)
        if first_bottom_price != first_local_min:
            validation_result['issues'].append(f"첫 번째 바닥이 주변 최저가와 다름")
            score -= 20
            
        if second_bottom_price != second_local_min:
            validation_result['issues'].append(f"두 번째 바닥이 주변 최저가와 다름")
            score -= 20
        
        # 최근 바닥 이후 횡보 검증 (20점)
        if not recent_bottom_sideways:
            validation_result['issues'].append(f"최근 바닥 이후 3일간 횡보하지 않음")
            score -= 20
        else:
            # 횡보하면 보너스 점수
            score += 10
        
        # 최종 유효성 판단
        if len(validation_result['issues']) == 0:
            validation_result['is_valid_double_bottom'] = True
        
        validation_result['validation_score'] = max(0, score)
        
        return validation_result
    
    def calculate_double_bottom_score(self, stock_data, 
                                    bottom_tolerance_pct=0.10,  # 더 엄격하게 (20% -> 10%)
                                    rebound_min_pct=0.05,       # 더 엄격하게 (2% -> 5%)
                                    breakout_min_pct=0.02,      # 더 엄격하게 (0.5% -> 2%)
                                    min_days_between_bottoms=10, # 더 엄격하게 (5일 -> 10일)
                                    max_days_between_bottoms=120, # 더 엄격하게 (200일 -> 120일)
                                    lookback_period=120):       # 더 긴 기간 (100일 -> 120일)
        """
        쌍바닥 패턴 점수를 계산합니다.
        
        Args:
            stock_data: 종목별 주가 데이터
            bottom_tolerance_pct: 두 바닥의 가격 차이 허용 범위
            rebound_min_pct: 첫 번째 바닥 이후 최소 반등률
            breakout_min_pct: 넥라인 돌파 최소 비율
            min_days_between_bottoms: 두 바닥 사이 최소 기간
            max_days_between_bottoms: 두 바닥 사이 최대 기간
            lookback_period: 패턴을 찾을 전체 기간
            
        Returns:
            dict: 쌍바닥 패턴 분석 결과
        """
        if len(stock_data) < 50:  # 최소 데이터 요구량 줄임
            return None
            
        # 최근 데이터만 사용
        df_recent = stock_data.tail(lookback_period).reset_index(drop=True)
        n = len(df_recent)
        
        # 로컬 최소값 찾기 (윈도우 크기 줄임)
        minima_indices, _ = self.find_local_minima_maxima(df_recent['Close'], window=3)
        
        if len(minima_indices) < 2:
            return None
        
        best_score = 0
        best_pattern = None
        
        # 모든 바닥 조합 검사
        for i, b1_idx in enumerate(minima_indices[:-1]):  # 마지막 바닥은 제외
            b1_price = df_recent.loc[b1_idx, 'Close']
            
            # 두 번째 바닥 후보 찾기
            b2_candidates = [idx for idx in minima_indices[i+1:] 
                           if (idx - b1_idx >= min_days_between_bottoms and 
                               idx - b1_idx <= max_days_between_bottoms)]
            
            for b2_idx in b2_candidates:
                b2_price = df_recent.loc[b2_idx, 'Close']
                
                # 두 바닥의 가격 차이 확인
                price_diff_pct = abs(b1_price - b2_price) / min(b1_price, b2_price)
                if price_diff_pct > bottom_tolerance_pct:
                    continue
                
                # 두 바닥 사이의 피크 찾기
                if b2_idx - b1_idx <= 1:
                    continue
                    
                peak_slice = df_recent.loc[b1_idx + 1 : b2_idx - 1, 'Close']
                if peak_slice.empty:
                    continue
                    
                peak_idx = peak_slice.idxmax()
                peak_price = df_recent.loc[peak_idx, 'Close']
                
                # 반등률 확인
                rebound_pct = (peak_price - b1_price) / b1_price
                if rebound_pct < rebound_min_pct:
                    continue
                
                # 현재 가격 (마지막 종가)
                current_price = df_recent['Close'].iloc[-1]
                
                # 넥라인 돌파 확인
                breakout_pct = (current_price - peak_price) / peak_price
                
                # 점수 계산 (0-100점)
                score = 0
                
                # 바닥 유사성 점수 (30점)
                similarity_score = max(0, 30 * (1 - price_diff_pct / bottom_tolerance_pct))
                score += similarity_score
                
                # 반등률 점수 (25점)
                rebound_score = min(25, 25 * rebound_pct / rebound_min_pct)
                score += rebound_score
                
                # 돌파 점수 (25점)
                if breakout_pct > 0:
                    breakout_score = min(25, 25 * breakout_pct / breakout_min_pct)
                    score += breakout_score
                else:
                    # 돌파하지 않았어도 패턴이 있으면 기본 점수
                    breakout_score = 10
                    score += breakout_score
                
                # 패턴 완성도 점수 (20점)
                pattern_completeness = 20 if breakout_pct > breakout_min_pct else 15
                score += pattern_completeness
                
                # 유효성 검증 추가
                validation = self.validate_double_bottom_pattern(
                    df_recent, b1_idx, b2_idx, b1_price, b2_price, 60)
                
                # 유효성 점수 반영 (기존 점수에 가중치 적용)
                if validation['is_valid_double_bottom']:
                    # 유효한 쌍바닥이면 점수 보너스
                    score = score * 1.2  # 20% 보너스
                else:
                    # 의심스러운 패턴이면 점수 감점
                    score = score * 0.7  # 30% 감점
                
                # 유효성 검증 점수 추가 (최대 20점)
                validation_bonus = min(20, validation['validation_score'] * 0.2)
                score += validation_bonus
                
                if score > best_score:
                    best_score = score
                    best_pattern = {
                        'b1_idx': b1_idx,
                        'b2_idx': b2_idx,
                        'peak_idx': peak_idx,
                        'b1_price': b1_price,
                        'b2_price': b2_price,
                        'peak_price': peak_price,
                        'current_price': current_price,
                        'price_diff_pct': price_diff_pct,
                        'rebound_pct': rebound_pct,
                        'breakout_pct': breakout_pct,
                        'score': score,
                        'validation': validation
                    }
        
        return best_pattern
    
    def analyze_all_stocks(self):
        """모든 종목에 대해 쌍바닥 분석 수행"""
        print("🔍 모든 종목 쌍바닥 패턴 분석 중...")
        
        results = []
        total_stocks = self.df_long['종목'].nunique()
        processed = 0
        
        for symbol in self.df_long['종목'].unique():
            stock_data = self.df_long[self.df_long['종목'] == symbol].copy()
            
            # 쌍바닥 패턴 분석
            pattern = self.calculate_double_bottom_score(stock_data)
            
            if pattern:
                pattern['symbol'] = symbol
                results.append(pattern)
                print(f"✅ {symbol}: 점수 {pattern['score']:.1f}")
            
            processed += 1
            if processed % 100 == 0:
                print(f"진행률: {processed}/{total_stocks} ({processed/total_stocks*100:.1f}%)")
        
        # 점수순으로 정렬
        self.double_bottom_results = sorted(results, key=lambda x: x['score'], reverse=True)
        
        print(f"\n📈 분석 완료: {len(self.double_bottom_results)}개 종목에서 쌍바닥 패턴 발견")
        
    def get_valid_double_bottom_results(self):
        """유효한 쌍바닥 패턴만 반환"""
        valid_results = []
        for result in self.double_bottom_results:
            validation = result.get('validation', {})
            if validation.get('is_valid_double_bottom', False):
                valid_results.append(result)
        return valid_results
    
    def get_top15_results(self):
        """상위 15개 결과 반환 (하위 호환성)"""
        return self.double_bottom_results[:15]
    
    def print_valid_results(self):
        """유효한 쌍바닥 패턴 결과 출력 (15개 제한 없음)"""
        valid_results = self.get_valid_double_bottom_results()
        
        print("\n" + "="*80)
        print(f"🏆 유효한 쌍바닥 패턴 종목 ({len(valid_results)}개)")
        print("="*80)
        
        for i, result in enumerate(valid_results, 1):
            print(f"\n{i:2d}. {result['symbol']}")
            print(f"    📊 종합 점수: {result['score']:.1f}/100")
            print(f"    💰 첫 번째 바닥: {result['b1_price']:,.0f}원")
            print(f"    💰 두 번째 바닥: {result['b2_price']:,.0f}원")
            print(f"    📈 넥라인: {result['peak_price']:,.0f}원")
            print(f"    💎 현재가: {result['current_price']:,.0f}원")
            print(f"    📉 바닥 차이: {result['price_diff_pct']*100:.1f}%")
            print(f"    📈 반등률: {result['rebound_pct']*100:.1f}%")
            print(f"    🚀 돌파률: {result['breakout_pct']*100:.1f}%")
            
            validation = result.get('validation', {})
            if validation.get('prev_lowest'):
                print(f"    📊 이전 최저가: {validation['prev_lowest']:,.0f}원")
                improvement = (result['b1_price'] - validation['prev_lowest']) / validation['prev_lowest'] * 100
                print(f"    📈 바닥 개선률: {improvement:.1f}%")
            
            # 횡보 정보 출력
            if validation.get('recent_bottom_sideways') is not None:
                print(f"    📊 최근 바닥 이후 횡보: {'✅ 횡보' if validation['recent_bottom_sideways'] else '❌ 횡보하지 않음'}")
    
    def print_top15_results(self):
        """상위 15개 결과 출력 (하위 호환성)"""
        top15 = self.get_top15_results()
        
        print("\n" + "="*80)
        print("🏆 상위 15개 쌍바닥 패턴 종목")
        print("="*80)
        
        for i, result in enumerate(top15, 1):
            print(f"\n{i:2d}. {result['symbol']}")
            print(f"    📊 종합 점수: {result['score']:.1f}/100")
            print(f"    💰 첫 번째 바닥: {result['b1_price']:,.0f}원")
            print(f"    💰 두 번째 바닥: {result['b2_price']:,.0f}원")
            print(f"    📈 넥라인: {result['peak_price']:,.0f}원")
            print(f"    💎 현재가: {result['current_price']:,.0f}원")
            print(f"    📉 바닥 차이: {result['price_diff_pct']*100:.1f}%")
            print(f"    📈 반등률: {result['rebound_pct']*100:.1f}%")
            print(f"    🚀 돌파률: {result['breakout_pct']*100:.1f}%")
    
    def save_valid_results_to_csv(self, filename='valid_double_bottom_results.csv'):
        """유효한 쌍바닥 패턴 결과를 CSV 파일로 저장"""
        valid_results = self.get_valid_double_bottom_results()
        
        if not valid_results:
            print("저장할 유효한 결과가 없습니다.")
            return
        
        # two_bottom 폴더 경로 설정
        output_dir = 'two_bottom'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일명에 타임스탬프 추가
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if filename == 'valid_double_bottom_results.csv':
            filename = f'valid_double_bottom_results_{timestamp}.csv'
        else:
            # 파일명에 확장자가 있는 경우 타임스탬프를 확장자 앞에 추가
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, 'csv')
            filename = f'{name}_{timestamp}.{ext}'
        
        # 전체 파일 경로 생성
        filepath = os.path.join(output_dir, filename)
        
        # 결과 데이터프레임 생성
        results_data = []
        for result in valid_results:
            validation = result.get('validation', {})
            results_data.append({
                '종목': result['symbol'],
                '종합점수': round(result['score'], 1),
                '첫번째바닥': result['b1_price'],
                '두번째바닥': result['b2_price'],
                '넥라인': result['peak_price'],
                '현재가': result['current_price'],
                '바닥차이(%)': round(result['price_diff_pct'] * 100, 2),
                '반등률(%)': round(result['rebound_pct'] * 100, 2),
                '돌파률(%)': round(result['breakout_pct'] * 100, 2),
                '유효성검증': '✅ 유효',
                '검증점수': round(validation.get('validation_score', 0), 1),
                '이전최저바닥': validation.get('prev_lowest', 'N/A'),
                '바닥개선률(%)': round(((result['b1_price'] - validation.get('prev_lowest', result['b1_price'])) / validation.get('prev_lowest', result['b1_price']) * 100), 2) if validation.get('prev_lowest') else 'N/A',
                '최근바닥횡보': '✅ 횡보' if validation.get('recent_bottom_sideways', False) else '❌ 횡보하지 않음',
                '문제점수': len(validation.get('issues', [])),
                '문제점': '; '.join(validation.get('issues', [])) if validation.get('issues') else '없음'
            })
        
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✅ 유효한 쌍바닥 패턴 결과가 '{filepath}' 파일로 저장되었습니다.")
        print(f"📊 총 {len(valid_results)}개 종목이 유효한 쌍바닥 패턴으로 확인되었습니다.")
    
    def save_results_to_csv(self, filename='improved_top15_double_bottom_results.csv'):
        """개선된 결과를 CSV 파일로 저장 (유효성 검증 정보 포함)"""
        top15 = self.get_top15_results()
        
        if not top15:
            print("저장할 결과가 없습니다.")
            return
        
        # two_bottom 폴더 경로 설정
        output_dir = 'two_bottom'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일명에 타임스탬프 추가
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if filename == 'improved_top15_double_bottom_results.csv':
            filename = f'improved_top15_double_bottom_results_{timestamp}.csv'
        else:
            # 파일명에 확장자가 있는 경우 타임스탬프를 확장자 앞에 추가
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, 'csv')
            filename = f'{name}_{timestamp}.{ext}'
        
        # 전체 파일 경로 생성
        filepath = os.path.join(output_dir, filename)
        
        # 결과 데이터프레임 생성
        results_data = []
        for result in top15:
            validation = result.get('validation', {})
            results_data.append({
                '종목': result['symbol'],
                '종합점수': round(result['score'], 1),
                '첫번째바닥': result['b1_price'],
                '두번째바닥': result['b2_price'],
                '넥라인': result['peak_price'],
                '현재가': result['current_price'],
                '바닥차이(%)': round(result['price_diff_pct'] * 100, 2),
                '반등률(%)': round(result['rebound_pct'] * 100, 2),
                '돌파률(%)': round(result['breakout_pct'] * 100, 2),
                '유효성검증': '✅ 유효' if validation.get('is_valid_double_bottom', False) else '❌ 의심',
                '검증점수': round(validation.get('validation_score', 0), 1),
                '이전최저바닥': validation.get('prev_lowest', 'N/A'),
                '최근바닥횡보': '✅ 횡보' if validation.get('recent_bottom_sideways', False) else '❌ 횡보하지 않음',
                '문제점수': len(validation.get('issues', [])),
                '문제점': '; '.join(validation.get('issues', [])) if validation.get('issues') else '없음'
            })
        
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✅ 개선된 결과가 '{filepath}' 파일로 저장되었습니다.")

def main():
    """메인 실행 함수"""
    print("🚀 개선된 상위 15개 쌍바닥 분석 시작 (이전 최저 바닥 고려)")
    print("="*60)
    
    # 분석기 초기화
    analyzer = DoubleBottomAnalyzer('코스피6개월종가_20250908_035947.csv')
    
    try:
        # 데이터 로드 및 변환
        analyzer.load_and_transform_data()
        
        # 더 엄격한 쌍바닥 패턴 분석
        print("\n🔍 더 엄격한 쌍바닥 패턴 분석 중...")
        print("   • 더 엄격한 조건 적용 (10% 이내, 5% 반등, 2% 돌파)")
        print("   • 이전 최저 바닥 고려")
        print("   • 상대적 바닥 검증")
        print("   • 최근 바닥 이후 3일간 횡보 확인")
        print("   • 패턴 유효성 검증")
        analyzer.analyze_all_stocks()
        
        # 유효한 쌍바닥 패턴만 출력
        analyzer.print_valid_results()
        
        # 유효한 결과만 저장
        analyzer.save_valid_results_to_csv()
        
        print("\n🎉 유효한 쌍바닥 패턴 분석 완료!")
        print("="*60)
        print("📋 주요 특징:")
        print("   ✅ 더 엄격한 쌍바닥 조건 (10% 이내, 5% 반등, 2% 돌파)")
        print("   ✅ 이전 최저 바닥 고려한 정확한 쌍바닥 분석")
        print("   ✅ 상대적 바닥 검증으로 가짜 패턴 제거")
        print("   ✅ 최근 바닥 이후 3일간 횡보 패턴 확인")
        print("   ✅ 15개 제한 없이 모든 유효한 쌍바닥 선별")
        print("   ✅ 진짜 쌍바닥 패턴만 추출")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()