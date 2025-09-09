import pandas as pd
import numpy as np

def find_local_minima_maxima(series, window=5):
    """
    주어진 시계열에서 로컬 최소값과 최대값의 인덱스를 찾습니다.
    window: 해당 지점의 앞뒤 몇 개의 지점과 비교할지 결정합니다.
    """
    minima_indices = []
    maxima_indices = []
    # 윈도우 크기만큼의 여유를 두고 탐색
    for i in range(window, len(series) - window):
        # 현재 지점이 윈도우 내에서 가장 낮은 값인 경우
        if series.iloc[i] == series.iloc[i-window : i+window+1].min():
            minima_indices.append(i)
        # 현재 지점이 윈도우 내에서 가장 높은 값인 경우
        if series.iloc[i] == series.iloc[i-window : i+window+1].max():
            maxima_indices.append(i)
    return minima_indices, maxima_indices

def find_double_bottom(df_stock,
                       bottom_tolerance_pct=0.03,  # 두 바닥의 가격 차이 허용 범위 (3%)
                       rebound_min_pct=0.05,       # 첫 번째 바닥 이후 최소 반등률 (5%)
                       breakout_min_pct=0.03,      # 넥라인 돌파 최소 비율 (3%)
                       min_days_between_bottoms=20, # 두 바닥 사이 최소 기간 (거래일 기준)
                       max_days_between_bottoms=120, # 두 바닥 사이 최대 기간
                       lookback_period=252,        # 패턴을 찾을 전체 기간 (1년)
                       local_extremum_window=5):   # 로컬 극점 탐색 윈도우
    """
    주어진 주식 데이터프레임에서 쌍바닥 패턴을 탐지합니다.
    df_stock은 'Close' 컬럼을 포함해야 하며, 인덱스는 순차적인 정수 인덱스여야 합니다.
    """
    # 패턴 탐색에 필요한 최소 데이터 길이 확인
    # lookback_period + local_extremum_window * 2 는 로컬 극점 탐색 윈도우를 고려한 최소 길이
    if len(df_stock) < lookback_period + local_extremum_window * 2:
        return False

    # 최근 lookback_period 기간 동안의 데이터만 사용
    # 인덱스를 리셋하여 순차적인 정수 인덱스로 작업하는 것이 편리
    df_recent = df_stock.tail(lookback_period).reset_index(drop=True)
    
    n = len(df_recent)
    
    # 로컬 최소값 인덱스 찾기 (바닥 후보)
    minima_indices, _ = find_local_minima_maxima(df_recent['Close'], window=local_extremum_window)

    # 패턴 탐색
    for i_b1 in minima_indices:
        b1_price = df_recent.loc[i_b1, 'Close']

        # 1. 첫 번째 바닥 (B1) 이후의 두 번째 바닥 (B2) 후보 찾기
        # B2는 B1 이후 min_days_between_bottoms ~ max_days_between_bottoms 기간 내에 있어야 함
        i_b2_candidates = [idx for idx in minima_indices 
                           if (i_b1 + min_days_between_bottoms <= idx <= i_b1 + max_days_between_bottoms)]
        
        for i_b2 in i_b2_candidates:
            b2_price = df_recent.loc[i_b2, 'Close']

            # 두 바닥의 가격이 유사한지 확인
            if abs(b1_price - b2_price) / b1_price > bottom_tolerance_pct:
                continue
            
            # 2. 두 바닥 사이의 피크 (Peak, 넥라인) 찾기
            # Peak는 B1과 B2 사이의 최고점이어야 함
            # i_b1 + 1 부터 i_b2 - 1 까지 슬라이싱
            peak_slice = df_recent.loc[i_b1 + 1 : i_b2 - 1, 'Close']
            
            # 바닥 사이에 최소 1일은 있어야 함 (peak_slice가 비어있지 않아야 함)
            if peak_slice.empty: 
                continue
            
            i_peak = peak_slice.idxmax() # 해당 슬라이스 내의 최고점 인덱스
            peak_price = df_recent.loc[i_peak, 'Close']

            # 첫 번째 바닥에서 피크까지 충분히 반등했는지 확인
            if (peak_price - b1_price) / b1_price < rebound_min_pct:
                continue
            
            # 3. 넥라인 돌파 확인 (최근 데이터에서)
            # 두 번째 바닥 이후 현재까지 넥라인(peak_price)을 돌파했는지 확인
            # 현재 가격은 df_recent의 마지막 종가
            current_price = df_recent['Close'].iloc[-1]
            
            # 현재 가격이 넥라인을 충분히 돌파했는지 확인
            if (current_price - peak_price) / peak_price > breakout_min_pct:
                # 쌍바닥 패턴이 성공적으로 완성되고 돌파까지 이루어진 경우
                return True
            
    return False

def detect_double_bottom_stocks(file_path,
                                date_col='Date',
                                symbol_col='Symbol',
                                close_col='Close',
                                **kwargs):
    """
    CSV 파일에서 쌍바닥 패턴을 보이는 종목을 탐지합니다.

    Args:
        file_path (str): 코스피 종가 데이터가 포함된 CSV 파일 경로.
        date_col (str): 날짜 컬럼 이름 (기본값: 'Date').
        symbol_col (str): 종목 코드/심볼 컬럼 이름 (기본값: 'Symbol').
        close_col (str): 종가 컬럼 이름 (기본값: 'Close').
        **kwargs: find_double_bottom 함수에 전달될 추가 파라미터 (예: bottom_tolerance_pct 등).

    Returns:
        list: 쌍바닥 패턴이 탐지된 종목 코드 리스트.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return []
    except Exception as e:
        print(f"CSV 파일을 읽는 중 오류 발생: {e}")
        return []

    # 필요한 컬럼이 있는지 확인
    required_cols = [date_col, symbol_col, close_col]
    if not all(col in df.columns for col in required_cols):
        print(f"오류: CSV 파일에 필요한 컬럼({', '.join(required_cols)})이 없습니다.")
        print(f"현재 컬럼: {df.columns.tolist()}")
        return []

    # 날짜 컬럼을 datetime 객체로 변환하고, 종목 및 날짜 기준으로 정렬
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=[symbol_col, date_col])

    double_bottom_stocks = []
    
    print("쌍바닥 패턴 탐지 시작...")
    # 종목별로 그룹화하여 처리
    for symbol, df_stock in df.groupby(symbol_col):
        # 각 종목별로 쌍바닥 패턴 탐지 함수 호출
        if find_double_bottom(df_stock, **kwargs):
            double_bottom_stocks.append(symbol)
            print(f"✅ 종목 {symbol} 에서 쌍바닥 패턴이 탐지되었습니다.")
        # else:
        #     print(f"❌ 종목 {symbol} 에서 쌍바닥 패턴이 탐지되지 않았습니다.")

    print("\n--- 쌍바닥 패턴 탐지 결과 ---")
    if double_bottom_stocks:
        print(f"총 {len(double_bottom_stocks)}개의 종목에서 쌍바닥 패턴이 발견되었습니다.")
        for stock in double_bottom_stocks:
            print(f"- {stock}")
    else:
        print("쌍바닥 패턴을 보이는 종목이 없습니다.")
        
    return double_bottom_stocks

# --- 테스트를 위한 가상 데이터 생성 함수 ---
def generate_sample_data(num_stocks=3, num_days=300):
    """
    쌍바닥 패턴을 포함하거나 포함하지 않는 샘플 주식 데이터를 생성하여 CSV 파일로 저장합니다.
    """
    all_data = []
    dates = pd.to_datetime(pd.date_range(start='2022-01-01', periods=num_days))

    # 1. 쌍바닥 패턴을 보이는 종목 (DB_STOCK)
    db_close_prices = []
    base_price = 10000
    # 하락 -> 바닥1
    for i in range(50): db_close_prices.append(base_price - i*20) 
    # 반등 -> 넥라인
    for i in range(30): db_close_prices.append(db_close_prices[-1] + i*30) 
    # 하락 -> 바닥2 (바닥1과 유사)
    for i in range(40): db_close_prices.append(db_close_prices[-1] - i*25) 
    # 돌파 및 상승
    for i in range(num_days - len(db_close_prices)): db_close_prices.append(db_close_prices[-1] + i*40)
    db_close_prices = db_close_prices[:num_days] # Ensure length matches num_days
    
    all_data.append(pd.DataFrame({
        'Date': dates,
        'Symbol': 'DB_STOCK',
        'Close': db_close_prices
    }))

    # 2. 또 다른 쌍바닥 패턴 종목 (DB_STOCK_2)
    db2_close_prices = []
    base_price_2 = 50000
    # 하락 -> 바닥1
    for i in range(60): db2_close_prices.append(base_price_2 - i*100) 
    # 반등 -> 넥라인
    for i in range(40): db2_close_prices.append(db2_close_prices[-1] + i*120) 
    # 하락 -> 바닥2 (바닥1과 유사)
    for i in range(50): db2_close_prices.append(db2_close_prices[-1] - i*90) 
    # 돌파 및 상승
    for i in range(num_days - len(db2_close_prices)): db2_close_prices.append(db2_close_prices[-1] + i*150)
    db2_close_prices = db2_close_prices[:num_days]
    
    all_data.append(pd.DataFrame({
        'Date': dates,
        'Symbol': 'DB_STOCK_2',
        'Close': db2_close_prices
    }))

    # 3. 쌍바닥 패턴이 없는 종목 (NORMAL_STOCK) - 단순 상승 추세
    normal_close_prices = [10000 + i*10 + np.random.randint(-50, 50) for i in range(num_days)]
    all_data.append(pd.DataFrame({
        'Date': dates,
        'Symbol': 'NORMAL_STOCK',
        'Close': normal_close_prices
    }))
    
    df_sample = pd.concat(all_data).reset_index(drop=True)
    sample_file_path = 'sample_kospi_data.csv'
    df_sample.to_csv(sample_file_path, index=False)
    print(f"\n테스트를 위한 샘플 CSV 파일 '{sample_file_path}'이 생성되었습니다.")
    return sample_file_path

# --- 실제 사용 예시 (스크립트 직접 실행 시) ---
if __name__ == '__main__':
    print("코스피 종가 CSV에서 쌍바닥 패턴 종목을 걸러내는 코드")
    print("="*50)

    # 1. 샘플 데이터 생성 (테스트용)
    # 실제 사용 시에는 이 부분을 주석 처리하고, 'file_path_to_use' 변수에 실제 CSV 파일 경로를 지정하세요.
    sample_csv_path = generate_sample_data()

    # 2. 쌍바닥 패턴 탐지 실행
    # CSV 파일 경로를 실제 파일 경로로 변경하여 사용하세요.
    file_path_to_use = sample_csv_path # 현재는 샘플 파일 사용
    # file_path_to_use = 'your_kospi_data.csv' # 실제 파일 경로 예시 (예: C:/Users/user/Documents/kospi_data.csv)

    print(f"\n--- '{file_path_to_use}' 파일에서 쌍바닥 패턴 탐지 ---")
    detected_stocks = detect_double_bottom_stocks(
        file_path_to_use,
        bottom_tolerance_pct=0.03,       # 두 바닥의 가격 차이 허용 범위 (3%)
        rebound_min_pct=0.05,            # 첫 번째 바닥 이후 최소 반등률 (5%)
        breakout_min_pct=0.03,           # 넥라인 돌파 최소 비율 (3%)
        min_days_between_bottoms=20,     # 두 바닥 사이 최소 기간 (거래일 기준)
        max_days_between_bottoms=120,    # 두 바닥 사이 최대 기간
        lookback_period=252,             # 패턴을 찾을 전체 기간 (1년)
        local_extremum_window=5          # 로컬 극점 탐색 윈도우
    )

    print("\n최종 탐지된 쌍바닥 종목:", detected_stocks)
    print("="*50)
    print("분석 완료!")
