from pykrx import stock
from datetime import datetime, timedelta

# 오늘 날짜 및 6개월 전 날짜
today = datetime.today()
six_months_ago = today - timedelta(weeks=26)

# 종목 코드 예시 (삼성전자: 005930)
ticker = '005930'

# OHLCV 데이터 가져오기
df = stock.get_market_ohlcv_by_date(
    fromdate=six_months_ago.strftime('%Y%m%d'),
    todate=today.strftime('%Y%m%d'),
    ticker=ticker
)

# 결과 확인
df.to_csv(f"{ticker}_OHLCV_6months.csv", encoding='utf-8-sig')
