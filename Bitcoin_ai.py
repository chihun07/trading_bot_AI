import os
import pyupbit
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# -----------------------
# 1) 이메일, 업비트 API, 모델 경로 등 설정
# -----------------------
SENDER_EMAIL = "chihunslee63@gmail.com"
RECEIVER_EMAIL = "chihun9344@naver.com"
PASSWORD = "ghuy esep npfu byuy"  # 발신자 이메일 비밀번호

access_key = "your-access-key"
secret_key = "your-secret-key"
upbit = pyupbit.Upbit(access_key, secret_key)

model_path = "dogecoin_model.pth"
scaler_path = "dogecoin_scaler.pkl"  # MinMaxScaler를 저장할 경로(옵션)

# -----------------------
# 2) LSTM 모델 정의
# -----------------------
class BitcoinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BitcoinLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 초기 hidden state, cell state 0으로 초기화
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])  # 마지막 시점의 결과만 사용
        return out

# -----------------------
# 3) 초기 설정
# -----------------------
input_size = 5   # open, high, low, close, volume
hidden_size = 64
num_layers = 2
output_size = 1

seq_length = 200  # LSTM에 넣을 시퀀스 길이(예: 200개 봉)
batch_size = 32
epochs = 200
learning_rate = 0.0001

criterion = nn.MSELoss()
model = BitcoinLSTM(input_size, hidden_size, num_layers, output_size)

# -----------------------
# 4) 디바이스 설정(GPU or CPU)
# -----------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------
# 5) MinMaxScaler 준비
#    - 모델 최초 학습 시 'fit'하고, 
#      이후 실시간 예측 때는 'transform'만 사용하도록 변경.
# -----------------------
scaler = MinMaxScaler()

# -----------------------
# 6) (옵션) 모델이 이미 있으면 불러오고, 없으면 학습
# -----------------------
if os.path.exists(model_path):
    print("저장된 모델을 불러옵니다. 과거 학습은 건너뜁니다.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # 만약 scaler도 별도로 저장했다면 여기서 로드
    # 예: with open(scaler_path, "rb") as f: scaler = pickle.load(f)
else:
    print("모델이 없으므로 과거 데이터를 불러와 학습을 시작합니다.")

    # 1) 과거 데이터 가져오기
    df = pyupbit.get_ohlcv("KRW-DOGE", interval="minute15", count=4000)
    if df is None or len(df) < seq_length:
        raise ValueError("데이터를 충분히 가져오지 못했습니다. API 호출을 확인하세요.")

    # 2) 스케일러 fit
    #    open, high, low, close, volume 5개의 컬럼만 사용
    #    여기서는 학습 시점 전체 데이터로 fit
    raw_data = df[['open', 'high', 'low', 'close', 'volume']].values
    scaled_data = scaler.fit_transform(raw_data)

    # 3) 학습용 시퀀스 만들기
    x_data, y_data = [], []
    for i in range(seq_length, len(scaled_data)):
        x_data.append(scaled_data[i - seq_length:i])  # seq_length 길이
        y_data.append(scaled_data[i, 3])  # 종가 위치(인덱스 3)를 예측

    x_data = torch.tensor(x_data, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)

    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4) 학습 함수
    def train_model(model, train_loader, criterion, optimizer, epochs=50):
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x).squeeze()  # (batch,)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.5f}")

    # 5) 모델 학습
    train_model(model, train_loader, criterion, optimizer, epochs=epochs)

    # 6) 모델 저장
    torch.save(model.state_dict(), model_path)
    print("모델 학습 완료 후 저장되었습니다.")

    # (옵션) 스케일러도 저장하고 싶다면
    # import pickle
    # with open(scaler_path, "wb") as f:
    #    pickle.dump(scaler, f)

# -----------------------
# 7) 이메일 전송 함수
# -----------------------
def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("이메일 발송 성공:", subject)
    except Exception as e:
        print(f"이메일 발송 실패: {e}")

# -----------------------
# 8) 잔고/거래 관련 함수
# -----------------------
def check_balance(ticker):
    """ticker 예: 'KRW' or 'DOGE' 등"""
    return upbit.get_balance(ticker)

def get_current_price(ticker):
    """현재가 조회. 예: 'KRW-DOGE'"""
    return pyupbit.get_current_price(ticker)

def buy_by_ratio(ticker, ratio):
    """
    원화 잔고 대비 ratio 비율로 매수 주문(시장가).
    예: ratio=1 => 전액 매수.
    """
    krw_balance = check_balance("KRW")
    amount_to_spend = krw_balance * ratio
    if amount_to_spend >= 5000:  # 최소 주문금액(업비트 기준)
        order = upbit.buy_market_order(ticker, amount_to_spend)
        print(f"[매수 주문] {order}")
        return get_current_price(ticker)  # 체결 가격을 정확히 알 수 없으나, 임시로 현재가 반환
    else:
        print("매수 금액 부족.")
        return None

def sell_by_ratio(ticker, ratio):
    """
    코인 잔고 대비 ratio 비율로 매도 주문(시장가).
    예: ratio=1 => 전량 매도.
    """
    coin_symbol = ticker.split('-')[1]  # "KRW-DOGE" -> "DOGE"
    coin_balance = check_balance(coin_symbol)
    amount_to_sell = coin_balance * ratio
    if amount_to_sell > 0.0001:  # 최소 매도 수량
        order = upbit.sell_market_order(ticker, amount_to_sell)
        print(f"[매도 주문] {order}")
        return True
    else:
        print("매도 수량 부족.")
        return False

def get_balances():
    """(KRW 잔고, DOGE 잔고) 반환"""
    return check_balance("KRW"), check_balance("DOGE")

# -----------------------
# 9) 자동매매 로직
# -----------------------
ticker = "KRW-DOGE"
purchase_price = None  # 마지막 매수 단가(단순화)
total_profit_loss = 0.0
err_count = 0
buy_count = 0
max_loss_ratio = 0.4

initial_balance = check_balance("KRW")
print(f"시작 시점 원화 잔고: {initial_balance}원")

try:
    while True:
        try:
            # 1) 최신 200개 봉 데이터를 가져와서 transform만 적용
            df_recent = pyupbit.get_ohlcv(ticker, interval="minute15", count=seq_length)
            if df_recent is None or len(df_recent) < seq_length:
                print("최근 데이터 부족으로 예측 불가.")
                time.sleep(60)
                continue

            recent_raw_data = df_recent[['open', 'high', 'low', 'close', 'volume']].values
            # 이전에 fit된 scaler를 사용해 transform만 수행
            recent_scaled = scaler.transform(recent_raw_data)

            # 2) 텐서 변환
            recent_scaled_tensor = torch.tensor(recent_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 3) 예측
            model.eval()
            with torch.no_grad():
                predicted = model(recent_scaled_tensor).item()
            
            current_price = get_current_price(ticker)
            print(f"\nPredicted Price: {predicted:.4f}, Current Price: {current_price}")

            # 4) 손실 제한 체크
            #    total_profit_loss는 누적 이익(+) 손실(-)로 가정.
            #    initial_balance * max_loss_ratio => 최대 허용 손실 금액
            if total_profit_loss <= -initial_balance * max_loss_ratio:
                krw_balance, doge_balance = get_balances()
                body = (
                    f"[자동매매 종료] 최대 손실 초과.\n"
                    f"보유 원화: {krw_balance}원\n"
                    f"보유 DOGE: {doge_balance}\n"
                    f"총 손익: {total_profit_loss}"
                )
                send_email("자동 손실 초과 알림", body)
                break

            # ----------------------------------------------------------------------
            # 5) 매수/매도 조건
            #    일반적으로 => 미래 가격이 현재가보다 '상승할' 것으로 예측 -> 매수
            #                미래 가격이 현재가보다 '하락할' 것으로 예측 -> 매도
            #    여기서는 2% 차이를 임계값으로 사용.
            # ----------------------------------------------------------------------
            if predicted > current_price * 1.02:
                # => 2% 이상 오를 것으로 예측: '매수' 시그널
                if purchase_price is None:
                    # 아직 코인을 보유하지 않았다면 매수
                    buy_price = buy_by_ratio(ticker, 1)  # 전액 매수
                    if buy_price is not None:
                        purchase_price = buy_price
                        buy_count += 1
                        print(f"매수 체결가(가정): {purchase_price}")
            elif predicted < current_price * 0.98:
                # => 2% 이상 떨어질 것으로 예측: '매도' 시그널
                if purchase_price is not None:
                    # 이미 코인을 가지고 있다면 매도
                    sold = sell_by_ratio(ticker, 1)
                    if sold:
                        # 매도 후 이익 계산 (단순화: 전량 매도했다고 가정)
                        # purchase_price != None인 상태에서만 매도하므로 diff 계산 가능
                        profit = (current_price - purchase_price) * check_balance("DOGE")
                        # 위처럼 매도 즉시 check_balance("DOGE")를 부르면 체결 지연 문제가 있을 수 있지만, 예시로 간단히 처리
                        total_profit_loss += profit
                        print(f"매도 이익: {profit}, 총 손익: {total_profit_loss}")
                        purchase_price = None  # 보유 포지션 해제

            # 6) 100회 매수 시 이메일 알림 (단순 이벤트)
            if buy_count >= 100:
                buy_count = 0
                krw_balance, doge_balance = get_balances()
                body = (
                    f"100회 매수 완료.\n"
                    f"보유 원화: {krw_balance}원\n"
                    f"보유 DOGE: {doge_balance}\n"
                    f"총 손익: {total_profit_loss}"
                )
                send_email("학습 매수 알림", body)

            # 7) 일정 시간 대기
            time.sleep(360)  # 6분

        except Exception as e:
            print(f"[에러 발생] {e}")
            err_count += 1
            if err_count >= 10:
                krw_balance, doge_balance = get_balances()
                body = (
                    f"연속 에러 발생.\n"
                    f"보유 원화: {krw_balance}원\n"
                    f"보유 DOGE: {doge_balance}\n"
                    f"총 손익: {total_profit_loss}\n"
                    f"에러: {e}"
                )
                send_email("학습 에러 알림", body)
                break

except Exception as e:
    print(f"[메인 루프 종료] 에러 발생: {e}")
