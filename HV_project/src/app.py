from flask import Flask, request, jsonify, render_template, render_template_string
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings


#경고 메세지 생략
warnings.filterwarnings('ignore')

#Flask 애플리케이션 생성
app = Flask(__name__)

#XGBClassifier 모델 로드
model = XGBClassifier()
model.load_model("xgb_model.json")

# 셀 번호 역변환 리스트
le_inverse = ['YSDG10-1', 'YSDGHFC0053', 'YSDGHFC0054', 'YSDGHFC0069',
'YSDGRFOG0065', 'YSHS0011', 'YSHS0013', 'YSHS0014', 'YSHS0015',
'YSHS0024', 'YSHS0035', 'YSHS0036', 'YSHS0037', 'YSHS0038',
'YSHS0039', 'YSHS0042', 'YSHS0043', 'YSHS0045', 'YSHS0046',
'YSHS0047', 'YSHS0051', 'YSHS0053', 'YSHS0054', 'YSHS0055',
'YSHS0058', 'YSHS0065', 'YSHS0069-1', 'YSHS0069-2', 'YSHS0070',
'YSHS0075', 'YSHS0079', 'YSHS0083', 'YSHS0092', 'YSHSH5005',
'YSHSHFC0007', 'YSHSHFC0007-2', 'YSHSHFC0008', 'YSHSHFC0009',
'YSJB0075', 'YSJB1-9', 'YSJB5-25', 'YSJB5-27', 'YSJBF3-2',
'YSJBG6-1F', 'YSJBH910', 'YSJSC2003', 'YSJSC2003-1', 'YSJSC2004',
'YSJSC3004', 'YSJSC3010', 'YSJSC4007', 'YSJSC5001', 'YSMMH4004',
'YSMMH5006', 'YSMMH6003', 'YSMYRFOG0038', 'YSSB1-11', 'YSSB1-6',
'YSSB2-15', 'YSSB2-4', 'YSSB2-7', 'YSWS0049', 'YSWS0051',
'YSWS0087', 'YSWS0154', 'YSWS0156', 'YSWS0189', 'YSWS0217',
'YSWS0217-2', 'YSWS0225', 'YSWS0225-1', 'YSWS0244', 'YSWS0265',
'YSWS0267', 'YSWS0278', 'YSWS0278A', 'YSWS0287', 'YSWS0296',
'YSWS0297', 'YSWS1-1', 'YSWS10-8', 'YSWS2-13', 'YSWS2-5',
'YSWS2-7', 'YSWS4-2', 'YSWS4-5', 'YSWS5-27', 'YSWS5-30', 'YSWS6-2',
'YSWS8-5', 'YSWSF22-17', 'YSWSG3-4B', 'YSYW0009', 'YSYW0025',
'YSYW0030', 'YSYW0047', 'YSYW0063B', 'YSYW0069', 'YSYW0107']

#경고 메세지 저장할 빈 리스트 생성
alerts = []

#이메일 전송 함수 생성 - 전송 이메일 정보 
def send_email_alert(to_email, subject, message):
    from_email = "predicthunter1@naver.com"
    from_password = "vmgjs12!"
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    
    msg.attach(MIMEText(message, 'plain'))
    
    #네이버 메일 전송 환경 및 포트 설정
    try:
        server = smtplib.SMTP_SSL('smtp.naver.com', 465)
        server.ehlo()  # Check connection
        server.login(from_email, from_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
    except smtplib.SMTPAuthenticationError:
        print("Error: Unable to send email. Check the login credentials and try again.")
    except Exception as e:
        print(f"Error: {e}")

#웹 페이지를 렌더링 해주는 route
@app.route('/')
def home():
    return render_template('index.html')

#예측 수행하는 route, 웹페이지에서 입력된 값 get할 수 있게 설정
@app.route('/predict', methods=['POST'])
def predict():
    global alerts
    alerts = [] 
    date = request.form['date']
    threshold = float(request.form.get('threshold', 0.033))
    speed = int(request.form.get('speed', 1))
    data_with_lags = pd.read_csv("data_with_lags2.csv", encoding='cp949')
    data_with_lags['측정시간'] = pd.to_datetime(data_with_lags['측정시간'])   # lags data의 측정시간을 datetime 형식으로 변환
    
    date_start = pd.to_datetime(date) #입력된 날짜를 datetime 형식으로 변환
    date_end = date_start + timedelta(days=1)   # 입력된 날짜의 다음날을 계산
    filtered_data = data_with_lags[(data_with_lags['측정시간'] >= date_start) & (data_with_lags['측정시간'] < date_end)]   # 입력된 날짜의 데이터 필터링
    
    sim_time = date_start   # 시물레이션 시작 시간 설정
    
    while sim_time < date_end:
        current_time_str = sim_time.strftime('%Y-%m-%d %H:%M:%S')   # 현재 시뮬레이션 시간 문자열 변환
        print(current_time_str)

        #현재 시뮬레이션 시간의 데이터 필터링
        current_data = filtered_data[(filtered_data['측정시간'].dt.hour == sim_time.hour) &
                                     (filtered_data['측정시간'].dt.minute == sim_time.minute) &
                                     (filtered_data['측정시간'].dt.second == sim_time.second)]
    
        for _, row in current_data.iterrows():
            X = row.drop(['장애여부', '측정시간']).values.reshape(1, -1) # 특징 데이터 준비
            y_proba = float(model.predict_proba(X)[:, 1][0])  # 예측 확률 계산
            cell_name = le_inverse[int(row['셀번호'])]  # 셀번호 le_invers로 변환

            #예측 확률이 임계값을 초과하면 경고 메세지 생성
            if y_proba >= threshold:   
                alert_message = f"장애 발생 경고!\n장애 확률: {y_proba:.2f}, 셀번호: {cell_name}, 시간: {current_time_str}\n"
                alert_message += f"현재 상태 상향파워2 : {row['상향파워2']}, 상향SNR : {row['상향SNR']}, 하향파워 : {row['하향파워']}, 하향SNR : {row['하향SNR']}"
                send_email_alert('predicthunter1@naver.com', '장애 발생 경고', alert_message)
                alerts.append({
                    'time': current_time_str,
                    'probability': y_proba,
                    'cell_number': cell_name,
                    'message': alert_message
                })
                print(alert_message)
    
        sim_time += timedelta(minutes=1)   # 시뮬레이션 시간 1분 증가
        time.sleep(1 / speed)   #  시뮬레이션 속도에 맞춰 대기 진행
    
    return jsonify({'status': 'completed'})   # 예측 완료시 상태 반환

#경고 메세지 반환 route
@app.route('/alerts', methods=['GET'])
def get_alerts():
    if alerts:
        return jsonify({'alert': True, 'time': alerts[-1]['time'], 'probability': alerts[-1]['probability'], 'cell_number': alerts[-1]['cell_number'], 'message': alerts[-1]['message']})
    else:
        return jsonify({'alert': False})

#대시보드 렌더링 route
@app.route('/dashboard', methods=['GET'])
def dashboard():
    cell_id = request.args.get('cell_id')
    specific_date = request.args.get('date')  # HTML에서 선택된 날짜 가져오기

    data = pd.read_csv("avail_data2.csv", parse_dates=['측정시간'], encoding='cp949').sort_values('측정시간').dropna().reset_index(drop=True)
    df_cell = data[data['셀번호'] == cell_id]
    df_cell = df_cell[pd.to_datetime(df_cell['측정시간']).dt.date == pd.to_datetime(specific_date).date()]
    df_cell = df_cell.sort_values(by='측정시간')

    plt.figure(figsize=(15, 12))  # 크기 조정
    plt.rc('font', family='NanumGothic')

    plt.subplot(2, 2, 1)
    sns.lineplot(x='측정시간', y='상향파워2', data=df_cell, color='blue')
    plt.title('상향 파워2 시계열')
    plt.xlabel('시간')
    plt.ylabel('상향 파워2')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    sns.lineplot(x='측정시간', y='상향SNR', data=df_cell, color='blue')
    plt.title('상향 SNR 시계열')
    plt.xlabel('시간')
    plt.ylabel('상향 SNR')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    sns.lineplot(x='측정시간', y='하향파워', data=df_cell, color='blue')
    plt.title('하향 파워 시계열')
    plt.xlabel('시간')
    plt.ylabel('하향 파워')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    sns.lineplot(x='측정시간', y='하향SNR', data=df_cell, color='blue')
    plt.title('하향 SNR 시계열')
    plt.xlabel('시간')
    plt.ylabel('하향 SNR')
    plt.xticks(rotation=45)

    plt.tight_layout()   #레이아웃 조정
 
    img = io.BytesIO()    #이미지 저장을 위한 버퍼 생성
    plt.savefig(img, format='png')   #이미지를 버퍼에 저장
    img.seek(0)    #버퍼에 시작으로 이동
    plot_url = base64.b64encode(img.getvalue()).decode()   #이미지를 base64로 인코딩

    return render_template_string('<img src="data:image/png;base64,{{ plot_url }}">', plot_url=plot_url)   # 인코딩된 이미지 html로 반환

if __name__ == '__main__':
    app.run(debug=True)
