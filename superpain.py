import cv2
import mediapipe as mp
import os
import time
import numpy as np
# Khởi tạo camera và các thông số
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
cam.set(10, 150)
x_truoc, y_truoc = 0, 0
mp_tay = mp.solutions.hands
tay = tay = mp_tay.Hands(
    static_image_mode=False,  # Sử dụng chế độ theo dõi động
    max_num_hands=1,          # Đặt số lượng bàn tay để theo dõi
    min_detection_confidence=0.3,  # Tăng độ tin cậy khi phát hiện tay
    min_tracking_confidence=0.1    # Tăng độ tin cậy khi theo dõi
)
check = 0
ve = mp.solutions.drawing_utils
d=0
file = 'colors'
ds_anh = os.listdir(file)
hien_mau = [cv2.resize(cv2.imread(f'{file}/{img}'), (640, 100)) for img in ds_anh]
mau = (0, 0, 255) 
hopmau = hien_mau[0]
luu1= cv2.imread("save.jpg")
luu = cv2.resize(luu1,(10,80))
hoi1 = cv2.imread("luucanvas.png")
hoi = cv2.resize(hoi1,(220,60))
co1 = cv2.imread("co.png")
co = cv2.resize(co1,(80,50))
khong1 = cv2.imread("khong.png")
khong = cv2.resize(khong1,(80,50))
bang_ve = np.zeros((480, 640, 3), np.uint8)
luutru = "luutru"
# Kalman Filter để làm mượt đường vẽ
loc_kf = cv2.KalmanFilter(4, 2)
loc_kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
loc_kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.03, 0], [0, 0, 0, 0.03]], np.float32) * 0.01
loc_kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0.9, 0], [0, 0, 0, 0.9]], np.float32)
while True:
    # Đọc từng khung hình từ camera
    _, khung = cam.read()
    khung = cv2.flip(khung, 1)
    rgb_khung = cv2.cvtColor(khung, cv2.COLOR_BGR2RGB)
    ket_qua = tay.process(rgb_khung)
    ds_tay = []
    # Nhận diện bàn tay và lưu tọa độ các điểm
    if ket_qua.multi_hand_landmarks:
        for diem_tay in ket_qua.multi_hand_landmarks:
            for idx, diem in enumerate(diem_tay.landmark):
                h, w, _ = khung.shape
                cx, cy = int(diem.x * w), int(diem.y * h)
                ds_tay.append([idx, cx, cy])
    # Vẽ theo đầu ngón tay
    if ds_tay:  
        x1, y1 = ds_tay[8][1], ds_tay[8][2] #lưu vị trí các đốt tay dưới dạng tọa độ
        x2, y2 = ds_tay[12][1], ds_tay[12][2]
        do_luong = np.array([[np.float32(x1)], [np.float32(y1)]])
        loc_kf.correct(do_luong)
        du_doan = loc_kf.predict()
        lx, ly = int(du_doan[0]), int(du_doan[1])
        if ds_tay[8][2] < ds_tay[5][2] and ds_tay[12][2] < ds_tay[9][2]:
            x_truoc, y_truoc = 0, 0
            if y1<100:
                if 11<x1<72:
                    hopmau = hien_mau[1]
                    mau = (0, 0, 0)
                elif 72<x1<153:
                    hopmau = hien_mau[2]
                    mau = (255, 255, 255)
                elif 153<x1<224:
                    hopmau = hien_mau[3]
                    mau = (135, 206, 235)
                elif 224<x1<295:
                    hopmau = hien_mau[4]
                    mau = (209, 87, 56)   
                elif 295<x1< 366:
                    hopmau = hien_mau[5]
                    mau = (0, 255, 0)
                elif 366<x1<437:
                    hopmau = hien_mau[6]
                    mau = (0, 0, 255)
                elif 437<x1<508:
                    hopmau = hien_mau[7]
                    mau = (245, 66, 218)
                elif ds_tay[8][1] > 540 and ds_tay[12][1] > 540:
                    hopmau = hien_mau[8]
                    bang_ve = np.zeros((480, 640, 3), np.uint8)  # Xóa canvas
            elif(200<=y1<=280 and x1>=630):
                check = 1
            if(check == 1):
                if(300<=y1<=350):
                    if(170<=x1<=250):
                        dem = 0
                        while(1):
                            file_name = f"{dem}.jpg"
                            file_path = os.path.join("luutru", file_name)
                            if not os.path.exists(file_path):
                                cv2.imwrite(f"{luutru}/{dem}.jpg", bang_ve) 
                                break
                            dem+=1
                        time.sleep(1)
                        bang_ve = np.zeros((480, 640, 3), np.uint8)  # Xóa canvas
                        check = 0
                    if(390<=x1<=470):
                        check = 0
            cv2.rectangle(khung, (x1, y1), (x2, y2), mau, cv2.FILLED)
        elif  ds_tay[8][2] < ds_tay[3][2]:
                if(d==25):
                    if x_truoc == 0 and y_truoc == 0:
                        x_truoc, y_truoc = lx, ly
                    do_day = 50 if mau == (0, 0, 0) else 5
                    cv2.line(khung, (x_truoc, y_truoc), (lx, ly), mau, do_day, cv2.LINE_AA)
                    cv2.line(bang_ve, (x_truoc, y_truoc), (lx, ly), mau, do_day, cv2.LINE_AA)
                x_truoc, y_truoc = lx, ly
                if(d<25):
                    d+=1
    #ghép canvas vào khung
    img_xam = cv2.cvtColor(bang_ve, cv2.COLOR_BGR2GRAY)
    _, img_nguoc = cv2.threshold(img_xam, 50, 255, cv2.THRESH_BINARY_INV)
    img_nguoc = cv2.cvtColor(img_nguoc, cv2.COLOR_GRAY2BGR)
    # Đảm bảo các hình ảnh có cùng kích thước và kiểu dữ liệu
    h, w, c = khung.shape
    bang_ve = cv2.resize(bang_ve, (w, h))
    img_nguoc = cv2.resize(img_nguoc, (w, h))
    # Đảm bảo tất cả ảnh đều là kiểu uint8
    khung = khung.astype(np.uint8)
    bang_ve = bang_ve.astype(np.uint8)
    img_nguoc = img_nguoc.astype(np.uint8)
    # Thực hiện thao tác bitwise sau khi đã đồng bộ kích thước và kiểu dữ liệu
    khung = cv2.bitwise_and(khung, img_nguoc)
    khung = cv2.bitwise_or(khung, bang_ve)
    #hiển thị hộp màu 
    khung[0:100, 0:640] = hopmau
    khung[200:280, 630:640] = luu
    if(check == 1):
        khung[210:270,210:430] = hoi
        khung[300:350,170:250] = co
        khung[300:350,390:470] = khong
    # Tạo ảnh kết hợp giữa webcam và canvas
    hien_thi = np.hstack((khung, bang_ve))
    # Hiển thị cửa sổ Paint App
    cv2.imshow('SuperPaint', hien_thi)
    # Thoát nếu nhấn phím 'q' hoặc đóng cửa sổ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Thóa nếu nhấn 'X' góc ngoài cùng
    if cv2.getWindowProperty('SuperPaint', cv2.WND_PROP_VISIBLE) < 1:
        break
cam.release()
cv2.destroyAllWindows()