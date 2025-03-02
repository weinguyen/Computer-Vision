import cv2
import mediapipe as mp
import os
import time
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont,ImageTk
from functools import lru_cache

idx = 0
cam = cv2.VideoCapture(0)
cam.set(3, 640) 
cam.set(4, 480)
cam.set(10, 150)
chedo = 0

def menu():
    global chedo
    def vetranh():
        global chedo
        chedo = 1
        window.destroy()
    def huongdan():
        global chedo
        chedo = 2
        window.destroy() 
    def thoat():
        chedo = 0
        window.destroy()

    window = tk.Tk()
    window.title("Magic Draw")
    window.geometry("900x600")
    window.resizable(False, False)
    window.configure(bg="#e0f7fa")

    title = tk.Label(
        window, text="🎨 Magic Draw 🎮", font=("Comic Sans MS", 30, "bold"), fg="#34495e", bg="#e0f7fa"
    )
    title.pack(pady=30)

    frame = tk.Frame(window, bg="#e0f7fa")
    frame.pack(pady=40)

    button_config = {
        "font": ("Comic Sans MS", 20, "bold"),
        "fg": "white",
        "width": 20,
        "height": 2,
        "relief": "raised",
        "borderwidth": 5
    }

    vebutton = tk.Button(
        frame, text="✨ Vẽ Tranh ✨", bg="#1abc9c",
        activebackground="#16a085", command=vetranh,
        **button_config
    )
    vebutton.grid(row=0, column=0, padx=30, pady=15)

    hdbutton = tk.Button(
        frame, text="🔥 Hướng dẫn 🔥", bg="#e67e22",
        activebackground="#d35400", command=huongdan,
        **button_config
    )
    hdbutton.grid(row=1, column=0, padx=30, pady=15)

    exit = tk.Button(
        frame, text="❌ Thoát ❌", bg="#e74c3c",
        activebackground="#c0392b", command=thoat,
        **button_config
    )
    exit.grid(row=2, column=0, padx=30, pady=15)

    window.mainloop()

    if chedo == 1:
        xtruoc, ytruoc = 0, 0
        mp_tay = mp.solutions.hands
        tay = mp_tay.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        check = 0
        ve = mp.solutions.drawing_utils
        d = 0

        # Cache image loading
        @lru_cache(maxsize=None)
        def load_image(path):
            return cv2.imread(path)

        file = 'colors'
        dsanh = os.listdir(file)
        hienmau = [cv2.resize(load_image(f'{file}/{img}'), (640, 100)) for img in dsanh]
        mau = (0, 0, 255)
        hopmau = hienmau[0]

        luu = cv2.resize(load_image("save.jpg"), (10,80))
        hoi = cv2.resize(load_image("luucanvas.png"), (220,60))
        co = cv2.resize(load_image("co.png"), (80,50))
        khong = cv2.resize(load_image("khong.png"), (80,50))

        canvas = np.zeros((480, 640, 3), np.uint8)
        luutru = "luutru"

        # Optimize Kalman filter initialization
        loc = cv2.KalmanFilter(4, 2)
        loc.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        loc.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.03, 0], [0, 0, 0, 0.03]], np.float32) * 0.01
        loc.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0.9, 0], [0, 0, 0, 0.9]], np.float32)

        mphands = mp.solutions.hands
        mpdraw = mp.solutions.drawing_utils
        k = 0

        cv2.namedWindow('MagicDraw')

        while True:
            _, khung = cam.read()
            if khung is None:
                continue

            khung = cv2.flip(khung, 1)
            rgbkhung = cv2.cvtColor(khung, cv2.COLOR_BGR2RGB)
            kq = tay.process(rgbkhung)
            dstay = []

            if kq.multi_hand_landmarks:
                for diemtay in kq.multi_hand_landmarks:
                    drawing_spec = mpdraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1)
                    connection_spec = mpdraw.DrawingSpec(color=(0, 255, 0) if k == 1 else (0, 0, 255), thickness=2)
                    
                    mpdraw.draw_landmarks(
                        khung,
                        diemtay,
                        mphands.HAND_CONNECTIONS,
                        drawing_spec,
                        connection_spec
                    )

                    h, w, _ = khung.shape
                    for i, diem in enumerate(diemtay.landmark):
                        cx, cy = int(diem.x * w), int(diem.y * h)
                        dstay.append([i, cx, cy])

            if dstay:
                x1, y1 = dstay[8][1], dstay[8][2]
                x2, y2 = dstay[12][1], dstay[12][2]

                do = np.array([[np.float32(x1)], [np.float32(y1)]])
                loc.correct(do)
                dudoan = loc.predict()
                lx, ly = int(dudoan[0]), int(dudoan[1])

                if dstay[8][2] < dstay[5][2] and dstay[12][2] < dstay[10][2]:
                    if dstay[16][2] < dstay[15][2] and dstay[20][2] < dstay[19][2]:
                        k = 1
                    xtruoc, ytruoc = 0, 0

                    if y1 < 100:
                        x_ranges = [(11,72), (72,153), (153,224), (224,295), (295,366), 
                                  (366,437), (437,508)]
                        colors = [(0,0,0), (255,255,255), (135,206,235), (209,87,56),
                                (0,255,0), (0,0,255), (245,66,218)]
                        
                        for i, (x_min, x_max) in enumerate(x_ranges):
                            if x_min < x1 < x_max:
                                hopmau = hienmau[i+1]
                                mau = colors[i]
                                break

                        if dstay[8][1] > 540 and dstay[12][1] > 540:
                            hopmau = hienmau[8]
                            canvas = np.zeros((480, 640, 3), np.uint8)

                    elif 200 <= y1 <= 280 and x1 >= 600:
                        check = 1

                    if check == 1:
                        if 300 <= y1 <= 350:
                            if 170 <= x1 <= 250:
                                dem = 0
                                while True:
                                    if not os.path.exists(os.path.join("luutru", f"{dem}.jpg")):
                                        cv2.imwrite(f"{luutru}/{dem}.jpg", canvas)
                                        break
                                    dem += 1
                                time.sleep(1)
                                canvas = np.zeros((480, 640, 3), np.uint8)
                                check = 0
                            if 390 <= x1 <= 470:
                                check = 0

                    cv2.rectangle(khung, (x1, y1), (x2, y2), mau, cv2.FILLED)

                elif dstay[8][2] < dstay[3][2] and k == 1:
                    if d == 25:
                        if xtruoc == 0 and ytruoc == 0:
                            xtruoc, ytruoc = lx, ly
                        doday = 50 if mau == (0, 0, 0) else 5
                        cv2.line(khung, (xtruoc, ytruoc), (lx, ly), mau, doday, cv2.LINE_AA)
                        cv2.line(canvas, (xtruoc, ytruoc), (lx, ly), mau, doday, cv2.LINE_AA)
                    xtruoc, ytruoc = lx, ly
                    if d < 25:
                        d += 1

                if (dstay[8][2] > dstay[5][2] and dstay[12][2] > dstay[9][2] and 
                    dstay[16][2] > dstay[13][2] and dstay[20][2] > dstay[17][2]):
                    k = 0

            imgxam = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, imgnguoc = cv2.threshold(imgxam, 50, 255, cv2.THRESH_BINARY_INV)
            imgnguoc = cv2.cvtColor(imgnguoc, cv2.COLOR_GRAY2BGR)

            h, w, c = khung.shape
            canvas = cv2.resize(canvas, (w, h))
            imgnguoc = cv2.resize(imgnguoc, (w, h))

            khung = cv2.bitwise_and(khung.astype(np.uint8), imgnguoc.astype(np.uint8))
            khung = cv2.bitwise_or(khung, canvas.astype(np.uint8))
            khung[0:100, 0:640] = hopmau
            khung[200:280, 630:640] = luu

            cv2.putText(khung, "Drawing" if k == 1 else "Stop drawing",
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 255, 0) if k == 1 else (0, 0, 255), 2, cv2.LINE_AA)

            if check == 1:
                khung[210:270, 210:430] = hoi
                khung[300:350, 170:250] = co
                khung[300:350, 390:470] = khong

            hienthi = np.hstack((khung, canvas))
            cv2.imshow('MagicDraw', hienthi)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('MagicDraw', cv2.WND_PROP_VISIBLE) < 1:
                break

        cam.release()
        cv2.destroyAllWindows()

    elif chedo == 2:
        wd = tk.Tk()
        wd.geometry("1280x700")
        wd.title("Chuyển đổi hình ảnh")
        imgs = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg']

        @lru_cache(maxsize=None)
        def load_and_resize_image(img_path, width, height):
            img = Image.open(img_path)
            return img.resize((width, height))

        def show():
            img = load_and_resize_image(imgs[idx], wd.winfo_width(), wd.winfo_height())
            img_tk = ImageTk.PhotoImage(img)
            lb.config(image=img_tk)
            lb.image = img_tk
            
            if idx == 5:
                if not hasattr(wd, 'menu'):
                    menu = tk.Button(wd, text="THOÁT", command=thoat,
                                   font=("Arial", 40), bg="blue", fg="white",
                                   width=6, height=2)
                    menu.place(relx=0.5, rely=0.9, anchor="center")
                    wd.menu = menu
            else:
                if hasattr(wd, 'menu'):
                    wd.menu.destroy()
                    del wd.menu

        def thoat():
            global idx
            idx = 0
            wd.destroy()
            menu()

        def next_img():
            global idx
            idx = (idx + 1) % len(imgs)
            show()

        def prev_img():
            global idx
            idx = (idx - 1) % len(imgs)
            show()

        lb = tk.Label(wd)
        lb.place(relx=0.5, rely=0.5, anchor="center")

        button_style = {
            "font": ("Arial", 40, "bold"),
            "fg": "white",
            "width": 3,
            "height": 1,
            "relief": "raised",
            "bd": 5
        }

        prev = tk.Button(wd, text="⏪", command=prev_img, bg="#ff6347",
                        activebackground="#ff4500", activeforeground="yellow",
                        **button_style)
        prev.place(relx=0.05, rely=0.9, anchor="center")

        next = tk.Button(wd, text="⏩", command=next_img, bg="#32cd32",
                        activebackground="#228b22", activeforeground="yellow",
                        **button_style)
        next.place(relx=0.95, rely=0.9, anchor="center")

        wd.bind("<Configure>", lambda event: show())
        wd.mainloop()

menu()
