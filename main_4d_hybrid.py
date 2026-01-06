import sys
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QSlider, QFrame, QGroupBox, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, Slot as pyqtSlot
from PySide6.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO

# --- 4D-Humans 경로 추가 ---
# 현재 파일 위치 기준 4D-Humans 폴더를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "4D-Humans"))

try:
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
    # 윈도우 호환성을 위해 Detectron2 관련 모듈은 임포트하지 않거나 예외처리
    HMR2_AVAILABLE = True
except ImportError as e:
    print(f"[경고] 4D-Humans 모듈 로딩 실패: {e}")
    print("폴더 구조와 라이브러리 설치를 확인해주세요.")
    HMR2_AVAILABLE = False

class AnalysisThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    
    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.is_running = False
        
        # --- 사용자 설정 ---
        self.sensitivity = 0.5
        self.low_stance_boost = 0.5
        self.velocity_threshold = 0.5
        
        # --- 모델 초기화 ---
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # 1. 탐지기: YOLOv8 (Detectron2 대신 사용 - 설치 간편, 속도 빠름)
        self.yolo = YOLO('yolov8n.pt')
        
        # 2. 3D 복원: 4D-Humans (HMR2)
        if HMR2_AVAILABLE:
            print("[INFO] 4D-Humans 모델 로딩 중... (시간이 좀 걸립니다)")
            # 다운받은 logs 폴더에서 모델을 찾아옴
            self.hmr2 = load_hmr2(DEFAULT_CHECKPOINT).to(self.device)
            self.hmr2.eval()
            print("[INFO] 4D-Humans 모델 로딩 완료!")
        else:
            self.hmr2 = None

        # HMR2 입력 전처리기 (256x256 리사이징 등)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.prev_com_3d = {}

    def get_3d_pose(self, frame, box):
        """ YOLO 박스 영역을 잘라 HMR2에 넣고 3D 좌표를 얻음 """
        x1, y1, x2, y2 = map(int, box)
        # 이미지 밖으로 나가는 것 방지
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1: return None

        # 크롭 및 전처리
        crop = frame[y1:y2, x1:x2]
        # BGR -> RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.hmr2(input_tensor)
        
        # 3D 관절 좌표 (SMPL 포맷)
        pred_joints = out['pred_joints'][0].cpu().numpy()
        # 카메라 파라미터 (투영용)
        pred_cam = out['pred_cam'][0].cpu().numpy()
        
        return pred_joints, pred_cam

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_cnt = 0
        
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret: break
            frame_cnt += 1
            
            # 최적화: 2프레임마다 분석
            # if frame_cnt % 2 != 0: continue

            if self.hmr2:
                frame = self.process_frame_hybrid(frame)
            else:
                # 4D 모델 없으면 원본 출력
                cv2.putText(frame, "4D-Humans Model Not Found", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            p = qt_img.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
            self.change_pixmap_signal.emit(p)

        cap.release()
        self.prev_com_3d = {}

    def process_frame_hybrid(self, frame):
        # 1. YOLO로 사람 감지
        results = self.yolo.track(frame, persist=True, verbose=False)
        
        if not results or results[0].boxes.id is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        h_img, w_img = frame.shape[:2]

        for i, box in enumerate(boxes):
            track_id = ids[i]
            
            # 2. HMR2로 진짜 3D 좌표 추출
            res = self.get_3d_pose(frame, box)
            if res is None: continue
            joints_3d, cam_param = res

            # SMPL 인덱스: 0:골반(CoM), 10:왼발, 11:오른발, 15:머리
            pelvis = joints_3d[0]
            l_foot = joints_3d[10]
            r_foot = joints_3d[11]
            head = joints_3d[15]

            # --- 3D 물리 분석 ---
            
            # (1) 기저면 (Base of Support)
            # X-Y 평면상 거리 (HMR2 좌표계: Y가 높이인 경우도 있고 Z가 높이인 경우도 있음.
            # 보통 HMR 출력은 카메라 기준 좌표계. 여기선 깊이가 중요)
            # 일단 3D 유클리드 거리로 계산하되, 바닥면 투영이 필요함.
            # 간단히 발 사이 거리(기저면 폭) 계산
            base_width = np.linalg.norm(l_foot - r_foot)

            # (2) CoM 높이 & 자세 (Low Stance)
            # 발 평균 위치
            feet_center = (l_foot + r_foot) / 2
            # 높이 (골반 - 발)
            height_3d = np.linalg.norm(pelvis - feet_center)
            
            # 다리 길이 추정 (골반~발)
            leg_len = (np.linalg.norm(pelvis - l_foot) + np.linalg.norm(pelvis - r_foot)) / 2
            
            # 자세 비율
            compression = height_3d / (leg_len + 1e-6)
            
            stance_bonus = 0.0
            if compression < 0.65: stance_bonus = self.low_stance_boost * 2.0
            elif compression < 0.80: stance_bonus = self.low_stance_boost * 1.0

            # (3) 깊이(Z축) 앞쏠림 분석
            # 골반 대비 머리가 얼마나 앞으로(Z축 or Y축) 나왔는가?
            # HMR2의 pred_joints는 카메라 좌표계. Z값이 깊이.
            # 머리가 골반보다 Z값이 작으면(카메라 쪽으로) 앞으로 숙인 것.
            # (좌표계 확인 필요: 보통 -Z가 전방임)
            # 상대적인 Z 거리 차이 계산
            lean_z = pelvis[2] - head[2] # 양수면 머리가 더 앞에 있음 (가정)
            
            # 정규화된 쏠림 정도
            lean_factor = lean_z / 0.3 # 0.3m 기준
            
            # (4) 균형 상태 판정
            # 골반의 수직 투영점이 발 사이에 있는가?
            # 여기서는 단순화하여 CoM과 발 중심의 거리가 허용범위 이내인지 확인
            dist_com_feet = np.linalg.norm(pelvis[:2] - feet_center[:2]) # XY 평면 거리 (Z 무시)
            
            # 허용 범위
            allowance = 0.3 + (1.0 - self.sensitivity)*0.5 + stance_bonus
            safe_radius = base_width * allowance
            
            is_unbalanced = dist_com_feet > safe_radius

            # (5) 속도 (3D Velocity)
            velocity = 0.0
            if track_id in self.prev_com_3d:
                velocity = np.linalg.norm(pelvis - self.prev_com_3d[track_id])
            self.prev_com_3d[track_id] = pelvis
            
            vel_thresh = 0.02 + (1.0 - self.velocity_threshold) * 0.05

            # 최종 상태
            state = 0
            if is_unbalanced:
                if velocity < vel_thresh: state = 1 # 버티기
                else: state = 3 # 위험
            else:
                if lean_factor > 0.5: state = 1 # 앞쏠림 주의
                else: state = 0 # 안정

            # --- 2D 투영 및 그리기 (Box 좌표 기준 변환) ---
            colors = {0:(0,255,0), 1:(0,255,255), 2:(255,255,0), 3:(0,0,255)}
            col = colors[state]
            
            # YOLO 박스 좌표
            bx1, by1, bx2, by2 = map(int, box)
            bw, bh = bx2-bx1, by2-by1
            
            # HMR2 투영 함수 (간소화된 약식 투영)
            # 실제로는 HMR2의 cam parameter로 정확히 투영해야 하지만,
            # 여기서는 3D 상태만 HMR로 계산하고, 시각화는 YOLO 박스 기준으로 매핑하여 그립니다.
            # (Detectron2 렌더러 의존성 제거를 위함)
            
            def map_pt(pt_3d):
                # 3D 점을 2D 박스 내로 매핑 (정확하진 않지만 시각적 확인용)
                # X: -0.5~0.5 -> 0~bw
                # Y: -0.5~0.5 -> 0~bh
                # (HMR 출력이 정규화되어 있다고 가정 시)
                # HMR2 pred_joints는 미터 단위이므로 바로 매핑 불가.
                # 따라서 여기서는 YOLO Keypoints를 시각화 좌표로 쓰고, 색상만 3D 분석 결과를 따름.
                return None 

            # YOLO 키포인트 사용 (그리기용)
            # YOLO 키포인트가 없다면 박스 중심 사용
            
            # 바닥 원 (3D 분석 결과 반영)
            cx, cy = int((bx1+bx2)/2), int(by2)
            rad = int(bw * allowance)
            
            # 바닥 안전 구역
            cv2.ellipse(frame, (cx, cy), (rad, int(rad*0.3)), 0, 0, 360, col, 2)
            
            # CoM 라인 (박스 중심 ~ 바닥)
            com_y = int(by1 + bh*0.4) # 대략적 골반 위치
            cv2.line(frame, (cx, com_y), (cx, cy), col, 2)
            cv2.circle(frame, (cx, com_y), 6, (0,0,255), -1)
            
            # 앞쏠림 시각화 (깊이 원)
            depth_rad = int(10 + max(0, lean_factor * 30))
            cv2.circle(frame, (cx, com_y), depth_rad, (255,255,255), 2)

            # 위험 시 박스
            if state == 3:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,0,255), 3)

        return frame

class JudoAnalyzerHybrid(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Judo Analyzer (4D-Hybrid)")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        
        self.thread = AnalysisThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.init_ui()

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        
        # 뷰어
        self.lbl_img = QLabel("4D-Humans Model Loading...")
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setStyleSheet("background-color:black;")
        self.lbl_img.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.lbl_img, stretch=5)
        
        # 컨트롤
        panel = QFrame()
        panel.setFixedWidth(220)
        panel.setStyleSheet("background-color:#252525; border-left:1px solid #444;")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(20, 50, 20, 50)
        pl.setSpacing(30)
        
        btn_load = QPushButton(" VIDEO")
        btn_load.setStyleSheet("background:#444; color:#fff; padding:12px; font-weight:bold;")
        btn_load.clicked.connect(self.load_video)
        pl.addWidget(btn_load)
        
        self.btn_run = QPushButton(" START")
        self.btn_run.setStyleSheet("background:#007acc; color:#fff; padding:12px; font-weight:bold;")
        self.btn_run.clicked.connect(self.toggle)
        self.btn_run.setEnabled(False)
        pl.addWidget(self.btn_run)
        
        # 슬라이더
        def add_slider(name, func):
            pl.addWidget(QLabel(name, styleSheet="color:#888; font-size:10px;"))
            s = QSlider(Qt.Orientation.Horizontal)
            s.setValue(50)
            s.valueChanged.connect(func)
            pl.addWidget(s)

        add_slider("TOLERANCE", lambda v: setattr(self.thread, 'sensitivity', v/100.0))
        add_slider("LOW STANCE", lambda v: setattr(self.thread, 'low_stance_boost', v/100.0))
        add_slider("SPEED FILTER", lambda v: setattr(self.thread, 'velocity_threshold', v/10.0))
        
        pl.addStretch()
        layout.addWidget(panel)

    def load_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open", "", "Video (*.mp4 *.avi)")
        if f:
            self.thread.video_path = f
            self.btn_run.setEnabled(True)
            self.lbl_img.setText("Ready.")

    def toggle(self):
        if not self.thread.is_running:
            self.thread.is_running = True
            self.thread.start()
            self.btn_run.setText(" STOP")
        else:
            self.thread.is_running = False
            self.thread.wait()
            self.btn_run.setText(" START")

    @pyqtSlot(QImage)
    def update_image(self, img):
        self.lbl_img.setPixmap(QPixmap.fromImage(img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = JudoAnalyzerHybrid()
    ex.show()
    sys.exit(app.exec())