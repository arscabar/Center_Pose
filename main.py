import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QSlider, QFrame, QGroupBox, QSizePolicy)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont
from ultralytics import YOLO

class AnalysisThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    # 텍스트 대신 색상 코드만 전송 (#00FF00, #FF0000 등)
    status_signal = pyqtSignal(str) 
    
    def __init__(self):
        super().__init__()
        self.model = YOLO('yolov8n-pose.pt')
        self.video_path = ""
        self.is_running = False
        
        # 설정값
        self.sensitivity = 0.5      
        self.low_stance_boost = 0.5 
        self.velocity_threshold = 0.5 
        
        self.frame_count = 0
        self.skip_interval = 2 
        self.last_results = None
        self.prev_data = {}

        # 생체역학 데이터
        self.body_weights = {
            'head': 0.081, 'torso': 0.497, 
            'arm': 0.05, 'leg': 0.1465 
        }
        
        self.bone_ratios = {
            'thigh': 0.245, 'shin': 0.246, 'torso': 0.28   
        }

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while cap.isOpened() and self.is_running:
            ret, frame = cap.read()
            if not ret: break

            self.frame_count += 1
            
            if self.frame_count % self.skip_interval == 0 or self.last_results is None:
                results = self.model.track(frame, persist=True, imgsz=640, verbose=False)
                self.last_results = results
            else:
                results = self.last_results

            # 프레임 처리
            frame, color_code = self.process_frame_silent(frame, results)
            
            # GUI로 색상만 전송
            self.status_signal.emit(color_code)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            p = qt_img.scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio)
            self.change_pixmap_signal.emit(p)

        cap.release()
        self.prev_data = {}
        self.last_results = None

    def estimate_z_depth(self, kps):
        """ [수정됨] KeyError 수정 및 Z축 추정 """
        def dist(i1, i2): return np.sqrt(np.sum((kps[i1][:2]-kps[i2][:2])**2))
        
        # 관측 길이
        obs_len = {
            'thigh': max(dist(11, 13), dist(12, 14)),
            'shin': max(dist(13, 15), dist(14, 16)),
            'torso': dist(5, 11)
        }
        
        scales = []
        if obs_len['shin'] > 10: scales.append(obs_len['shin'] / self.bone_ratios['shin'])
        if obs_len['thigh'] > 10: scales.append(obs_len['thigh'] / self.bone_ratios['thigh'])
        
        if not scales: return None
        current_scale = max(scales) 

        z_coords = np.zeros(17) 
        
        def solve_z(obs, ratio_key):
            actual = current_scale * self.bone_ratios[ratio_key]
            if actual > obs:
                return np.sqrt(actual**2 - obs**2)
            return 0.0

        # [오류 수정 구간] 괄호 위치 수정됨
        z_coords[15] = solve_z(dist(13, 15), 'shin')
        z_coords[16] = solve_z(dist(14, 16), 'shin')
        z_coords[13] = solve_z(dist(11, 13), 'thigh')
        z_coords[14] = solve_z(dist(12, 14), 'thigh')
        
        z_torso = solve_z(dist(5, 11), 'torso')
        z_coords[0] = z_torso * 1.2 

        kps_3d = []
        for i in range(17):
            kps_3d.append([kps[i][0], kps[i][1], z_coords[i]])
            
        return np.array(kps_3d), current_scale

    def calculate_precise_com(self, kps):
        def pt(idx): return kps[idx][:2]
        if kps[3][2]>0.5 and kps[4][2]>0.5: head = (pt(3)+pt(4))/2
        else: head = pt(0)
        torso = (pt(5)+pt(6)+pt(11)+pt(12))/4
        arms = (pt(5)+pt(7)+pt(9) + pt(6)+pt(8)+pt(10))/6
        legs = (pt(11)+pt(13)+pt(15) + pt(12)+pt(14)+pt(16))/6
        
        w = self.body_weights
        com = (head * w['head']) + (torso * w['torso']) + (arms * w['arm']*2) + (legs * w['leg']*2)
        total_w = w['head'] + w['torso'] + w['arm']*2 + w['leg']*2
        return com / total_w

    def calculate_posture(self, kps):
        def dist(i1, i2): return np.sqrt(np.sum((kps[i1][:2]-kps[i2][:2])**2))
        actual_leg_len = (dist(11,13)+dist(13,15) + dist(12,14)+dist(14,16)) / 2
        curr_height = abs(((kps[11][1]+kps[12][1])/2) - ((kps[15][1]+kps[16][1])/2))
        if actual_leg_len == 0: return 1.0
        return curr_height / actual_leg_len

    def process_frame_silent(self, frame, results):
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        gui_color = "#333333" # 기본 회색

        if not results or results[0].keypoints is None or results[0].boxes.id is None:
            return frame, gui_color

        kps_batch = results[0].keypoints.data.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        max_severity = -1

        for i, kps in enumerate(kps_batch):
            if i >= len(ids): break
            track_id = ids[i]
            if kps[15][2] < 0.5 or kps[16][2] < 0.5: continue

            # CoM
            com_xy = self.calculate_precise_com(kps)
            if track_id in self.prev_data:
                prev_com, _, _ = self.prev_data[track_id]
                com_xy = com_xy * 0.7 + prev_com * 0.3
            com_x, com_y = com_xy

            # 3D Depth
            z_res = self.estimate_z_depth(kps)
            lean_fwd = 0.0
            scale = 100.0
            if z_res:
                kps_3d, scale = z_res
                lean_fwd = np.mean(kps_3d[[0,5,6], 2]) / (scale * 0.3)

            # Posture
            compression = self.calculate_posture(kps)
            stance_bonus = 0.0
            if compression < 0.65: stance_bonus = self.low_stance_boost * 2.0
            elif compression < 0.80: stance_bonus = self.low_stance_boost * 1.0

            # Logic
            l_ank_x, r_ank_x = kps[15][0], kps[16][0]
            feet_center_x = (l_ank_x + r_ank_x) / 2
            base_width = abs(l_ank_x - r_ank_x)
            
            depth_penalty = max(0, abs(lean_fwd) - 0.5)
            allowance = 0.5 + (1.0 - self.sensitivity) + stance_bonus - depth_penalty
            safe_width = (base_width / 2) * max(0.2, allowance)
            
            is_outside = abs(com_x - feet_center_x) > safe_width

            # Velocity
            current_state = 0
            com_vel = 0
            feet_vel = 0
            
            if track_id in self.prev_data:
                prev_com, prev_feet, _ = self.prev_data[track_id]
                com_vel = abs(com_x - prev_com[0])
                feet_vel = abs(feet_center_x - prev_feet)
            
            self.prev_data[track_id] = (com_xy, feet_center_x, 0)
            vel_thresh = base_width * (0.02 + (self.velocity_threshold * 0.1))

            if is_outside:
                if com_vel < vel_thresh: current_state = 1
                else:
                    if feet_vel > (com_vel * 0.6): current_state = 2
                    else: current_state = 3
            else:
                if lean_fwd > 1.2: current_state = 1
                else: current_state = 0

            # 시각화 (텍스트 없음)
            colors = {0:(0,255,0), 1:(0,255,255), 2:(255,255,0), 3:(0,0,255)}
            zone_color = colors[current_state]
            
            ground_y = max(int(kps[15][1]), int(kps[16][1]))
            
            # 매트
            safe_l = int(feet_center_x - safe_width)
            safe_r = int(feet_center_x + safe_width)
            pts = np.array([[safe_l-20, ground_y+20], [safe_l+10, ground_y-10], 
                            [safe_r-10, ground_y-10], [safe_r+20, ground_y+20]], np.int32)
            cv2.fillPoly(overlay, [pts], zone_color)
            
            # 뼈대
            skel = [(5,7),(7,9),(6,8),(8,10),(11,13),(13,15),(12,14),(14,16),(5,6),(11,12)]
            for p1, p2 in skel:
                if kps[p1][2]<0.3 or kps[p2][2]<0.3: continue
                cv2.line(frame, (int(kps[p1][0]), int(kps[p1][1])), (int(kps[p2][0]), int(kps[p2][1])), zone_color, 2)

            # 헤드 라인
            head_pt = ((kps[3][0]+kps[4][0])/2, (kps[3][1]+kps[4][1])/2)
            neck = ((kps[5][0]+kps[6][0])/2, (kps[5][1]+kps[6][1])/2)
            if kps[3][2]>0.5:
                cv2.line(frame, (int(neck[0]), int(neck[1])), (int(head_pt[0]), int(head_pt[1])), (255,255,255), 3)

            # CoM 및 Z-Circle
            cv2.line(frame, (int(com_x), int(com_y)), (int(com_x), ground_y), zone_color, 2)
            cv2.circle(frame, (int(com_x), int(com_y)), 7, (0,0,255), -1)
            
            depth_radius = int(10 + (lean_fwd * 15))
            cv2.circle(overlay, (int(com_x), int(com_y)), depth_radius, (255, 255, 255), 2)
            
            if current_state == 3:
                cv2.rectangle(overlay, (0,0), (w,h), (0,0,255), 20)
            
            # 우측 패널 색상 결정 (가장 높은 위험도 기준)
            if current_state >= max_severity:
                max_severity = current_state
                state_cols = {0:"#00FF00", 1:"#FFFF00", 2:"#00FFFF", 3:"#FF0000"}
                gui_color = state_cols[current_state]

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        return frame, gui_color

class JudoAnalyzerNoText(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Judo Analyzer (Visual Only)")
        self.setGeometry(100, 100, 1300, 750)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        
        self.thread = AnalysisThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.status_signal.connect(self.update_status_color)
        
        self.init_ui()

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        layout.setContentsMargins(20,20,20,20)
        layout.setSpacing(30)
        
        # 1. 영상
        self.lbl_img = QLabel("Video Ready")
        self.lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setStyleSheet("background-color:black; border:2px solid #333; border-radius:15px;")
        self.lbl_img.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.lbl_img, stretch=4)
        
        # 2. 우측 패널 (텍스트 없음)
        panel = QFrame()
        panel.setStyleSheet("QFrame{background-color:#2d2d2d; border-radius:15px;} QLabel{font-weight:bold; color:#ccc;}")
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(30, 40, 30, 40)
        pl.setSpacing(20)
        
        # 상태 표시기 (단순 컬러 박스)
        self.status_box = QFrame()
        self.status_box.setStyleSheet("background-color:#333; border-radius:12px; border:2px solid #555;")
        self.status_box.setFixedHeight(80)
        pl.addWidget(self.status_box)
        
        # 버튼
        btn_load = QPushButton("📂 영상 열기")
        btn_load.setStyleSheet("background:#444; padding:15px; border-radius:8px;")
        btn_load.clicked.connect(self.load_video)
        pl.addWidget(btn_load)
        
        self.btn_run = QPushButton("▶ 분석 시작")
        self.btn_run.setStyleSheet("background:#007acc; padding:15px; border-radius:8px; font-weight:bold;")
        self.btn_run.clicked.connect(self.toggle)
        self.btn_run.setEnabled(False)
        pl.addWidget(self.btn_run)
        
        # 설정 슬라이더 (라벨은 조작을 위해 최소한으로 유지)
        gs = QGroupBox("분석 옵션")
        gs.setStyleSheet("QGroupBox{border:1px solid #555; border-radius:8px; margin-top:10px; padding-top:15px;}")
        gl = QVBoxLayout()
        gl.setSpacing(15)
        
        gl.addWidget(QLabel("Tolerance"))
        s1 = QSlider(Qt.Orientation.Horizontal)
        s1.setValue(50)
        s1.valueChanged.connect(lambda v: setattr(self.thread, 'sensitivity', v/100.0))
        gl.addWidget(s1)
        
        gl.addWidget(QLabel("Low Stance"))
        s2 = QSlider(Qt.Orientation.Horizontal)
        s2.setValue(50)
        s2.valueChanged.connect(lambda v: setattr(self.thread, 'low_stance_boost', v/100.0))
        gl.addWidget(s2)
        
        gl.addWidget(QLabel("Speed Filter"))
        s3 = QSlider(Qt.Orientation.Horizontal)
        s3.setValue(50)
        s3.valueChanged.connect(lambda v: setattr(self.thread, 'velocity_threshold', v/10.0))
        gl.addWidget(s3)
        
        gs.setLayout(gl)
        pl.addWidget(gs)
        pl.addStretch()
        
        layout.addWidget(panel, stretch=1)

    def load_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open", "", "Video (*.mp4 *.avi *.mkv)")
        if f:
            self.thread.video_path = f
            self.btn_run.setEnabled(True)

    def toggle(self):
        if not self.thread.is_running:
            self.thread.is_running = True
            self.thread.start()
            self.btn_run.setText("⏹ 중지")
            self.btn_run.setStyleSheet("background:#d32f2f; padding:15px; border-radius:8px;")
        else:
            self.thread.is_running = False
            self.thread.wait()
            self.btn_run.setText("▶ 분석 시작")
            self.btn_run.setStyleSheet("background:#007acc; padding:15px; border-radius:8px;")

    @pyqtSlot(QImage)
    def update_image(self, img):
        self.lbl_img.setPixmap(QPixmap.fromImage(img))
        
    @pyqtSlot(str)
    def update_status_color(self, color_code):
        # 텍스트 없이 박스 색상만 변경
        self.status_box.setStyleSheet(f"background-color:{color_code}; border-radius:12px; border:3px solid #FFF;")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = JudoAnalyzerNoText()
    ex.show()
    sys.exit(app.exec())