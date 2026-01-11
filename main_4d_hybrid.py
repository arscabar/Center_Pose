import sys
import os
# [FIX] Import PySide6 before cv2 to avoid plugin conflicts
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QSlider, QFrame, QGroupBox, QSizePolicy, QLineEdit)
from PySide6.QtCore import Qt, QThread, Signal as pyqtSignal, Slot as pyqtSlot
from PySide6.QtGui import QImage, QPixmap, QFont
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO

# --- 4D-Humans 경로 추가 ---
# 현재 파일 위치 기준 4D-Humans 폴더를 시스템 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "4D-Humans_disabled"))

try:
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
    # Import the Renderer from local directory
    from renderer import Renderer, cam_crop_to_full
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
        
        # 1. 탐지기: YOLOv8
        self.yolo = YOLO('yolov8n.pt')
        
        # 2. 3D 복원: 4D-Humans (HMR2)
        self.hmr2 = None
        self.model_cfg = None # Add model_cfg
        self.renderer = None # Add renderer
        self.focal_length = None # Add focal_length
        if HMR2_AVAILABLE:
            print("[INFO] 4D-Humans 모델 로딩 중... (시간이 좀 걸립니다)")
            self.hmr2, self.model_cfg = load_hmr2(DEFAULT_CHECKPOINT) # Get model_cfg here
            self.hmr2 = self.hmr2.to(self.device)
            self.hmr2.eval()
            self.renderer = Renderer(self.model_cfg, self.hmr2.smpl.faces) # Initialize renderer
            self.focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH # Store focal length
            print("[INFO] 4D-Humans 모델 로딩 완료!")

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
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        if x2 <= x1 or y2 <= y1: return None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.hmr2({'img': input_tensor})
        
        # [DEBUG] Check available keys
        # print(f"[DEBUG] Model output keys: {out.keys()}")

        if 'pred_joints' in out:
            pred_joints = out['pred_joints'][0].cpu().numpy()
        elif 'pred_keypoints_3d' in out:
            pred_joints = out['pred_keypoints_3d'][0].cpu().numpy()
        else:
            print(f"[ERROR] Cannot find joints in output. Keys: {out.keys()}")
            return None

        if 'pred_cam' in out:
            pred_cam = out['pred_cam'][0].cpu().numpy()
        elif 'pred_cam_t' in out: # Sometimes it might be named differently
             pred_cam = out['pred_cam_t'][0].cpu().numpy()
        else:
             # Fallback or error
             print(f"[ERROR] Cannot find camera params in output. Keys: {out.keys()}")
             return None

        pred_vertices = out['pred_vertices'][0].cpu().numpy() # Extract pred_vertices

        return pred_joints, pred_cam, pred_vertices # Return pred_vertices

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
            # Now get pred_vertices as well
            res = self.get_3d_pose(frame, box)
            if res is None: continue
            joints_3d, cam_param, vertices_3d = res # Unpack pred_vertices

            # Calculate bbox_center and bbox_size for cam_crop_to_full
            x1, y1, x2, y2 = box
            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            bbox_size = max(x2 - x1, y2 - y1)
            
            # Convert weak perspective camera parameters to full camera translation
            img_size_tensor = torch.from_numpy(np.array([[w_img, h_img]])).float()
            full_cam_t = cam_crop_to_full(
                torch.from_numpy(cam_param[np.newaxis, :]).float(), # cam_param needs to be (1, 3)
                torch.from_numpy(bbox_center[np.newaxis, :]).float(), # bbox_center needs to be (1, 2)
                torch.from_numpy(np.array([bbox_size])).float(), # bbox_size needs to be (1,)
                img_size_tensor,
                focal_length=self.focal_length
            ).squeeze(0).cpu().numpy() # Convert back to numpy (3,)

            # Render the 3D mesh onto the image
            # [FIX] imgname=frame 추가하여 빈 문자열 문제 해결
            rendered_person_rgba = self.renderer(
                vertices_3d,
                full_cam_t,
                image=torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2,0,1).to(self.device),
                full_frame=True,
                imgname=frame,  # [FIX] numpy 배열로 프레임 직접 전달
                scene_bg_color=(0,0,0),
                return_rgba=True
            )
            
            # Convert rendered output to BGR for OpenCV
            rendered_person_bgr = cv2.cvtColor((rendered_person_rgba[:,:,:3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            alpha = rendered_person_rgba[:,:,3] # Alpha channel

            # Overlay the rendered person onto the original frame using the alpha channel
            # For c in range(0, 3) (blue, green, red channels)
            for c in range(0, 3):
                frame[alpha > 0, c] = rendered_person_bgr[alpha > 0, c]


            # SMPL Index mapping:
            # HMR2 pred_joints are 49 joints for SMPL-X (24 SMPL, 25 extra like face/hands)
            # Standard SMPL 24 joints:
            # 0-Pelvis, 1-L_Hip, 2-R_Hip, 3-Spine1, 4-L_Knee, 5-R_Knee, 6-Spine2, 7-L_Ankle, 8-R_Ankle,
            # 9-Spine3, 10-L_Foot, 11-R_Foot, 12-Neck, 13-L_Collar, 14-R_Collar, 15-Head,
            # 16-L_Shoulder, 17-R_Shoulder, 18-L_Elbow, 19-R_Elbow, 20-L_Wrist, 21-R_Wrist,
            # 22-L_Hand, 23-R_Hand

            # Using specific indices for pelvis, feet, and head for 3D physical analysis.
            pelvis_idx = 0 # SMPL pelvis
            l_foot_idx = 10 # SMPL left foot
            r_foot_idx = 11 # SMPL right foot
            head_idx = 15 # SMPL head top

            pelvis = joints_3d[pelvis_idx]
            l_foot = joints_3d[l_foot_idx]
            r_foot = joints_3d[r_foot_idx]
            head = joints_3d[head_idx]


            # --- 3D 물리 분석 ---
            
            # (1) 기저면 (Base of Support)
            base_width_3d = np.linalg.norm(l_foot - r_foot)

            # (2) CoM 높이 & 자세 (Low Stance)
            feet_center_3d = (l_foot + r_foot) / 2
            height_3d = np.linalg.norm(pelvis - feet_center_3d)
            
            leg_len_3d = (np.linalg.norm(pelvis - l_foot) + np.linalg.norm(pelvis - r_foot)) / 2
            
            compression = height_3d / (leg_len_3d + 1e-6)
            
            stance_bonus = 0.0
            if compression < 0.65: stance_bonus = self.low_stance_boost * 2.0
            elif compression < 0.80: stance_bonus = self.low_stance_boost * 1.0

            # (3) 깊이(Z축) 앞쏠림 분석
            lean_z = pelvis[2] - head[2] 
            
            lean_factor = lean_z / 0.3 # 0.3m as a reference for significant lean
            
            # (4) 균형 상태 판정
            dist_com_feet_3d = np.linalg.norm(pelvis - feet_center_3d) # Distance between 3D pelvis and 3D feet center
            
            allowance = 0.3 + (1.0 - self.sensitivity)*0.5 + stance_bonus
            safe_radius_3d = base_width_3d * allowance
            
            is_unbalanced = dist_com_feet_3d > safe_radius_3d

            # (5) 속도 (3D Velocity)
            velocity = 0.0
            if track_id in self.prev_com_3d:
                velocity = np.linalg.norm(pelvis - self.prev_com_3d[track_id])
            self.prev_com_3d[track_id] = pelvis
            
            vel_thresh = 0.02 + (1.0 - self.velocity_threshold) * 0.05

            # Final state determination
            state = 0
            if is_unbalanced:
                if velocity < vel_thresh: state = 1 # Stalling
                else: state = 3 # Danger
            else:
                if lean_factor > 0.5: state = 1 # Leaning forward caution
                else: state = 0 # Stable

            # --- 2D Visualization Adaptation (Overlays on 3D mesh) ---
            colors = {0:(0,255,0), 1:(0,255,255), 2:(255,255,0), 3:(0,0,255)}
            col = colors[state]
            
            # Project 3D pelvis and feet to 2D for drawing indicators
            # This requires a function to project 3D world coordinates to 2D image coordinates given camera parameters.
            # Since full_cam_t describes the camera, we can use a simple projection from pyrender's camera.

            # Simplified 2D projection of pelvis and feet center for drawing indicators
            # This is an approximation. For exact projection, one would typically use
            # intrinsic matrix (focal_length, principal_point) and extrinsic (rotation, translation)
            # from the pyrender camera setup.
            
            # Given cam_param = [s, tx, ty]
            # s is scale, (tx, ty) is 2D translation in the normalized camera frame.
            # We need to transform 3D points from camera space to 2D image plane.
            # For now, we'll approximate the CoM and ground projection based on the bounding box and 3D info.

            # Approximate 2D location for drawing CoM and ground indicators.
            # We'll use the center of the bounding box as the base for these 2D projections.
            com_projected_2d_x = int(bbox_center[0])
            # Project the 3D pelvis Z (depth) to affect its apparent Y position slightly,
            # or just use an approximate fixed ratio within the bbox for simplicity.
            # Using a fixed ratio for now:
            com_projected_2d_y = int(y1 + (y2-y1)*0.4) # Approximate pelvis height within bbox
            
            # Ground line / "mat" indicator
            # We can draw a simple line or ellipse at the bottom of the bounding box,
            # colored according to the state, representing the base of support.
            ground_y = int(y2)
            cv2.line(frame, (int(x1), ground_y), (int(x2), ground_y), col, 5) # Ground line
            
            # Draw CoM and Z-depth (lean) indicators
            cv2.circle(frame, (com_projected_2d_x, com_projected_2d_y), 7, col, -1)
            cv2.line(frame, (com_projected_2d_x, com_projected_2d_y), (com_projected_2d_x, ground_y), col, 2)
            
            depth_radius = int(10 + (lean_factor * 15))
            cv2.circle(frame, (com_projected_2d_x, com_projected_2d_y), depth_radius, (255,255,255), 2)
            
            # Red box for danger state
            if state == 3:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 3)

        return frame

class JudoAnalyzerHybrid(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Judo Analyzer (4D-Hybrid)")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        
        self.thread = AnalysisThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.setAcceptDrops(True) # Enable Drag & Drop
        self.init_ui()

        # Check for command line argument
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
            if os.path.exists(video_path):
                print(f"[INFO] Loading video from arguments: {video_path}")
                self.thread.video_path = video_path
                if HMR2_AVAILABLE:
                    self.btn_run.setEnabled(True)
                    self.lbl_img.setText(f"Video loaded: {os.path.basename(video_path)}. Click START.")
                else:
                    self.lbl_img.setText("Video loaded, but model unavailable.")
            else:
                print(f"[WARN] File not found: {video_path}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            video_path = files[0]
            if os.path.exists(video_path):
                self.thread.video_path = video_path
                self.txt_path.setText(video_path) # Update text field
                if HMR2_AVAILABLE:
                    self.btn_run.setEnabled(True)
                    self.lbl_img.setText(f"Video loaded: {os.path.basename(video_path)}. Click START.")
                else:
                    self.lbl_img.setText("Video loaded, but model unavailable.")

    def init_ui(self):
        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        
        # 뷰어
        if not HMR2_AVAILABLE:
            self.lbl_img = QLabel("4D-Humans Model Not Available. Check 4D-Humans folder and dependencies.")
        else:
            self.lbl_img = QLabel("Drag & Drop video here or click VIDEO to browse.")
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
        pl.setSpacing(20) # Reduced spacing to fit more
        
        # [NEW] File Path Input & Button
        pl.addWidget(QLabel("FILE PATH:", styleSheet="color:#888; font-size:10px; font-weight:bold;"))
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Paste path (/mnt/c/...)")
        self.txt_path.setStyleSheet("background:#333; color:#fff; border:1px solid #555; padding:5px;")
        pl.addWidget(self.txt_path)

        btn_load_text = QPushButton("LOAD PATH")
        btn_load_text.setStyleSheet("background:#555; color:#fff; padding:8px; font-size:11px;")
        btn_load_text.clicked.connect(self.load_video_from_text)
        pl.addWidget(btn_load_text)
        
        pl.addSpacing(10)
        
        btn_load = QPushButton("BROWSE VIDEO")
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
            pl.addWidget(QLabel(name, styleSheet="color:#888; font-size:10px; margin-top:10px;"))
            s = QSlider(Qt.Orientation.Horizontal)
            s.setValue(50)
            s.valueChanged.connect(func)
            pl.addWidget(s)

        add_slider("TOLERANCE", lambda v: setattr(self.thread, 'sensitivity', v/100.0))
        add_slider("LOW STANCE", lambda v: setattr(self.thread, 'low_stance_boost', v/100.0))
        add_slider("SPEED FILTER", lambda v: setattr(self.thread, 'velocity_threshold', v/10.0))
        
        pl.addStretch()
        layout.addWidget(panel)

    def load_video_from_text(self):
        path = self.txt_path.text().strip()
        # Remove quotes if user pasted them
        path = path.strip('"').strip("'")
        
        if path and os.path.exists(path):
            self.thread.video_path = path
            if HMR2_AVAILABLE:
                self.btn_run.setEnabled(True)
                self.lbl_img.setText(f"Video loaded: {os.path.basename(path)}. Click START.")
            else:
                self.lbl_img.setText("Video loaded, but model unavailable.")
        else:
            self.lbl_img.setText(f"File not found:\n{path}")

    def load_video(self):
        # Default dir set to /mnt to show host drives (c, d, e...)
        start_dir = "/mnt" if os.path.exists("/mnt") else ""
        f, _ = QFileDialog.getOpenFileName(self, "Open Video", start_dir, "Video (*.mp4 *.avi *.mkv)")
        if f:
            self.thread.video_path = f
            self.txt_path.setText(f) # Update text field
            # Enable run button only if HMR2 model is available
            if HMR2_AVAILABLE:
                self.btn_run.setEnabled(True)
                self.lbl_img.setText("Video loaded. Click START to begin analysis.")
            else:
                self.btn_run.setEnabled(False) # Ensure disabled if HMR2 is not available
                self.lbl_img.setText("Video loaded, but 4D-Humans Model not available for analysis.")

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