import sys
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import gradio as gr
from ultralytics import YOLO

# --- 4D-Humans Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "4D-Humans_disabled"))

try:
    from hmr2.models import load_hmr2, DEFAULT_CHECKPOINT
    from hmr2.utils.renderer import Renderer, cam_crop_to_full
    HMR2_AVAILABLE = True
except ImportError as e:
    print(f"[Warning] Failed to load 4D-Humans module: {e}")
    HMR2_AVAILABLE = False

class JudoAnalyzer:
    def __init__(self):
        # --- Settings ---
        self.sensitivity = 0.5
        self.low_stance_boost = 0.5
        self.velocity_threshold = 0.5
        
        # --- Model Initialization ---
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using device: {self.device}")

        # 1. Detector: YOLOv8
        print("Loading YOLOv8...")
        self.yolo = YOLO('yolov8n.pt')
        
        # 2. 3D Reconstruction: 4D-Humans (HMR2)
        self.hmr2 = None
        self.model_cfg = None
        self.renderer = None
        self.focal_length = None
        
        if HMR2_AVAILABLE:
            print("Loading 4D-Humans model (this may take a while)...")
            self.hmr2, self.model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
            self.hmr2 = self.hmr2.to(self.device)
            self.hmr2.eval()
            self.renderer = Renderer(self.model_cfg, self.hmr2.smpl.faces)
            self.focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH
            print("4D-Humans model loaded!")
        else:
            print("4D-Humans model is NOT available. Only YOLO will be used.")

        # HMR2 Input Preprocessor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.prev_com_3d = {}

    def get_3d_pose(self, frame, box):
        """ Crop YOLO box area and feed to HMR2 to get 3D coords """
        x1, y1, x2, y2 = map(int, box)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        if x2 <= x1 or y2 <= y1: return None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.hmr2(input_tensor)
        
        pred_joints = out['pred_joints'][0].cpu().numpy()
        pred_cam = out['pred_cam'][0].cpu().numpy()
        pred_vertices = out['pred_vertices'][0].cpu().numpy()

        return pred_joints, pred_cam, pred_vertices

    def process_frame_hybrid(self, frame):
        # 1. Detect humans with YOLO
        results = self.yolo.track(frame, persist=True, verbose=False)
        
        if not results or results[0].boxes.id is None:
            return frame

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        h_img, w_img = frame.shape[:2]

        for i, box in enumerate(boxes):
            track_id = ids[i]
            x1, y1, x2, y2 = box
            
            # 2. Extract true 3D coords with HMR2
            res = self.get_3d_pose(frame, box)
            if res is None: continue
            joints_3d, cam_param, vertices_3d = res

            # Render 3D mesh
            bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            bbox_size = max(x2 - x1, y2 - y1)
            
            img_size_tensor = torch.from_numpy(np.array([[w_img, h_img]])).float()
            full_cam_t = cam_crop_to_full(
                torch.from_numpy(cam_param[np.newaxis, :]).float(),
                torch.from_numpy(bbox_center[np.newaxis, :]).float(),
                torch.from_numpy(np.array([bbox_size])).float(),
                img_size_tensor,
                focal_length=self.focal_length
            ).squeeze(0).cpu().numpy()

            rendered_person_rgba = self.renderer(
                vertices_3d, 
                full_cam_t, 
                image=torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).permute(2,0,1).to(self.device),
                full_frame=True,
                scene_bg_color=(0,0,0)
            )
            
            rendered_person_bgr = cv2.cvtColor((rendered_person_rgba[:,:,:3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            alpha = rendered_person_rgba[:,:,3]

            for c in range(0, 3):
                frame[alpha > 0, c] = rendered_person_bgr[alpha > 0, c]

            # 3D Physical Analysis
            pelvis = joints_3d[0]
            l_foot = joints_3d[10]
            r_foot = joints_3d[11]
            head = joints_3d[15]

            # (1) Base of Support
            base_width_3d = np.linalg.norm(l_foot - r_foot)

            # (2) CoM Height & Stance
            feet_center_3d = (l_foot + r_foot) / 2
            height_3d = np.linalg.norm(pelvis - feet_center_3d)
            
            leg_len_3d = (np.linalg.norm(pelvis - l_foot) + np.linalg.norm(pelvis - r_foot)) / 2
            
            compression = height_3d / (leg_len_3d + 1e-6)
            
            stance_bonus = 0.0
            if compression < 0.65: stance_bonus = self.low_stance_boost * 2.0
            elif compression < 0.80: stance_bonus = self.low_stance_boost * 1.0

            # (3) Depth Lean Analysis
            lean_z = pelvis[2] - head[2] 
            lean_factor = lean_z / 0.3
            
            # (4) Balance
            dist_com_feet_3d = np.linalg.norm(pelvis - feet_center_3d)
            
            allowance = 0.3 + (1.0 - self.sensitivity)*0.5 + stance_bonus
            safe_radius_3d = base_width_3d * allowance
            
            is_unbalanced = dist_com_feet_3d > safe_radius_3d

            # (5) Velocity
            velocity = 0.0
            if track_id in self.prev_com_3d:
                velocity = np.linalg.norm(pelvis - self.prev_com_3d[track_id])
            self.prev_com_3d[track_id] = pelvis
            
            vel_thresh = 0.02 + (1.0 - self.velocity_threshold) * 0.05

            state = 0
            if is_unbalanced:
                if velocity < vel_thresh: state = 1 
                else: state = 3 
            else:
                if lean_factor > 0.5: state = 1 
                else: state = 0 

            # Visualization
            colors = {0:(0,255,0), 1:(0,255,255), 2:(255,255,0), 3:(0,0,255)}
            col = colors[state]
            
            com_projected_2d_x = int(bbox_center[0])
            com_projected_2d_y = int(y1 + (y2-y1)*0.4) 
            ground_y = int(y2)
            
            cv2.line(frame, (int(x1), ground_y), (int(x2), ground_y), col, 5)
            cv2.circle(frame, (com_projected_2d_x, com_projected_2d_y), 7, col, -1)
            cv2.line(frame, (com_projected_2d_x, com_projected_2d_y), (com_projected_2d_x, ground_y), col, 2)
            
            depth_radius = int(10 + (lean_factor * 15))
            cv2.circle(frame, (com_projected_2d_x, com_projected_2d_y), depth_radius, (255,255,255), 2)
            
            if state == 3:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 3)

        return frame

# --- Initialize Global Analyzer ---
# We initialize it once to avoid reloading models on every request
analyzer = JudoAnalyzer()

def process_video_gradio(input_video_path, sensitivity, low_stance, speed_filter):
    if not input_video_path:
        return None

    # Update settings
    analyzer.sensitivity = sensitivity
    analyzer.low_stance_boost = low_stance
    analyzer.velocity_threshold = speed_filter
    analyzer.prev_com_3d = {} # Reset tracking history

    cap = cv2.VideoCapture(input_video_path)
    
    # Output setup
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_cnt = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing video: {input_video_path} ({total_frames} frames)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_cnt += 1
        # Simple progress log
        if frame_cnt % 30 == 0:
            print(f"Processing frame {frame_cnt}/{total_frames}...")

        if analyzer.hmr2:
            frame = analyzer.process_frame_hybrid(frame)
        else:
            cv2.putText(frame, "4D-Humans Model Not Found", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        out.write(frame)

    cap.release()
    out.release()
    print("Processing complete.")
    return output_path

# --- Gradio UI ---
with gr.Blocks(title="Judo Analyzer (Web)") as app:
    gr.Markdown("# Judo Analyzer (4D-Humans Hybrid)")
    gr.Markdown("Upload a video to analyze Judo movements using YOLOv8 and 4D-Humans.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Input Video", format="mp4")
            
            gr.Markdown("### Analysis Options")
            s_sensitivity = gr.Slider(0.0, 1.0, value=0.5, label="Tolerance (Sensitivity)")
            s_low_stance = gr.Slider(0.0, 1.0, value=0.5, label="Low Stance Boost")
            s_speed = gr.Slider(0.0, 1.0, value=0.5, label="Speed Filter")
            
            btn_run = gr.Button("▶ Start Analysis", variant="primary")
            
        with gr.Column():
            video_output = gr.Video(label="Analyzed Video")
    
    btn_run.click(
        fn=process_video_gradio,
        inputs=[video_input, s_sensitivity, s_low_stance, s_speed],
        outputs=video_output
    )

if __name__ == "__main__":
    # Allow external access for Docker (server_name="0.0.0.0")
    app.launch(server_name="0.0.0.0", server_port=7860)
