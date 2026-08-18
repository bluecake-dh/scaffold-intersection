import os
from dotenv import load_dotenv
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import threading
import time
from inference import get_model

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("MY_API_KEY")
MODEL_ID = "dereks-workspace-pw4df/scaffold_keypoint-1-yolo26n-pose-t1"

"""
Members:
-- Widgets --
    root
    canvas
    lbl_status
    progress_bar
    btn_file_browse
    lbl_file
    lbl_file_preview
    lbl_resize
    input_resize
    btn_preprocess
    btn_map
    btn_gen_print
    lbl_intersections_preview
    lbl_intersections_count
    input_confidence
    input_inter_sep
    input_win_size
    input_win_overlap

-- Data --
    model
    selected_file_path
    auto_preprocess
    attempt_clean
    preprocess_size
    confidence_threshold
    sliding_window_size
    sliding_window_overlap
    intersection_separation
    preprocessed_img
    annotated_img
"""
class ScaffoldPrintGUI:
    def __init__(self, root):
        self.model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
        self.selected_file_path = ""
        self.preprocessed_img = None
        self.annotated_img = None
        self.preprocess_size = 0
        self.confidence_threshold = 0.3
        self.sliding_window_size = 640
        self.sliding_window_overlap = 330
        self.intersection_separation = 10

        """Initialize GUI"""
        self.root = root
        self.root.title("Scaffold Print File Generator")
        self.root.geometry("1000x800")
        self.canvas = tk.Canvas(root, highlightthickness=0)
        scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        root_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=root_frame, anchor="nw")
        root_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel) # Win/macOS
        self.canvas.bind_all("<Button-4>", self._on_mouse_wheel)   # Linux
        self.canvas.bind_all("<Button-5>", self._on_mouse_wheel)   # Linux
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.canvas.pack(side=tk.LEFT, fill="both", expand=True, pady=(0, 70))

        # -- Status Frame
        status_frame = tk.Frame(root, height=70, bg="#bbbbbb")
        status_frame.pack_propagate(False)
        status_frame.place(relx=0.0, rely=1.0, relwidth=1.0, anchor="sw")
        self.lbl_status = tk.Label(status_frame, text="No current processes", bg="#bbbbbb", padx=10, pady=10)
        self.lbl_status.pack()
        self.progress_bar = ttk.Progressbar(status_frame, mode="indeterminate")

        # -- File select frame
        file_frame = tk.LabelFrame(root_frame, text=" 1. Select Scaffold Image ", padx=10, pady=10)
        file_frame.pack(fill="x", padx=15, pady=15)
        file_frame.rowconfigure(0, weight=1)
        file_frame.columnconfigure(0, weight=1)
        file_frame.columnconfigure(1, weight=1)
        file_frame_left = tk.Frame(file_frame)
        file_frame_left.grid(row=0, column=0, sticky="nw")
        file_frame_right = tk.Frame(file_frame)
        file_frame_right.grid(row=0, column=1, sticky="ne", pady=(0, 10))
        
        # Left side
        lbl_file_browse = tk.Label(file_frame_left, text="Valid formats:  .tif  .png\n(Ideal size: >1000x1000)")
        lbl_file_browse.pack(anchor="w", pady=(0, 10))

        self.btn_file_browse = tk.Button(file_frame_left, text="Browse File", command=self.browse_file)
        self.btn_file_browse.pack(anchor="w")

        self.lbl_file = tk.Label(file_frame_left, text="No file selected...", wraplength=300, justify="left")
        self.lbl_file.pack(anchor="w", pady=(5, 0))

        # Right side
        file_preview_frame = tk.LabelFrame(file_frame_right, text="Preview Original Image", width=350, height=350)
        file_preview_frame.propagate(False)
        file_preview_frame.pack()

        self.lbl_file_preview = tk.Label(file_preview_frame)
        self.lbl_file_preview.pack(expand=True)

        # -- Preprocessing frame
        preprocess_frame = tk.LabelFrame(root_frame, text=" 2. Preprocess Image ", padx=10, pady=10)
        preprocess_frame.pack(fill="x", padx=15, pady=10)
        preprocess_frame.rowconfigure(0, weight=1)
        preprocess_frame.columnconfigure(0, weight=1)
        preprocess_frame.columnconfigure(1, weight=1)
        preprocess_frame_left = tk.Frame(preprocess_frame)
        preprocess_frame_left.grid(row=0, column=0, sticky="nw")
        preprocess_frame_right = tk.Frame(preprocess_frame)
        preprocess_frame_right.grid(row=0, column=1, sticky="ne", pady=(0, 10))

        # Left side
        lbl_execution = tk.Label(preprocess_frame_left, text="Preprocessing Options")
        lbl_execution.pack(anchor="w", pady=(0, 10))

        self.auto_preprocess = tk.BooleanVar()
        self.auto_preprocess.set(True)
        chk_auto_preprocess = tk.Checkbutton(preprocess_frame_left, text="Automatically preprocess image on file upload", variable=self.auto_preprocess)
        chk_auto_preprocess.pack(anchor="w")
        self.attempt_clean = tk.BooleanVar()
        self.attempt_clean.set(False)
        chk_attempt_clean = tk.Checkbutton(preprocess_frame_left, text="Attempt to clean and sharpen scaffold", variable=self.attempt_clean)
        chk_attempt_clean.pack(anchor="w")

        self.lbl_resize = tk.Label(preprocess_frame_left, text="Resize larger dimension to [x]:\n(Example: input 2000 for 1200x1000 -> 2000x1667)\n(Example: input 800 for 1200x1000 -> 800x667)", justify="left")
        self.lbl_resize.pack(anchor="w", pady=(10, 0))
        self.input_resize = tk.Entry(preprocess_frame_left, width=10)
        self.input_resize.pack(anchor="w")
        
        self.btn_preprocess = tk.Button(preprocess_frame_left, text="Preprocess Image", command=self.preprocess_image, state="disabled")
        self.btn_preprocess.pack(anchor="w", pady=(10, 0))

        self.btn_dwnld_preprocess = tk.Button(preprocess_frame_left, text="Download Preprocessed Image", command=self.download_preprocessed_image, state="disabled")
        self.btn_dwnld_preprocess.pack(anchor="w", pady=(10, 0))

        # Right side
        preprocess_preview_frame = tk.LabelFrame(preprocess_frame_right, text="Preview Preprocessed Image", width=350, height=350)
        preprocess_preview_frame.propagate(False)
        preprocess_preview_frame.pack()

        self.lbl_preprocess_preview = tk.Label(preprocess_preview_frame)
        self.lbl_preprocess_preview.pack(expand=True)

        # -- Mapping frame
        mapping_frame = tk.LabelFrame(root_frame, text=" 3. Map Print Locations ", padx=10, pady=10)
        mapping_frame.pack(fill="x", padx=15, pady=10)
        mapping_frame.rowconfigure(0, weight=1)
        mapping_frame.columnconfigure(0, weight=1)
        mapping_frame.columnconfigure(1, weight=1)
        mapping_frame_left = tk.Frame(mapping_frame)
        mapping_frame_left.grid(row=0, column=0, sticky="nw")
        mapping_frame_right = tk.Frame(mapping_frame)
        mapping_frame_right.grid(row=0, column=1, sticky="ne", pady=(0, 10))

        # Left side
        lbl_execution = tk.Label(
            mapping_frame_left,
            text="Tips:\n" \
            "Try to upscale image if unsatisfied with result\n" \
            "Try decrease window size if unsatisfied with result",
            justify="left"
        )
        lbl_execution.pack(anchor="w", pady=(0, 10))

        lbl_confidence = tk.Label(mapping_frame_left, text="Confidence Threshold (0 - 1):")
        lbl_confidence.pack(anchor="w")
        self.input_confidence = tk.Entry(mapping_frame_left, width=10)
        self.input_confidence.pack(anchor="w", pady=(0, 10))
        self.input_confidence.insert(0, str(self.confidence_threshold))

        lbl_inter_sep = tk.Label(mapping_frame_left, text="Approx. distance between intersections (px):")
        lbl_inter_sep.pack(anchor="w")
        self.input_inter_sep = tk.Entry(mapping_frame_left, width=10)
        self.input_inter_sep.pack(anchor="w", pady=(0, 10))
        self.input_inter_sep.insert(0, str(self.intersection_separation))

        lbl_win_size = tk.Label(mapping_frame_left, text="Sliding window size (px):")
        lbl_win_size.pack(anchor="w")
        self.input_win_size = tk.Entry(mapping_frame_left, width=10)
        self.input_win_size.pack(anchor="w", pady=(0, 10))
        self.input_win_size.insert(0, str(self.sliding_window_size))

        lbl_win_overlap = tk.Label(mapping_frame_left, text="Sliding window overlap (px):")
        lbl_win_overlap.pack(anchor="w")
        self.input_win_overlap = tk.Entry(mapping_frame_left, width=10)
        self.input_win_overlap.pack(anchor="w", pady=(0, 10))
        self.input_win_overlap.insert(0, str(self.sliding_window_overlap))

        self.btn_map = tk.Button(mapping_frame_left, text="Map Intersections", command=self.map_intersections, state="disabled")
        self.btn_map.pack(anchor="w", pady=(10, 0))

        self.btn_preview_viewer = tk.Button(mapping_frame_left, text="View Larger Preview in New Window", command=self.view_large_preview, state="disabled")
        self.btn_preview_viewer.pack(anchor="w", pady=(10, 0))

        self.btn_gen_print = tk.Button(mapping_frame_left, text="Generate Print File", command=self.generate_print, state="disabled")
        self.btn_gen_print.pack(anchor="w", pady=(50, 0))

        # Right side
        intersections_preview_frame = tk.LabelFrame(mapping_frame_right, text="Preview Intersections", width=350, height=350)
        intersections_preview_frame.propagate(False)
        intersections_preview_frame.pack()

        self.lbl_intersections_preview = tk.Label(intersections_preview_frame)
        self.lbl_intersections_preview.pack(expand=True)

        self.lbl_intersections_count = tk.Label(mapping_frame_right)
        self.lbl_intersections_count.pack()


    """
    UI CONTROLS
    """
    def _on_frame_configure(self, event):
        bbox = self.canvas.bbox("all")
        if not bbox:
            return

        content_height = bbox[3] - bbox[1]
        canvas_height = self.canvas.winfo_height()

        # Only allow scrolling when content does not fit window
        if content_height > canvas_height:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        else:
            self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), canvas_height))


    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas.find_withtag("all")[0], width=event.width)


    def _on_mouse_wheel(self, event):
        if event.num == 5 or event.delta == -120:
            self.canvas.yview_scroll(1, "units")
        elif event.num == 4 or event.delta == 120:
            self.canvas.yview_scroll(-1, "units")

    
    def _display_image(self, img, lbl, max_size):
        if img is None:
            lbl.config(text="Error loading file")
            return False
            
        # Resize to fit frame
        img_resized = self._resize_image(img, max_size)

        # Convert to RGB format
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        success, img_compressed = cv2.imencode(".png", img_rgb)
        if not success:
            lbl.config(text="Error loading file")
            return False
        img_bytes = img_compressed.tobytes()
        tk_img = tk.PhotoImage(data=img_bytes)
        lbl.config(image=tk_img, text="")
        lbl.image = tk_img
        return True
    

    def _lock_ui(self):
        self.progress_bar.pack(fill="x", padx=50)
        self.progress_bar.start()
        self.btn_file_browse.config(state="disabled")
        self.btn_preprocess.config(state="disabled")
        self.btn_dwnld_preprocess.config(state="disabled")
        self.btn_map.config(state="disabled")
        self.btn_preview_viewer.config(state="disabled")
        self.btn_gen_print.config(state="disabled")


    def _unlock_ui(self):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.lbl_status.config(text="No current processes")
        self.btn_file_browse.config(state="normal")
        self.btn_preprocess.config(state="normal")
        self.btn_dwnld_preprocess.config(state="normal")
        if self.preprocessed_img is not None:
            self.btn_map.config(state="normal")


    def _update_status_from_thread(self, msg):
        self.root.after(0, lambda: self.lbl_status.config(text=msg))
        

    """
    UI INTERACTION
    """
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a File",
            filetypes=[
                ("Images", "*.tif *.png")
            ]
        )

        if not file_path:
            return
        
        self.selected_file_path = file_path
        self.lbl_file.config(text=f"Selected: {file_path}")

        # Preview selected image
        img = cv2.imread(file_path)
        success = self._display_image(img, self.lbl_file_preview, 300)
        h, w = img.shape[:2]
        self.lbl_resize.config(text=f"Resize larger dimension to [x]:\n(Example: input 2000 for 1200x1000 -> 2000x1667)\n(Example: input 800 for 1200x1000 -> 800x667)\nORIGINAL: {h}x{w}")
        if not success:
            return

        # Move on to preprocessing automatically if indicated
        if self.auto_preprocess.get():
            self.preprocess_image()
        else:
            self.btn_preprocess.config(state="normal")


    def preprocess_image(self):
        if self.selected_file_path is None:
            self.lbl_status.config("Error with file path, reselect image")
            self.btn_preprocess.config(state="disabled")
            self.btn_map.config(state="disabled")
            self.btn_gen_print.config(state="disabled")
            return

        self.preprocess_size = self.input_resize.get()
        if self.preprocess_size:
            try:
                self.preprocess_size = float(self.preprocess_size)
                if self.preprocess_size < 0:
                    raise ValueError
            except:
                self.lbl_status.config("Bad resize")
                self.btn_map.config(state="disabled")
                self.btn_gen_print.config(state="disabled")
                return
        
        self._lock_ui()
        threading.Thread(target=self._preprocess_image, daemon=True).start()


    def download_preprocessed_image(self):
        file = self.selected_file_path.split(os.sep)[-1]
        cv2.imwrite(f"pp_{file}", self.preprocessed_img)

    
    def map_intersections(self):
        if self.preprocessed_img is None:
            self.lbl_status.config("Error with preprocessed image, rerun preprocessing")
            self.btn_map.config(state="disabled")
            self.btn_gen_print.config(state="disabled")
            return
        
        try:
            self.confidence_threshold = float(self.input_confidence.get())
            if not 0 < self.confidence_threshold < 1:
                raise ValueError
        except:
            self.lbl_status.config(text="Bad confidence threshold")
            return

        try:
            self.sliding_window_size = int(self.input_win_size.get())
            self.sliding_window_overlap = int(self.input_win_overlap.get())
            if self.sliding_window_size < 1:
                raise ValueError
            if not 0 < self.sliding_window_overlap < self.sliding_window_size:
                raise ValueError
        except:
            self.lbl_status.config(text="Bad sliding window configuration")
            return

        self._lock_ui()
        threading.Thread(target=self._map_intersections, daemon=True).start()


    def view_large_preview(self):
        new_window = tk.Toplevel(self.root)
        new_window.title("Image Viewer")
        new_window.geometry("900x900")
        lbl_img = tk.Label(new_window, text="")
        lbl_img.pack()
        self._display_image(self.annotated_img, lbl_img, 900)


    def generate_print(self):
        pass

    
    """
    CORE LOGIC
    """
    def _preprocess_image(self):
        self._update_status_from_thread("Preprocessing")
        self.preprocessed_img = cv2.imread(self.selected_file_path, cv2.IMREAD_GRAYSCALE)

        # Clean image
        if self.attempt_clean.get():
            self.preprocessed_img = self._apply_filters(self.preprocessed_img)

        # Set image size
        if self.preprocess_size:
            self.preprocessed_img = self._resize_image(self.preprocessed_img, self.preprocess_size)

        # Restore image channel so it can be properly analyzed by model
        if len(self.preprocessed_img.shape) == 2:
            self.preprocessed_img = cv2.cvtColor(self.preprocessed_img, cv2.COLOR_GRAY2BGR)

        # Preview preprocessed image
        self.root.after(0, lambda: self._display_image(self.preprocessed_img, self.lbl_preprocess_preview, 300))
        self.root.after(0, self._unlock_ui)
        self.root.after(0, lambda: self.btn_map.config(state="normal"))
        self.root.after(0, lambda: self.btn_dwnld_preprocess.config(state="normal"))


    def _map_intersections(self):
        start_time = time.perf_counter()

        self._update_status_from_thread("Loading machine learning model weights")

        self._update_status_from_thread("Running model")
        mapped_coords = []

        # Use multiple passes of sliding window inference to ensure windows overlap to capture all intersections
        step_size = self.sliding_window_size - self.sliding_window_overlap
        offset = int(step_size / 2)

        # Pass through whole image
        mapped_coords = self._sliding_window_pass(self.preprocessed_img, mapped_coords, self.sliding_window_size, self.sliding_window_overlap)
        mapped_coords = self._sliding_window_pass(self.preprocessed_img, mapped_coords, self.sliding_window_size, self.sliding_window_overlap, offset_x=offset, offset_y=offset)
        mapped_coords = self._reverse_sliding_window_pass(self.preprocessed_img, mapped_coords, self.sliding_window_size, self.sliding_window_overlap, offset_x=offset, offset_y=offset)

        # Create specific windows to infer
        center_y = self.preprocessed_img.shape[0] // 2
        center_x = self.preprocessed_img.shape[1] // 2
        grid_step = self.sliding_window_size // 2
        anchor_y = center_y - (self.sliding_window_size // 2)
        anchor_x = center_x - (self.sliding_window_size // 2)
        start_ys = [
            anchor_y - grid_step,
            anchor_y,
            anchor_y + grid_step
        ]
        start_xs = [
            anchor_x - grid_step,
            anchor_x,
            anchor_x + grid_step
        ]
        for start_y in start_ys:
            for start_x in start_xs:
                # Ensure coordinates with within image bounds
                safe_x = max(0, min(int(start_x), self.preprocessed_img.shape[1] - self.sliding_window_size))
                safe_y = max(0, min(int(start_y), self.preprocessed_img.shape[0] - self.sliding_window_size))
                mapped_coords = self._single_window_pass(self.preprocessed_img, mapped_coords, self.sliding_window_size, safe_x, safe_y)

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Inference runtime: {execution_time:.4f} seconds")

        # Generate image with intersections marked
        self.annotated_img = self.preprocessed_img.copy()
        for x, y in mapped_coords:
            cv2.circle(self.annotated_img, (int(x), int(y)), 5, (255, 0, 0), -1)

        # Preview annotated image
        self.root.after(0, lambda: self._display_image(self.annotated_img, self.lbl_intersections_preview, 300))
        self.root.after(0, lambda: self.lbl_intersections_count.config(text=f"Detected objects: {len(mapped_coords)}"))
        self.root.after(0, self._unlock_ui)
        self.root.after(0, lambda: self.btn_preview_viewer.config(state="normal"))
        self.root.after(0, lambda: self.btn_gen_print.config(state="normal"))


    """
    LOGIC UTILS
    """
    def _resize_image(self, img, max_size):
        # Resize maintaining aspect ratio
        h, w = img.shape[:2]
        scaling = min(max_size/w, max_size/h)
        if scaling != 1:
            if scaling < 1:
                # Better for shrinking
                interpolation_option = cv2.INTER_AREA
            else:
                # Better for upscaling
                interpolation_option = cv2.INTER_CUBIC
            new_h, new_w = int(h * scaling), int(w * scaling)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation_option)
        else:
            img_resized = img

        return img_resized


    def _apply_filters(self, img):
        # Blur to dissipate noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply Adaptive Thresholding to handle local lighting variations
        # Using a large block size (21) ensures grid lines are captured properly
        # Invert colors (black is treated as empty space)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )

        # Define structural elements (kernels) to target specific grid shapes
        # Use horizontal and vertical line kernels to ignore random, irregular shapes
        kernel_len = 7
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

        # Extract horizontal and vertical lines (removes dirt)
        hor_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        ver_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine the components back into a clean grid
        clean_grid = cv2.addWeighted(hor_lines, 1.0, ver_lines, 1.0, 0.0)

        # Invert back to black grid on white background
        preprocessed_img = cv2.bitwise_not(clean_grid)
        return preprocessed_img

    def _sliding_window_pass(self, img, detected_coords, sliding_window_size, sliding_window_overlap, move_start_x = 0, move_start_y = 0, offset_x = 0, offset_y = 0):
        # Offset image with border to offset window contents
        if offset_x > 0 or offset_y > 0:
            working_img = cv2.copyMakeBorder(img, offset_y, 0, offset_x, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            working_img = img
        
        img_height = working_img.shape[0]
        img_width = working_img.shape[1]
        step_size = sliding_window_size - sliding_window_overlap

        # Use sliding window technique to perform inference on snapshots of original image
        # (left -> right, top -> bottom)
        for y in range(move_start_y, img_height, step_size):
            for x in range(move_start_x, img_width, step_size):
                # Ensure window does not go out of bounds
                x_end = int(min(x + sliding_window_size, img_width))
                y_end = int(min(y + sliding_window_size, img_height))

                # Ensure window size is always constant even when it hits the boundary
                x_start = int(max(x_end - sliding_window_size, 0))
                y_start = int(max(y_end - sliding_window_size, 0))

                window = working_img[y_start:y_end, x_start:x_end]
                results = self.model.infer(window, confidence=self.confidence_threshold)[0]

                for prediction in results.predictions:
                    if not hasattr(prediction, "keypoints") or len(prediction.keypoints) == 0:
                        continue

                    intersection_pt = prediction.keypoints[0]
                    local_x = intersection_pt.x
                    local_y = intersection_pt.y
                    global_x = local_x + x_start - offset_x
                    global_y = local_y + y_start - offset_y
                    
                    # Ensure point is not duplicate due to overlap
                    is_duplicate = False
                    for existing_x, existing_y in detected_coords:
                        distance = np.sqrt((global_x - existing_x)**2 + (global_y - existing_y)**2)
                        if distance < self.intersection_separation:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        detected_coords.append((global_x, global_y))

        return detected_coords
    

    def _reverse_sliding_window_pass(self, img, detected_coords, sliding_window_size, sliding_window_overlap, move_start_x = 0, move_start_y = 0, offset_x = 0, offset_y = 0):
        # Offset image with border to offset window contents
        if offset_x > 0 or offset_y > 0:
            working_img = cv2.copyMakeBorder(img, 0, offset_y, 0, offset_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            working_img = img
        
        img_height = working_img.shape[0]
        img_width = working_img.shape[1]
        step_size = sliding_window_size - sliding_window_overlap

        # Use sliding window technique to perform inference on snapshots of original image
        # (right -> left, bottom -> top)
        for y in range(img_height - move_start_y, -1, -1 * step_size):
            for x in range(img_width - move_start_x, -1, -1 * step_size):
                # Ensure window does not go out of bounds
                x_end = int(max(x - sliding_window_size, 0))
                y_end = int(max(y - sliding_window_size, 0))

                # Ensure window size is always constant even when it hits the boundary
                x_start = int(min(x_end + sliding_window_size, img_width))
                y_start = int(min(y_end + sliding_window_size, img_height))

                window = working_img[y_end:y_start, x_end:x_start]
                results = self.model.infer(window, confidence=self.confidence_threshold)[0]

                for prediction in results.predictions:
                    if not hasattr(prediction, "keypoints") or len(prediction.keypoints) == 0:
                        continue

                    intersection_pt = prediction.keypoints[0]
                    local_x = intersection_pt.x
                    local_y = intersection_pt.y
                    global_x = local_x + x_end
                    global_y = local_y + y_end
                    
                    # Ensure point is not duplicate due to overlap
                    is_duplicate = False
                    for existing_x, existing_y in detected_coords:
                        distance = np.sqrt((global_x - existing_x)**2 + (global_y - existing_y)**2)
                        if distance < self.intersection_separation:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        detected_coords.append((global_x, global_y))

        return detected_coords
    
    
    def _single_window_pass(self, img, detected_coords, window_size, start_x, start_y):
        img_height = img.shape[0]
        img_width = img.shape[1]
    
        # Ensure window does not go out of bounds
        x_end = min(start_x + window_size, img_width)
        y_end = min(start_y + window_size, img_height)

        # Ensure window size is always constant even when it hits the boundary
        x_start = max(x_end - window_size, 0)
        y_start = max(y_end - window_size, 0)

        # Perform inference on a specific window of image
        window = img[y_start:y_end, x_start:x_end]
        results = self.model.infer(window, confidence=self.confidence_threshold)[0]

        for prediction in results.predictions:
            if not hasattr(prediction, "keypoints") or len(prediction.keypoints) == 0:
                continue

            intersection_pt = prediction.keypoints[0]
            local_x = intersection_pt.x
            local_y = intersection_pt.y
            global_x = local_x + x_start
            global_y = local_y + y_start
            
            # Ensure point is not duplicate due to overlap
            is_duplicate = False
            for existing_x, existing_y in detected_coords:
                distance = np.sqrt((global_x - existing_x)**2 + (global_y - existing_y)**2)
                if distance < self.intersection_separation:
                    is_duplicate = True
                    break

            if not is_duplicate:
                detected_coords.append((global_x, global_y))

        return detected_coords


if __name__ == "__main__":
    window = tk.Tk()
    app = ScaffoldPrintGUI(window)
    window.mainloop()