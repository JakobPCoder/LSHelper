import math
import os
import json
import tkinter as tk
from tkinter import filedialog, scrolledtext
from tkinter import ttk
import cv2
import numpy as np

# Global directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data")  # Adjust "data" to your local data folder name

def rewrite_json_file(relative_file_path):
    """
    Load the JSON file from the local data directory and rewrite it with proper indentation.
    
    :param relative_file_path: The path to the JSON file relative to LOCAL_DATA_DIR
    """
    abs_file_path = os.path.join(LOCAL_DATA_DIR, relative_file_path)
    print(abs_file_path)
    
    # Load JSON data from the file
    with open(abs_file_path, 'r') as f:
        data = json.load(f)
    
    # Rewrite the file with proper indentation (4 spaces)
    with open(abs_file_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Rewritten JSON file at: {abs_file_path}")

def load_config():
    """
    Load the configuration from 'config.json' located in the base directory.
    If the file does not exist, a default configuration is created and saved.

    The configuration stores:
      - video_folder: Path to the folder containing video files.
      - annotation_file: Path to the JSON file with annotations.
      - output_folder: Path where the output folder structure will be created.
      - frame_spacing: Integer value indicating the frame extraction spacing.

    :return: A dictionary with the configuration settings.
    """
    config_path = os.path.join(BASE_DIR, "config.json")
    default_config = {
        "video_folder": "",
        "annotation_file": "",
        "output_folder": "",
        "frame_spacing": 1
    }
    if not os.path.exists(config_path):
        save_config(default_config)
        return default_config
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return default_config


def save_config(config_data):
    """
    Save the provided configuration data to 'config.json' in the base directory.
    
    :param config_data: Dictionary containing configuration settings.
    """
    config_path = os.path.join(BASE_DIR, "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"Error saving config: {e}")



def get_unique_video_filenames(video_folder):
    """
    Recursively scan the given video folder and return a list of unique video base names.
    Only files with accepted video extensions are considered and the file extension is removed.

    Accepted extensions: .mp4, .avi, .mov, .mkv

    :param video_folder: The directory to scan for video files.
    :return: List of unique video base names (e.g., ['video1', 'video2']).
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    video_basenames = set()
    for root, dirs, files in os.walk(video_folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in video_extensions:
                base_name = os.path.splitext(file)[0]
                video_basenames.add(base_name)
    return list(video_basenames)


def get_tracks_by_video_from_annotations(annotation_file):
    """
    Parse the Label Studio annotation JSON file and extract a mapping from video base names 
    (without extension) to a list of unique track IDs. Each track ID represents a unique tracked whale.
    
    The JSON records are expected to have a structure where:
      - The video file reference is located at record["data"]["video"].
      - The annotation list is found in record["annotations"], and within each, each 
        result object in "result" contains the key "id" indicating the unique track ID.

    :param annotation_file: Path to the annotation JSON file.
    :return: Dictionary mapping video base names to a list of unique track IDs.
             Example: {'video1': ['PmmDPRyak4', 't-rvCGmGru'], 'video2': ['9o1X5enc0_']}
    """
    with open(annotation_file, "r") as f:
        records = json.load(f)
    tracks_by_video = {}
    for record in records:
        # Retrieve the video file path and then extract the base name (without extension)
        video_path = ""
        if "data" in record and isinstance(record["data"], dict):
            video_path = record["data"].get("video", "")
        if not video_path:
            continue
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        # Process the annotation entries, if any, to extract track IDs from the "result" list.
        if "annotations" in record:
            for ann in record["annotations"]:
                if "result" in ann:
                    for res in ann["result"]:
                        track_id = res.get("id")
                        if track_id:
                            tracks_by_video.setdefault(video_base, [])
                            if track_id not in tracks_by_video[video_base]:
                                tracks_by_video[video_base].append(track_id)
    return tracks_by_video



def process_video_annotations(video_base, video_record):
    """
    Process annotations for a single video and extract frame data with crop information.
    
    :param video_base: Base name of the video (without extension)
    :param video_record: The JSON record containing annotations for this video
    :return: Dictionary with video name and list of frames with crop information
    """
    # Create a structure to hold frames for this video
    video_frames = []
    
    # Process all annotations for this video
    if "annotations" in video_record:
        for annotation in video_record["annotations"]:
            if "result" in annotation:
                for result in annotation["result"]:
                    # Extract track ID
                    track_id = result.get("id", "")
                    
                    # Extract class name - use first label if available or "None"
                    class_name = "None"
                    if "value" in result and "labels" in result["value"]:
                        labels = result["value"]["labels"]
                        if labels and len(labels) > 0:
                            class_name = labels[0]
                    
                    # The coordinates are in the sequence array
                    if "value" in result and "sequence" in result["value"]:
                        sequence = result["value"]["sequence"]
                        
                        # Process each frame in the sequence
                        for frame_data in sequence:
                            frame_number = int(frame_data.get("frame", 0))
                            
                            # Extract coordinates directly from frame_data
                            x = frame_data.get("x", 0)
                            y = frame_data.get("y", 0)
                            w = frame_data.get("width", 0)  # Using w instead of width
                            h = frame_data.get("height", 0)  # Using h instead of height
                            r = frame_data.get("rotation", 0)  # Added rotation as r
                            
                            # Find if we already have an entry for this frame
                            frame_entry = None
                            for frame in video_frames:
                                if frame["frame"] == frame_number:
                                    frame_entry = frame
                                    break
                            
                            # If no entry exists for this frame, create one
                            if not frame_entry:
                                frame_entry = {
                                    "frame": frame_number,
                                    "crops": []
                                }
                                video_frames.append(frame_entry)
                            
                            # Add the crop information to this frame
                            frame_entry["crops"].append({
                                "track": track_id,
                                "class": class_name,
                                "x": x,
                                "y": y,
                                "w": w,  # Renamed from width to w
                                "h": h,  # Renamed from height to h
                                "r": r   # Added rotation as r
                            })
    
    # Sort frames by frame number
    video_frames.sort(key=lambda x: x["frame"])
    
    # Return structured data
    return {
        "video": video_base,
        "frames": video_frames
    }

def get_resolution_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    return width, height

def process_annotations_and_create_folders(video_folder, annotation_file, output_folder, main_class, conditional_classes, logger=None):
    """
    Create a nested folder structure in the output folder based on the annotation JSON, generate interpolated
    crop positions for each video that exists in both the video folder and the JSON, and save the combined
    interpolation output as a list of objects to a 'crops.json' file in the base directory.
    
    Procedure:
      - Recursively scan the video folder to obtain a set of unique video base names.
      - Parse the annotation JSON to extract a mapping of video base names to unique track IDs.
      - Load the full annotation JSON content.
      - For each video referenced in the JSON:
            * Create an output folder (named after the video base name) and corresponding track subfolders.
            * If the video exists in the folder:
                - Generate interpolated crop data by calling interpolate_tracks_for_video().
                - Append the resulting object to an accumulating list.
            * Otherwise, for each track in that video, issue a warning.
            * Output a progress message "Processed x/y videos" to both the console and the UI log.
      - For each video present in the folder but missing from the JSON, issue a warning.
      - Save the accumulated list of video objects (each containing the video name and its tracks)
        to a file named 'crops.json' in the base directory. If the file exists, it is overwritten.
    
    Debug messages are printed only to the console.
    Warning and progress messages are printed to both the console and to the UI log (if provided).
    
    :param video_folder: Directory containing the video files.
    :param annotation_file: Annotation JSON file produced by Label Studio.
    :param output_folder: Directory where the nested folder structure will be generated.
    :param logger: Optional logging function (accepting a string) to output warning and progress messages to the UI.
    """
    def debug(msg):
        print(msg)

    def warn(msg):
        print(msg)
        if logger:
            logger(msg)

    def progress(msg):
        print(msg)
        if logger:
            logger(msg)

    folder_videos = set(get_unique_video_filenames(video_folder))
    annotation_tracks = get_tracks_by_video_from_annotations(annotation_file)

    with open(annotation_file, "r") as f:
        json_content = json.load(f)

        debug(f"DEBUG: Found {len(folder_videos)} unique video file(s) in the video folder.")
        debug(f"DEBUG: Found {len(annotation_tracks)} unique video file(s) in the annotations.")

        total_videos = len(annotation_tracks)
        count = 0

        # Accumulate output as a list of plain objects.
        all_crops = []
        valid_videos = []

        for video_base, track_ids in annotation_tracks.items():
            count += 1
            debug(f"DEBUG: Creating folder structure for video '{video_base}' with {len(track_ids)} track(s) in annotations.")
            video_output_dir = os.path.join(output_folder, video_base)
            os.makedirs(video_output_dir, exist_ok=True)
            
            for track_id in track_ids:
                track_dir = os.path.join(video_output_dir, track_id)
                os.makedirs(track_dir, exist_ok=True)
            
            if video_base not in folder_videos:
                for track_id in track_ids:
                    warn(f"WARNING: Video '{video_base}' does not exist in the selected folder, "
                        f"but track '{track_id}' was found in annotations.")
            else:
                valid_videos.append(video_base)

        for video_base in valid_videos:
            if video_base not in annotation_tracks:
                warn(f"WARNING: Video '{video_base}' exists in the folder but has no annotations in the JSON.")
                valid_videos.remove(video_base)

        for video_base in valid_videos:
            debug(f"DEBUG: Processing video '{video_base}' with {len(annotation_tracks[video_base])} track(s) in annotations.")
            
            # Find the video record in JSON content
            video_record = None
            for record in json_content:
                if "data" in record and "video" in record["data"] and video_base in record["data"]["video"]:
                    video_record = record
                    break
            
            if not video_record:
                warn(f"WARNING: Could not find video record for '{video_base}' in annotation file.")
                continue
            
            # Use the new wrapper function to process this video's annotations
            video_data = process_video_annotations(video_base, video_record)
            all_crops.append(video_data)
            
            debug(f"DEBUG: Processed {len(video_data['frames'])} frames for video '{video_base}'")
            progress_msg = f"Progress: Processed {count}/{total_videos} videos."
            progress(progress_msg)

        # Save the accumulated crops data
        save_crops_data(all_crops, "all_crops.json")

        try:
            filtered_crops = filter_all_crops(all_crops, main_class, conditional_classes)
        except Exception as e:
            print(f"Error filtering crops: {e}")
            filtered_crops = []

        save_crops_data(filtered_crops, "filtered_crops.json")

        for video in filtered_crops:
            video_name = video["video"]
            video_path = os.path.join(video_folder, f"{video_name}.mp4")

            # cv2 video image loop
            cap = cv2.VideoCapture(video_path)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                for frame_data in video["frames"]:
                    if frame_data["frame"] == frame_number:
                        crops = get_crops_from_frames(image, frame_data, width, height)
                        for crop_image, crop in crops:
                            save_path = os.path.join(output_folder, video_name, crop["track"], f"{frame_number}.png")
                            print(f"Saving crop to {save_path}")
                            cv2.imwrite(save_path, crop_image)
                        continue

def rotate_point(px, py, cx, cy, angle):
    """Rotate a point (px, py) around (cx, cy) by angle in degrees."""
    rad = math.radians(-(angle))  # Negative for counter-clockwise rotation
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    new_x = dx * cos_a - dy * sin_a + cx
    new_y = dx * sin_a + dy * cos_a + cy
    return new_x, new_y   

def rotate_point_int_list(px, py, cx, cy, angle):
    """Rotate a point (px, py) around (cx, cy) by angle in degrees."""
    rad = math.radians(-(angle))  # Negative for counter-clockwise rotation
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    new_x = dx * cos_a - dy * sin_a + cx
    new_y = dx * sin_a + dy * cos_a + cy
    return [int(new_x), int(new_y)]  

def is_point_in_rotated_rect(x: float, y: float, rect: dict) -> bool:
    """
    Check if a point (x, y) is inside a rotated bounding box.
    The box is defined as a dictionary with keys 'x', 'y', 'w', 'h', 'r'.
    """

    
    # rotate point to align with the coordinate system of the rect
    x, y = rotate_point(x, y, rect['x'], rect['y'], rect['r'])
    # check if the point is inside the rect
    return (x >= rect['x']) and (x <= rect['x'] + rect['w']) and (y >= rect['y']) and (y <= rect['y'] + rect['h'])


def filter_crop_conditions(frame_data: dict, main_class: str, conditional_classes: list[str]):
    """
    Returns a copy of frame_data with only the crops that match the main class and if conditional classes are provided,
    a crop whose center is inside the bounding box of a crop from each of the conditional classes.
    """
    filtered_crops = []
    for crop in frame_data["crops"]:
        if crop["class"] == main_class:
            if not conditional_classes:
                filtered_crops.append(crop)
            else:
                conditions_met = []
                for condition_crop in frame_data["crops"]:
                    if condition_crop["class"] in conditional_classes:
                        if is_point_in_rotated_rect(condition_crop["x"], condition_crop["y"], crop):
                            conditions_met.append(condition_crop["class"])
                
                # Check if all conditional classes are represented at least once
                if all(cond_class in conditions_met for cond_class in conditional_classes):
                    filtered_crops.append(crop)


    filtered_frame = frame_data.copy()
    filtered_frame["crops"] = filtered_crops
    return filtered_frame
                    

def filter_all_crops(all_crops, main_class, conditional_classes):
    filtered_crops = []
    for video in all_crops:
        filtered_video = { "video": video["video"], "frames": [] }

        for frame in video["frames"]:
            filtered_frame = filter_crop_conditions(frame, main_class, conditional_classes)
            if filtered_frame["crops"]:
                filtered_video["frames"].append(filtered_frame)
        filtered_crops.append(filtered_video)
            
    return filtered_crops


def get_crop_from_frame(image: np.ndarray, crop: dict, width: int, height: int):
    x = int(crop["x"] * width / 100)
    y = int(crop["y"] * height / 100)
    w = int(crop["w"] * width / 100)
    h = int(crop["h"] * height / 100)
    r = -crop.get("r", 0)

    # get the four corners of the crop. x and y top left corner of crop
    corners = [
        (x, y),
        (x + w, y),
        (x, y + h),
        (x + w, y + h)
    ]
    corners_np = np.array(corners, dtype=np.float32)

    rotated_corners = []
    for corner in corners:
        rotated_corners.append(rotate_point_int_list(corner[0], corner[1], x, y, r))

    # Convert rotated_corners to a NumPy array
    rotated_corners_np = np.array(rotated_corners, dtype=np.float32)

    target_corners = np.array([
        [0, 0],
        [w, 0],
        [0, h],
        [w, h]
    ], dtype=np.float32)

    # target image 
    target_np = np.zeros((h, w, image.shape[-1]), dtype=np.uint8)
    M = cv2.getPerspectiveTransform(rotated_corners_np, target_corners)

    cv2.warpPerspective(image, M, (w, h), target_np, flags=cv2.INTER_CUBIC)

    return target_np

def get_crops_from_frames(image: np.ndarray, frame: dict, width: int, height: int):
    image_crops = []
    for crop in frame["crops"]:
        image_crops.append(np.array(get_crop_from_frame(image, crop, width, height)))

    # return image_crops alongside their framecrops
    return zip(image_crops, frame["crops"])


def get_class_list_from_annotation_file(annotation_file):
    """
    Parse the Label Studio annotation JSON file and extract a list of unique class names.
    """
    with open(annotation_file, "r") as f:
        records = json.load(f)

    # New logic to extract unique class names
    class_names = set()  # Use a set to store unique class names

    # Iterate over each record in the list
    for record in records:  
        if "annotations" in record:  
            for ann in record["annotations"]:
                if "result" in ann:
                    for result in ann["result"]:
                        if "value" in result and "labels" in result["value"]:
                            # Update class names with all entries in the labels list
                            class_names.update(result["value"]["labels"])  # Add class names to the set

    return list(class_names)  # Convert the set back to a list

def save_crops_data(crops_data, filename: str = "crops.json"):
    """
    Save the provided crops interpolation data (a list of video objects) to 'crops.json' 
    in the base directory. If the file already exists, it will be overwritten.
    
    :param crops_data: List of video objects, each with keys 'video' and 'tracks' containing interpolated crop data.
    :param filename: The name of the file to save the crops data to.
    """
    crops_path = os.path.join(BASE_DIR, filename)
    try:
        with open(crops_path, "w") as f:
            json.dump(crops_data, f, indent=4)
    except Exception as e:
        print(f"Error saving crops data: {e}")


def run_ui():
    """
    Launch a Windows UI to select a video folder, annotation JSON file, output folder,
    and frame spacing value. The UI allows saving/loading the configuration and starts the 
    process to create a nested folder structure upon clicking the "Run" button.
    
    The UI consists of:
      - Input fields for video folder, annotation file, and output folder.
      - A slider and text entry for frame spacing (range 1–128).
      - A console-like text area for real-time log messages.
      - Buttons: Save, Load, and Run in one row.
    """
    config_data = load_config()


    root = tk.Tk()
    root.title("LabelStudioHelper - Frame Crops from Videorects")
    root.geometry("900x500")

    # Layout: Two main frames (left for controls, right for console log)
    left_frame = ttk.Frame(root, padding="10")
    left_frame.grid(row=0, column=0, sticky="nsew")
    right_frame = ttk.Frame(root, padding="10")
    right_frame.grid(row=0, column=1, sticky="nsew")

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=2)
    root.rowconfigure(0, weight=1)


    
    # Variables to track selections
    selected_class_var = tk.StringVar()
    selected_classes_list = []

    def update_class_ui_elements(class_list):
        """
        Update the dropdown and scrollable list with the provided class list.

        :param class_list: List of class names to populate the UI elements.
        """
        # Update the dropdown
        class_dropdown['values'] = class_list
        if class_list:
            class_dropdown.current(0)  # Select the first item by default
        else:
            class_dropdown.set('')  # Set to blank if no classes are available
            selected_class_var.set('')  # Clear the selection variable

        # Update the scrollable list
        class_listbox.delete(0, tk.END)
        for class_name in class_list:
            class_listbox.insert(tk.END, class_name)
        if not class_list:
            selected_classes_list.clear()  # Clear the selection list

    def on_dropdown_select(event):
        """Update the selected class variable when the dropdown selection changes."""
        selected_class_var.set(class_dropdown.get())

    def on_listbox_select(event):
        """Update the selected classes list when the listbox selection changes."""
        selected_classes_list.clear()
        selected_classes_list.extend([class_listbox.get(i) for i in class_listbox.curselection()])

    def check_and_log_class_list(annotation_file, logger=None):
        """
        Check if the annotation file exists and log the class list if it does.
        Log a warning if the file does not exist.

        :param annotation_file: Path to the annotation JSON file.
        :param logger: Function to log messages.
        """
        if annotation_file and os.path.exists(annotation_file):
            class_list = get_class_list_from_annotation_file(annotation_file)
            # if logger:
            #     logger(f"Class List: {class_list}")
            # Update the UI elements with the new class list
            update_class_ui_elements(class_list)
        else:
            if logger:
                logger(f"Warning: Annotation file '{annotation_file}' does not exist.")
            # Clear the UI elements if the file does not exist
            update_class_ui_elements([])
            class_dropdown.set('')  # Set dropdown to blank selection
            

    # --- Video Folder Selection ---
    video_label = ttk.Label(left_frame, text="Video Folder:")
    video_label.grid(row=0, column=0, sticky="w")
    video_folder_var = tk.StringVar(value=config_data.get("video_folder", ""))
    video_entry = ttk.Entry(left_frame, textvariable=video_folder_var, width=50)
    video_entry.grid(row=1, column=0, sticky="we", padx=(0, 5))

    def browse_video_folder():
        folder = filedialog.askdirectory(initialdir=BASE_DIR, title="Select Video Folder")
        if folder:
            video_folder_var.set(folder)
            config_data["video_folder"] = folder
            save_config(config_data)

    video_browse_button = ttk.Button(left_frame, text="Browse", command=browse_video_folder)
    video_browse_button.grid(row=1, column=1, padx=(5, 0))

    # --- Annotation File Selection ---
    annotation_label = ttk.Label(left_frame, text="Annotation File:")
    annotation_label.grid(row=2, column=0, sticky="w", pady=(10, 0))
    annotation_file_var = tk.StringVar(value=config_data.get("annotation_file", ""))
    annotation_entry = ttk.Entry(left_frame, textvariable=annotation_file_var, width=50)
    annotation_entry.grid(row=3, column=0, sticky="we", padx=(0, 5))

    def browse_annotation_file():
        file = filedialog.askopenfilename(
            initialdir=BASE_DIR, title="Select Annotation File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file:
            annotation_file_var.set(file)
            config_data["annotation_file"] = file
            save_config(config_data)
            # Check and log class list after selecting a new file
            check_and_log_class_list(file, log)

    annotation_browse_button = ttk.Button(left_frame, text="Browse", command=browse_annotation_file)
    annotation_browse_button.grid(row=3, column=1, padx=(5, 0))

    # --- Output Folder Selection ---
    output_label = ttk.Label(left_frame, text="Output Folder:")
    output_label.grid(row=4, column=0, sticky="w", pady=(10, 0))
    output_folder_var = tk.StringVar(value=config_data.get("output_folder", ""))
    output_entry = ttk.Entry(left_frame, textvariable=output_folder_var, width=50)
    output_entry.grid(row=5, column=0, sticky="we", padx=(0, 5))

    def browse_output_folder():
        folder = filedialog.askdirectory(initialdir=BASE_DIR, title="Select Output Folder")
        if folder:
            output_folder_var.set(folder)
            config_data["output_folder"] = folder
            save_config(config_data)

    output_browse_button = ttk.Button(left_frame, text="Browse", command=browse_output_folder)
    output_browse_button.grid(row=5, column=1, padx=(5, 0))

    # --- Frame Spacing Selection ---
    spacing_label = ttk.Label(left_frame, text="Frame Spacing (1-128):")
    spacing_label.grid(row=6, column=0, sticky="w", pady=(10, 0))
    frame_spacing_var = tk.IntVar(value=config_data.get("frame_spacing", 1))

    spacing_entry = ttk.Entry(left_frame, width=5)
    spacing_entry.grid(row=7, column=1, pady=(14, 0), padx=(0, 30))
    spacing_entry.insert(0, str(frame_spacing_var.get()))

    def update_spacing_from_entry(event=None):
        try:
            value = int(spacing_entry.get())
            if value < 1:
                value = 1
            elif value > 128:
                value = 128
        except ValueError:
            value = 1
        frame_spacing_var.set(value)
        spacing_entry.delete(0, "end")
        spacing_entry.insert(0, str(value))

    spacing_entry.bind("<Return>", update_spacing_from_entry)
    spacing_entry.bind("<FocusOut>", update_spacing_from_entry)

    def update_spacing_slider(val):
        spacing_entry.delete(0, "end")
        spacing_entry.insert(0, str(int(float(val))))

    spacing_scale = tk.Scale(
        left_frame, from_=1, to=128, orient="horizontal",
        variable=frame_spacing_var, resolution=1, command=update_spacing_slider
    )
    spacing_scale.grid(row=7, column=0, sticky="we", padx=(0, 5))

    # --- Class Selection UI Elements ---
    class_label = ttk.Label(left_frame, text="Select Class:")
    class_label.grid(row=8, column=0, sticky="w")

    class_label = ttk.Label(left_frame, text="Condition:")
    class_label.grid(row=8, column=1, sticky="w")

    # Dropdown for single class selection
    class_dropdown = ttk.Combobox(left_frame, state="readonly", textvariable=selected_class_var)
    class_dropdown.grid(row=9, column=0, sticky="nwe")
    class_dropdown.bind("<<ComboboxSelected>>", on_dropdown_select)

    # Scrollable list for multiple class selection
    class_listbox = tk.Listbox(left_frame, selectmode=tk.MULTIPLE, height=5)
    class_listbox.grid(row=9, column=1, sticky="we")
    class_listbox.bind("<<ListboxSelect>>", on_listbox_select)

    # Add a scrollbar to the listbox
    scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=class_listbox.yview)
    scrollbar.grid(row=9, column=2, sticky="ns", pady=(0, 32))
    class_listbox.config(yscrollcommand=scrollbar.set)

    # --- Buttons: Save, Load, and Run ---
    button_frame = ttk.Frame(left_frame)
    button_frame.grid(row=10, column=0, columnspan=2, pady=(20, 0))


    def on_save():
        config_data["video_folder"] = video_folder_var.get()
        config_data["annotation_file"] = annotation_file_var.get()
        config_data["output_folder"] = output_folder_var.get()
        config_data["frame_spacing"] = frame_spacing_var.get()
        config_data["selected_class"] = selected_class_var.get()
        selection = class_listbox.curselection()
        print(selection)
        config_data["selected_classes_selected"] = [class_listbox.get(i) for i in selection]
        save_config(config_data)
        log("==============================")
        log("Configuration saved.")

    def on_load():
        new_config = load_config()
        video_folder_var.set(new_config.get("video_folder", ""))
        annotation_file_var.set(new_config.get("annotation_file", ""))
        output_folder_var.set(new_config.get("output_folder", ""))
        frame_spacing_var.set(new_config.get("frame_spacing", 1))
        spacing_entry.delete(0, "end")
        spacing_entry.insert(0, str(frame_spacing_var.get()))


     
        check_and_log_class_list(annotation_file_var.get(), log)
        selected_class_var.set(new_config.get("selected_class", ""))
        # Restore listbox selections
        class_listbox.selection_clear(0, tk.END)
        selected_classes = new_config.get("selected_classes_selected", [])
        all_items = class_listbox.get(0, tk.END)
        selected_indices = [all_items.index(item) for item in selected_classes if item in all_items]
        for index in selected_indices:
            class_listbox.selection_set(index)
        
        log("==============================")
        log("Configuration loaded.")


    def on_run():
        video_folder = video_folder_var.get()
        annotation_file = annotation_file_var.get()
        output_folder = output_folder_var.get()
        frame_spacing = frame_spacing_var.get()
        selected_class = selected_class_var.get()
        selected_classes_selected = [class_listbox.get(i) for i in class_listbox.curselection()]


        log("==============================")
        log("Run button clicked.")
        log(f"Video Folder: {video_folder}")
        log(f"Annotation File: {annotation_file}")
        log(f"Output Folder: {output_folder}")
        log(f"Frame Spacing: {frame_spacing}")
        log(f"Target Class: {selected_class}")
        log(f"Conditional Classes: {selected_classes_selected}")


        # class_list = get_class_list_from_annotation_file(annotation_file)
        # log(f"Class List: {class_list}")


        # Process the annotations and create the folder structure, passing the UI log function
        # so warnings are output in the UI, not debug messages.

        process_annotations_and_create_folders(video_folder, annotation_file, output_folder, selected_class, selected_classes_selected, logger=log)


    save_button = ttk.Button(button_frame, text="Save", command=on_save)
    save_button.grid(row=0, column=0, padx=(0, 10))
    load_button = ttk.Button(button_frame, text="Load", command=on_load)
    load_button.grid(row=0, column=1, padx=(0, 10))
    run_button = ttk.Button(button_frame, text="Run", command=on_run)
    run_button.grid(row=0, column=2, padx=(0, 10))

    # --- Console-like Log Output ---
    console_text = scrolledtext.ScrolledText(right_frame, wrap="word", state="disabled", width=60)
    console_text.pack(expand=True, fill="both")

    def log(message):
        console_text['state'] = 'normal'
        console_text.insert("end", message + "\n")
        console_text.see("end")
        console_text['state'] = 'disabled'


    log("Application started.")
    # Check and log class list on application start
    check_and_log_class_list(annotation_file_var.get(), log)
    on_load()
    root.mainloop()


if __name__ == "__main__":
    run_ui()
