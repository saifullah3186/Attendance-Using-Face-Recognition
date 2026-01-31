# Smart Attendance System - Configuration

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Capture
DEFAULT_PHOTOS_PER_STUDENT = 10
CAPTURE_DELAY = 0.5

# Recognition
FACE_DISTANCE_THRESHOLD = 0.6
FACE_DETECTION_UPSAMPLE = 1

# Attendance
NO_FACE_TIMEOUT = 20
ALLOW_DUPLICATE_MARKING = False
ATTENDANCE_FILE = 'attendance.xlsx'

# Paths
DATASET_PATH = 'dataset'
MODEL_FILE = 'trained_model.npz'

# Server
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Performance
FRAME_SKIP = 3
FRAME_SCALE = 0.25

# UI
APP_TITLE = "Smart Attendance System"
APP_SUBTITLE = "Face Recognition Based Attendance Management"
STATUS_REFRESH_INTERVAL = 10000

# Advanced
VERBOSE_LOGGING = True
SAVE_DEBUG_FRAMES = False
MAX_FILE_SIZE = 10
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
