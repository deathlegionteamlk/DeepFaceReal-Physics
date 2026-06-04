"""
DeepFaceReal-Physics Windows EXE Builder v2.0.0
Powered By DeathLegionTeamLK
Builds standalone DeepFaceReal.exe with PyInstaller bundling all core modules.
"""
import os
import sys
import platform
import shutil
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.resolve()
APP_SCRIPT = PROJECT_DIR / "app.py"
API_SCRIPT = PROJECT_DIR / "api.py"
OUTPUT_DIR = PROJECT_DIR / "dist"
EXE_NAME = "DeepFaceReal"
CREDITS = "Powered By DeathLegionTeamLK"

REQUIRED_MODULES = [
    "core",
    "core/__init__.py",
    "core/face_swapper.py",
    "core/character_manager.py",
    "core/llm_character.py",
    "core/physics_engine.py",
    "core/background_engine.py",
    "core/webcam_pipeline.py",
    "core/lip_sync.py",
    "core/face_3d_engine.py",
    "core/talking_head.py",
    "core/eye_engine.py",
    "core/gesture_engine.py",
    "core/pipeline.py",
]

HIDDEN_IMPORTS = [
    "insightface",
    "insightface.model_zoo",
    "insightface.app",
    "insightface.app.FaceAnalysis",
    "insightface.model_zoo.retinaface",
    "insightface.model_zoo.arcface_onnx",
    "insightface.model_zoo.inswapper",
    "mediapipe",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.face_mesh",
    "mediapipe.python.solutions.holistic",
    "mediapipe.python.solutions.drawing_utils",
    "cv2",
    "numpy",
    "onnxruntime",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFilter",
    "PIL.ImageFont",
    "streamlit",
    "streamlit.web.bootstrap",
    "fastapi",
    "uvicorn",
    "websockets",
    "pydantic",
    "requests",
    "httpx",
    "scipy",
    "scipy.interpolate",
    "scipy.signal",
    "scipy.ndimage",
    "scipy.spatial",
    "skimage",
    "skimage.transform",
    "skimage.filters",
    "skimage.morphology",
    "qrcode",
    "audioop",
]

DATA_FILES = [
    ("models/", "models/"),
    ("static/", "static/"),
    ("profiles/", "profiles/"),
]

INSIGHTFACE_MODEL_DIRS = [
    os.path.expanduser("~/.insightface/models"),
    os.path.join(os.path.dirname(__import__("insightface").__file__), "models"),
]
for model_dir in INSIGHTFACE_MODEL_DIRS:
    if os.path.exists(model_dir):
        DATA_FILES.append((model_dir, "insightface/models"))

WAV2LIP_MODEL_PATHS = [
    PROJECT_DIR / "models" / "wav2lip.pth",
    PROJECT_DIR / "models" / "wav2lip_gan.pth",
]
for w_path in WAV2LIP_MODEL_PATHS:
    if w_path.exists():
        DATA_FILES.append((str(w_path), "models/"))


class ExeBuilder:
    def __init__(self):
        self.version = "2.0.0"
        self.output_dir = OUTPUT_DIR
        self.check_dependencies()

    def check_dependencies(self):
        print(f"[Build v{self.version}] Checking dependencies... {CREDITS}")
        try:
            import PyInstaller
            print(f"[Build] PyInstaller: {PyInstaller.__version__}")
        except ImportError:
            print("[Build] Installing PyInstaller...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "pyinstaller"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        for mod in ["cv2", "numpy", "mediapipe", "insightface", "streamlit", "fastapi", "onnxruntime"]:
            try:
                __import__(mod)
                print(f"[Build] {mod}: OK")
            except ImportError:
                print(f"[Build] WARNING: {mod} not found, EXE may not run")

    def build_exe(self, script_path: Path, name: str, console: bool = False):
        print(f"[Build] Building {name}.exe from {script_path.name}...")
        cmd = [
            "pyinstaller",
            "--noconfirm",
            "--clean",
            "--name", name,
            "--distpath", str(self.output_dir),
            "--workpath", str(PROJECT_DIR / "build"),
            "--specpath", str(PROJECT_DIR / "build"),
        ]

        if not console:
            cmd.append("--noconsole")
            cmd.append("--windowed")

        for imp in HIDDEN_IMPORTS:
            cmd.extend(["--hidden-import", imp])

        for src, dst in DATA_FILES:
            if os.path.exists(src):
                cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])

        cmd.extend([
            "--collect-all", "insightface",
            "--collect-all", "mediapipe",
            "--collect-all", "streamlit",
            "--collect-all", "onnxruntime",
            "--collect-data", "streamlit",
            "--collect-data", "mediapipe",
            "--collect-submodules", "PIL",
            "--collect-submodules", "scipy",
            "--collect-submodules", "skimage",
        ])

        cmd.append(str(script_path))

        print(f"[Build] Running: {' '.join(cmd[:6])}...")
        subprocess.check_call(cmd, cwd=str(PROJECT_DIR))
        print(f"[Build] {name}.exe built successfully!")

    def create_launcher(self):
        launcher_path = PROJECT_DIR / f"{EXE_NAME}_Launcher.bat"
        launcher_content = f"""@echo off
title DeepFaceReal-Physics v{self.version} - {CREDITS}
echo ======================================================
echo   DeepFaceReal-Physics v{self.version}
echo   Hyper-Realistic AI Avatar
echo   {CREDITS}
echo ======================================================
echo.
echo [1] Start Streamlit UI (Port 8080)
echo [2] Start API Server (Port 8081)
echo [3] Start Both
echo [4] Exit
echo.
set /p choice="Select option (1-4): "
if "%choice%"=="1" (
    start "" "{EXE_NAME}.exe"
) else if "%choice%"=="2" (
    start "" "{EXE_NAME}_API.exe"
) else if "%choice%"=="3" (
    start "" "{EXE_NAME}.exe"
    start "" "{EXE_NAME}_API.exe"
) else (
    exit
)
echo.
echo Services started. Close this window or press any key to exit.
pause
"""
        launcher_path.write_text(launcher_content)
        print(f"[Build] Launcher created: {launcher_path}")

    def create_install_script(self):
        install_ps1 = PROJECT_DIR / "install_deps.ps1"
        content = f"""# DeepFaceReal-Physics Dependency Installer
# {CREDITS}
Write-Host "Installing DeepFaceReal-Physics dependencies..." -ForegroundColor Green
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python opencv-contrib-python numpy
pip install mediapipe onnxruntime insightface streamlit fastapi uvicorn
pip install websockets pydantic requests httpx
pip install scipy scikit-image Pillow qrcode
pip install python-multipart python-dotenv
Write-Host "Dependencies installed!" -ForegroundColor Green
"""
        install_ps1.write_text(content)
        print(f"[Build] Install script created: {install_ps1}")

    def build_all(self):
        self.build_exe(APP_SCRIPT, EXE_NAME, console=False)
        self.build_exe(API_SCRIPT, f"{EXE_NAME}_API", console=True)
        self.create_launcher()
        self.create_install_script()
        print(f"""
[Build] COMPLETE! v{self.version}
[Build] Output: {self.output_dir}/
[Build] - {EXE_NAME}.exe (Streamlit UI)
[Build] - {EXE_NAME}_API.exe (API Server)
[Build] - {EXE_NAME}_Launcher.bat
[Build] - install_deps.ps1
[Build] {CREDITS}
""")


def main():
    print(f"DeepFaceReal-Physics Windows EXE Builder v2.0.0")
    print(f"{CREDITS}")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")

    if platform.system() == "Linux":
        print("[Build] Note: Building on Linux. Use --target-os windows for cross-compile or build on Windows.")

    builder = ExeBuilder()
    builder.build_all()


if __name__ == "__main__":
    main()