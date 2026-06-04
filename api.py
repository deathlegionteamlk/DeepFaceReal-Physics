"""
DeepFaceReal-Physics FastAPI v2.0.0
Powered By DeathLegionTeamLK
Endpoints: /generate/talking-head, /animate/face, WebSocket /ws/realtime, /config/render
"""
import os
import sys
import cv2
import numpy as np
import json
import asyncio
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.face_swapper import get_swapper
from core.character_manager import get_character_manager
from core.llm_character import create_llm_character
from core.physics_engine import get_tracker
from core.webcam_pipeline import get_pipeline
from core.face_3d_engine import get_face_3d_engine
from core.talking_head import get_talking_head
from core.lip_sync import create_lip_sync
from core.eye_engine import get_eye_engine
from core.gesture_engine import get_gesture_engine
from core.pipeline import get_realtime_pipeline

CREDITS = "Powered By DeathLegionTeamLK"

swapper = get_swapper(det_size=(320, 320))
char_manager = get_character_manager()
llm = create_llm_character(os.environ.get("OPENROUTER_API_KEY", ""))
tracker = get_tracker()
pipeline = get_pipeline()
face_3d = get_face_3d_engine()
talking_head = get_talking_head()
lip_sync = create_lip_sync()
eye_engine = get_eye_engine()
gesture_engine = get_gesture_engine()
rt_pipeline = get_realtime_pipeline()

app = FastAPI(
    title="DeepFaceReal-Physics API v2",
    description=f"Hyper-Realistic AI Avatar engine. {CREDITS}",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TalkingHeadRequest(BaseModel):
    audio_b64: str = Field(..., description="Base64-encoded WAV audio")
    face_b64: str = Field(..., description="Base64-encoded source face image")
    fps: int = 20
    duration_seconds: Optional[float] = None


class AnimateFaceRequest(BaseModel):
    face_b64: str = Field(..., description="Base64-encoded face image")
    expression_coeffs: Optional[List[float]] = None
    head_pose: Optional[Dict[str, float]] = None
    return_mesh: bool = False


class RenderConfigRequest(BaseModel):
    engine: str = Field(..., description="Engine to configure: face3d|talking_head|lip_sync|eye|gesture|pipeline")
    config: Dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str
    character_name: Optional[str] = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    character: str
    timestamp: str
    credits: str = CREDITS


class CharacterCreate(BaseModel):
    name: str
    system_prompt: Optional[str] = ""
    voice_enabled: bool = False
    swap_enabled: bool = True
    enhance_quality: str = "medium"
    blend_enabled: bool = True


class SwapRequest(BaseModel):
    source_face_b64: Optional[str] = None
    target_b64: Optional[str] = None
    blend: bool = True
    enhance: bool = True
    enhance_quality: str = "medium"


class PhysicsConfigRequest(BaseModel):
    mass: Optional[float] = None
    damping: Optional[float] = None
    spring_stiffness: Optional[float] = None
    friction: Optional[float] = None
    gravity: Optional[float] = None
    smoothing_alpha: Optional[float] = None
    intensity: Optional[float] = None
    enabled: Optional[bool] = None


class CameraSourceRequest(BaseModel):
    source: str = "usb"
    url: Optional[str] = ""


def decode_b64_image(b64_str: str) -> np.ndarray:
    if ',' in b64_str:
        b64_str = b64_str.split(',')[1]
    img_bytes = base64.b64decode(b64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    return img


def encode_b64_image(img: np.ndarray, quality: int = 85) -> str:
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/")
async def root():
    return {
        "message": "DeepFaceReal-Physics API v2 - Hyper-Realistic AI Avatar",
        "version": "2.0.0",
        "docs": "/docs",
        "credits": CREDITS
    }


@app.get("/status")
async def get_status():
    active_char = char_manager.get_active_character()
    return {
        "status": "running",
        "version": "2.0.0",
        "swapper_loaded": True,
        "face3d_loaded": True,
        "talking_head_loaded": True,
        "eye_engine_loaded": True,
        "gesture_engine_loaded": True,
        "lip_sync_loaded": True,
        "characters_available": len(char_manager.list_characters()),
        "active_character": active_char.name if active_char else None,
        "llm_configured": True,
        "has_api_key": bool(os.environ.get("OPENROUTER_API_KEY", "")),
        "pipeline_fps": rt_pipeline.get_metrics().fps,
        "credits": CREDITS
    }


@app.post("/swap")
async def swap_face(request: SwapRequest):
    try:
        if request.source_face_b64:
            source_img = decode_b64_image(request.source_face_b64)
            source_face = swapper.get_source_face(source_img)
            if source_face is None:
                raise HTTPException(status_code=400, detail="No face found in source image")
        else:
            active_char = char_manager.get_active_character()
            if active_char and active_char.source_face is not None:
                source_face = active_char.source_face
            else:
                raise HTTPException(status_code=400, detail="No source face available")
        if request.target_b64:
            target_img = decode_b64_image(request.target_b64)
        else:
            raise HTTPException(status_code=400, detail="No target image provided")
        result = swapper.process_frame(
            target_img, source_face,
            blend=request.blend, enhance=request.enhance,
            enhance_quality=request.enhance_quality
        )
        result_b64 = encode_b64_image(result)
        return JSONResponse({
            "success": True,
            "image": f"data:image/jpeg;base64,{result_b64}",
            "width": result.shape[1],
            "height": result.shape[0],
            "credits": CREDITS
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/talking-head")
async def generate_talking_head(request: TalkingHeadRequest):
    try:
        audio_data = base64.b64decode(request.audio_b64)
        face_img = decode_b64_image(request.face_b64)
        result = talking_head.process_full_audio(audio_data, face_img, fps=request.fps)
        frames_b64 = [encode_b64_image(f) for f in result["frames"]]
        return JSONResponse({
            "success": True,
            "frames": [f"data:image/jpeg;base64,{f}" for f in frames_b64],
            "total_frames": len(result["frames"]),
            "head_pose_sequence": result.get("head_poses", []),
            "expression_sequence": result.get("expressions", []),
            "fps": request.fps,
            "credits": CREDITS
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Talking head generation failed: {str(e)}")


@app.post("/animate/face")
async def animate_face(request: AnimateFaceRequest):
    try:
        face_img = decode_b64_image(request.face_b64)
        if request.expression_coeffs:
            face_3d.apply_expression_coefficients(request.expression_coeffs)
        if request.head_pose:
            face_3d.set_head_pose(request.head_pose)
        mesh_data = face_3d.process_frame(face_img)
        output = face_3d.render_mesh()
        result_b64 = encode_b64_image(output)
        response = {
            "success": True,
            "face": f"data:image/jpeg;base64,{result_b64}",
            "mesh_points": mesh_data["mesh"][:50].tolist() if mesh_data.get("mesh") is not None else [],
            "head_pose_6dof": mesh_data.get("pose", {}),
            "expression_coefficients": mesh_data.get("expression_coeffs", []),
            "credits": CREDITS
        }
        if request.return_mesh and mesh_data.get("mesh") is not None:
            mesh_b64 = encode_b64_image(mesh_data.get("mesh_viz", output))
            response["mesh_visualization"] = f"data:image/jpeg;base64,{mesh_b64}"
        return JSONResponse(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face animation failed: {str(e)}")


@app.post("/config/render")
async def config_render(request: RenderConfigRequest):
    engine_map = {
        "face3d": face_3d.config,
        "talking_head": talking_head.config,
        "lip_sync": lip_sync.config,
        "eye": eye_engine.config,
        "gesture": gesture_engine.config,
    }
    if request.engine == "pipeline":
        for k, v in request.config.items():
            setattr(rt_pipeline, k, v)
        return {"success": True, "engine": "pipeline", "updated": request.config, "credits": CREDITS}
    cfg = engine_map.get(request.engine)
    if cfg is None:
        raise HTTPException(status_code=400, detail=f"Unknown engine '{request.engine}'")
    for k, v in request.config.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return {"success": True, "engine": request.engine, "updated": request.config, "credits": CREDITS}


@app.post("/chat")
async def chat(request: ChatRequest):
    if request.character_name:
        char = char_manager.get_character(request.character_name)
        if char:
            llm.set_character(char.name, char.system_prompt)
    response = llm.send_message(request.message)
    active_char = char_manager.get_active_character()
    char_name = active_char.name if active_char else "Assistant"
    return ChatResponse(
        response=response,
        character=char_name,
        timestamp=datetime.now().isoformat(),
        credits=CREDITS
    )


@app.post("/character")
async def create_character_endpoint(
    name: str = Form(...),
    system_prompt: str = Form(""),
    voice_enabled: bool = Form(False),
    swap_enabled: bool = Form(True),
    enhance_quality: str = Form("medium"),
    blend_enabled: bool = Form(True),
    photo: Optional[UploadFile] = File(None)
):
    char = char_manager.create_character(
        name=name, system_prompt=system_prompt, voice_enabled=voice_enabled
    )
    char.swap_enabled = swap_enabled
    char.enhance_quality = enhance_quality
    char.blend_enabled = blend_enabled
    has_photo = False
    if photo and photo.filename:
        photo_dir = os.path.join(os.path.dirname(__file__), 'profiles')
        os.makedirs(photo_dir, exist_ok=True)
        photo_path = os.path.join(photo_dir, f"{name.replace(' ', '_')}_photo.jpg")
        contents = await photo.read()
        with open(photo_path, 'wb') as f:
            f.write(contents)
        char.photo_path = photo_path
        has_photo = True
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        face_emb = swapper.extract_face_embedding(img)
        source_face = swapper.get_source_face(img)
        char_manager.set_character_face(name, photo_path, face_emb, source_face)
    char_manager.save_character(name)
    return {
        "name": char.name, "system_prompt": char.system_prompt,
        "voice_enabled": char.voice_enabled, "swap_enabled": char.swap_enabled,
        "enhance_quality": char.enhance_quality, "created_at": char.created_at,
        "has_photo": has_photo, "has_face_data": char.face_embedding is not None,
        "credits": CREDITS
    }


@app.get("/characters")
async def list_characters():
    summaries = char_manager.get_character_summaries()
    return {"characters": summaries, "active": char_manager.active_character_name, "credits": CREDITS}


@app.get("/characters/{name}")
async def get_character(name: str):
    char = char_manager.get_character(name)
    if char is None:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
    return {
        "name": char.name, "system_prompt": char.system_prompt,
        "voice_enabled": char.voice_enabled, "swap_enabled": char.swap_enabled,
        "enhance_quality": char.enhance_quality, "created_at": char.created_at,
        "has_photo": bool(char.photo_path), "has_face_data": char.face_embedding is not None,
        "credits": CREDITS
    }


@app.delete("/characters/{name}")
async def delete_character(name: str):
    success = char_manager.delete_character(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
    return {"success": True, "message": f"Character '{name}' deleted", "credits": CREDITS}


@app.post("/characters/{name}/activate")
async def activate_character(name: str):
    success = char_manager.set_active_character(name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Character '{name}' not found")
    char = char_manager.get_character(name)
    llm.set_character(char.name, char.system_prompt)
    return {"success": True, "active_character": name, "credits": CREDITS}


@app.post("/physics/config")
async def update_physics_config(request: PhysicsConfigRequest):
    config = tracker.config
    if request.mass is not None: config.mass = request.mass
    if request.damping is not None: config.damping = request.damping
    if request.spring_stiffness is not None: config.spring_stiffness = request.spring_stiffness
    if request.friction is not None: config.friction = request.friction
    if request.gravity is not None: config.gravity = request.gravity
    if request.smoothing_alpha is not None: config.smoothing_alpha = request.smoothing_alpha
    if request.intensity is not None: config.intensity = request.intensity
    if request.enabled is not None: config.enabled = request.enabled
    return {
        "success": True,
        "config": {
            "mass": config.mass, "damping": config.damping,
            "spring_stiffness": config.spring_stiffness, "friction": config.friction,
            "gravity": config.gravity, "smoothing_alpha": config.smoothing_alpha,
            "intensity": config.intensity, "enabled": config.enabled
        },
        "credits": CREDITS
    }


@app.get("/physics/status")
async def get_physics_status():
    physics_status = tracker.get_status()
    landmark_data = tracker.get_landmark_data_for_overlay()
    return {
        "physics": physics_status,
        "landmarks": landmark_data,
        "credits": CREDITS
    }


@app.post("/camera/source")
async def switch_camera_source(request: CameraSourceRequest):
    try:
        pipeline.switch_camera_source(request.source, request.url)
        return {"success": True, "source": request.source, "url": request.url, "credits": CREDITS}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/camera/status")
async def get_camera_status():
    status = pipeline.get_status()
    status["credits"] = CREDITS
    return status


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "DeepFaceReal-Physics real-time stream connected",
            "credits": CREDITS
        }))
        fps = 20
        frame_interval = 1.0 / fps
        while True:
            data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            message = json.loads(data)
            msg_type = message.get("type", "")
            if msg_type == "start_stream":
                fps = message.get("fps", 20)
                frame_interval = 1.0 / fps
                await websocket.send_text(json.dumps({"type": "stream_ready", "fps": fps}))
            elif msg_type == "frame":
                target_b64 = message.get("data", "")
                if target_b64:
                    target_img = decode_b64_image(target_b64)
                    active_char = char_manager.get_active_character()
                    if active_char and active_char.source_face is not None:
                        result = swapper.process_frame(target_img, active_char.source_face)
                        result_b64 = encode_b64_image(result)
                        mesh_data = face_3d.process_frame(result)
                        eye_state = eye_engine.update()
                        await websocket.send_text(json.dumps({
                            "type": "frame",
                            "data": f"data:image/jpeg;base64,{result_b64}",
                            "head_pose": mesh_data.get("pose", {}),
                            "eye_state": eye_state,
                            "credits": CREDITS
                        }))
                        await asyncio.sleep(frame_interval)
            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif msg_type == "stop":
                break
    except (asyncio.TimeoutError, WebSocketDisconnect):
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
        except:
            pass


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            character_name = message_data.get("character_name", None)
            if character_name:
                char = char_manager.get_character(character_name)
                if char:
                    llm.set_character(char.name, char.system_prompt)
            for chunk in llm.send_message_stream(user_message):
                await websocket.send_text(json.dumps({"type": "chunk", "content": chunk}))
            await websocket.send_text(json.dumps({"type": "done", "content": "", "credits": CREDITS}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
        except:
            pass


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_text(json.dumps({
            "type": "status", "content": "Video WebSocket connected", "credits": CREDITS
        }))
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            if message.get("type") == "frame":
                target_b64 = message.get("data", "")
                if target_b64:
                    target_img = decode_b64_image(target_b64)
                    active_char = char_manager.get_active_character()
                    if active_char and active_char.source_face is not None:
                        result = swapper.process_frame(target_img, active_char.source_face)
                        result_b64 = encode_b64_image(result)
                        await websocket.send_text(json.dumps({
                            "type": "frame",
                            "data": f"data:image/jpeg;base64,{result_b64}",
                            "credits": CREDITS
                        }))
                    else:
                        await websocket.send_text(json.dumps({"type": "error", "content": "No active character with face data"}))
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "content": str(e)}))
        except:
            pass


@app.on_event("startup")
async def startup():
    print(f"[API v2] Starting DeepFaceReal-Physics API v2.0.0... {CREDITS}")
    print(f"[API v2] Characters: {char_manager.list_characters()}")
    print(f"[API v2] OpenRouter: {'Set' if os.environ.get('OPENROUTER_API_KEY') else 'Demo mode'}")
    print("[API v2] Ready on port 8081")


@app.on_event("shutdown")
async def shutdown():
    print("[API v2] Shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)