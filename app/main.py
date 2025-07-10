from fastapi import FastAPI, UploadFile, File, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from urllib.parse import unquote
from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np
import tempfile
from PIL import Image
import io
import uvicorn
import shutil
import json
import pathlib
import mimetypes
from dotenv import load_dotenv
# Import your model components
import os
from huggingface_hub import hf_hub_download, snapshot_download
import time
from simImage import SimpleClassificationPipeline
from plant_nlp_system import PlantQA
from retrieval_system import ImageRetrievalSystem
from setup_db import PlantDBService
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Commented out for broader compatibility

load_dotenv()

app = FastAPI(title="Plant Classification and Q&A API", 
              description="API for classifying plants and answering questions about them")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "http://localhost:3001", 
        "https://*.vercel.app",   
        "*" 
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CONFIG = {
    "token": os.environ.get("HUGGINGFACE_TOKEN", ""),
    "classifier_path": os.environ.get("CLASSIFIER_PATH", "./models/classify.pth"),
    "embed_path": os.environ.get("EMBED_PATH", "./models/embedding.pth"), 
    "labels_path": os.environ.get("LABELS_PATH", "./plant_data"), 
    "metadata_path": os.environ.get("METADATA_PATH", "./data/plant_data.json"),
    "retrieval_index_file": os.environ.get("RETRIEVAL_INDEX_PATH", "./data/plant_index_new.pkl"),
    "images_path": os.environ.get("JSON_DB_PATH", "./data/plant_images_db.json"),
    "graph_path": os.environ.get("GRAPH_PATH", "./data/plant_graph.json")  # Path to the plant graph JSON
}

DATA_SOURCE ="json"
# Global variables for models & database
mongodb_service = None
classification_pipeline = None
llm_qa = None
retrieval_system = None
plant_metadata = None
plant_images_db = {}
# Temporary directory for uploaded images
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def download_model_files():
    repo_id = "hqta1110/plant-chatbot"
    token = MODEL_CONFIG["token"]

    download_targets = [
        {
            "name": "models",
            "local_dir": "./",
            "allow_patterns": ["models/*"]
        },
        {
            "name": "data",
            "local_dir": "./",
            "allow_patterns": ["data/*"]
        }
    ]
    for target in download_targets:
        print(f"📦 Downloading {target['name']} from Hugging Face...")
        os.makedirs(target["local_dir"], exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            allow_patterns=target["allow_patterns"],
            local_dir=target["local_dir"],
            local_dir_use_symlinks=False,
            ignore_patterns=None,
        )

    print("🎉 All files downloaded successfully and isolated inside container.")


@app.on_event("startup")
async def load_models():
    """Load all ML models and data on server startup"""
    global classification_pipeline, llm_qa, retrieval_system
    global plant_metadata, plant_images_db, mongodb_service, DATA_SOURCE

    print("📥 Downloading model & data files if needed...")
    download_model_files()  

    print("🔧 Initializing data sources...")

    # === 1. Load from MongoDB if available ===
    if DATA_SOURCE == "mongodb":
        try:
            mongodb_service = PlantDBService()
            if mongodb_service.connect():
                print("✅ MongoDB connected")
                plant_metadata = mongodb_service.get_all_plants()
                plant_images_db = {"plants": {}, "total_plants": 0, "total_images": 0}
                return  
            else:
                print("⚠️ MongoDB connection failed, fallback to JSON")
                DATA_SOURCE = "json"
        except Exception as e:
            print(f"⚠️ MongoDB error: {e}, fallback to JSON")
            DATA_SOURCE = "json"

    # === 2. Load from JSON ===
    if DATA_SOURCE == "json":
        print("📄 Loading data from JSON files...")
        try:
            with open(MODEL_CONFIG["metadata_path"], "r", encoding="utf-8") as f:
                plant_metadata = json.load(f)
            print(f"Loaded metadata for {len(plant_metadata)} plant species")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            plant_metadata = {}

        try:
            with open(MODEL_CONFIG["images_path"], "r", encoding="utf-8") as f:
                plant_images_db = json.load(f)
            print(f"Loaded image DB with {plant_images_db['total_plants']} plants")
        except Exception as e:
            print(f"Could not load image DB: {e}")
            plant_images_db = {"plants": {}, "total_images": 0, "total_plants": 0}

    # === 3. Initialize model pipeline ===
    print("⚙️ Initializing classification pipeline...")
    classification_pipeline = SimpleClassificationPipeline(
        classifier_path=MODEL_CONFIG["classifier_path"],
        label_path=MODEL_CONFIG["labels_path"],
        token = MODEL_CONFIG['token']
    )

    print("🧠 Initializing LLM QA module...")
    llm_qa = PlantQA(
        metadata_path=MODEL_CONFIG["metadata_path"],
        db_service=mongodb_service,
        cache_dir="./hf_cache"
    )

    print("🔎 Initializing image retrieval system...")
    retrieval_system = ImageRetrievalSystem(
        model_path=MODEL_CONFIG["embed_path"],
        image_folder=MODEL_CONFIG["labels_path"],
        index_file=MODEL_CONFIG["retrieval_index_file"]
    )

    print("✅ All models & systems initialized successfully!")



# Pydantic models for request/response
class ClassificationResult(BaseModel):
    label: str
    confidence: float
    image_path: Optional[str] = None

class ClassificationResponse(BaseModel):
    results: List[ClassificationResult]

class QARequest(BaseModel):
    question: str
    label: Optional[str] = None  
    session_id: Optional[str] = None  # Add session support

class QAResponse(BaseModel):
    answer: str
    session_id: Optional[str] = None  # Return session ID for client tracking

class ResetConversationRequest(BaseModel):
    session_id: Optional[str] = None

class SessionStatsResponse(BaseModel):
    active_sessions: int
    session_timeout_minutes: int


@app.on_event("shutdown")
async def cleanup():
    """Clean up resources when shutting down"""
    global classification_pipeline, llm_qa, retrieval_system, mongodb_service
    
    print("Cleaning up resources...")
    if classification_pipeline:
        classification_pipeline.cleanup()
    if llm_qa:
        llm_qa.close()
    
    # Close MongoDB connection if it exists
    if mongodb_service:
        print("Closing MongoDB connection...")
        mongodb_service.close()
    
    # Clean temporary files
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Cleanup complete!")


@app.get("/api/plant-images/{scientific_name}")
async def get_plant_images(scientific_name: str):
    """Serve image information for a specific plant species"""
    global plant_images_db, mongodb_service, DATA_SOURCE
    
    # Use MongoDB if available
    if DATA_SOURCE == "mongodb" and mongodb_service:
        try:
            # Get plant images from MongoDB using our service
            return mongodb_service.get_plant_images(scientific_name)
        except Exception as e:
            print(f"Error fetching images from MongoDB: {e}")
            # Fall back to JSON if MongoDB query fails
    
    # MongoDB not available or query failed, use original JSON implementation
    try:
        # Either use the global variable if loaded or load the JSON file
        images_data = plant_images_db
        if not images_data:
            with open(MODEL_CONFIG['images_path'], 'r', encoding='utf-8') as f:
                images_data = json.load(f)
        
        if scientific_name not in images_data.get("plants", {}):
            return {
                "plant": scientific_name,
                "images": [],
                "error": "Plant not found in image database"
            }
        
        plant_data = images_data["plants"][scientific_name]
        
        # Transform absolute paths to web-accessible URLs
        transformed_images = []
        for img in plant_data.get("images", []):
            # Create web-accessible URL instead of filesystem path
            web_url = f"/plant-images/{scientific_name}/{img['filename']}"
            transformed_images.append({
                "filename": img["filename"],
                "path": web_url,
                "is_primary": img.get("is_primary", False),
                "order": img.get("order", 0)
            })
        
        return {
            "plant": scientific_name,
            "total_images": plant_data.get("total_images", 0),
            "images": transformed_images
        }
    except Exception as e:
        return {
            "plant": scientific_name,
            "images": [],
            "error": str(e)
        }
def compress_image(image_path: str, max_width: int = 800, quality: int = 85) -> bytes:
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (maintain aspect ratio)
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Save to memory buffer with compression
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            return buffer.getvalue()
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error compressing image: {str(e)}")
@app.get("/plant-images/{species_name}/{filename}")
@app.head("/plant-images/{species_name}/{filename}")
async def serve_plant_image(species_name: str, filename: str):
    """Serve individual plant image files"""
    try:
        species_name = species_name.replace("..", "").replace("/", "")
        filename = filename.replace("..", "").replace("/", "")
        
        # Construct the full path
        image_path = os.path.join(MODEL_CONFIG['labels_path'], species_name, filename)
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")  
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Check if it's an image file
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            raise HTTPException(status_code=400, detail="Not an image file")
        
        # Guess MIME type
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        compressed_image = compress_image(image_path)
        print(f"Serving image: {image_path}")  # Debug log
        
        return Response(
            content=compressed_image,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "public, max-age=86400"
            }
    )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving image {species_name}/{filename}: {e}")
        raise HTTPException(status_code=500, detail="Error serving image")
    
@app.get("/api/plants")
async def get_plants():
    """Return all plants metadata for the library mode"""
    global plant_metadata, mongodb_service, DATA_SOURCE
    
    print("Received request for /api/plants")
    
    # Use MongoDB if available
    if DATA_SOURCE == "mongodb" and mongodb_service:
        try:
            print("Fetching plants from MongoDB")
            start_time = time.time()  
            plants_data = mongodb_service.get_all_plants()
            fetch_time = time.time() - start_time  
            print(f"MongoDB fetch completed in {fetch_time:.2f} seconds, retrieved {len(plants_data)} plants")
            
            if plants_data:
                return plants_data
            else:
                print("No plants returned from MongoDB, falling back to JSON")
                # If MongoDB returned empty results, fall back to JSON
        except Exception as e:
            print(f"Error fetching plants from MongoDB: {e}")
            # Fall back to JSON if MongoDB query fails
    
    # MongoDB not available or query failed, use original JSON implementation
    if not plant_metadata:
        print("Error: Plant metadata not loaded")
        raise HTTPException(
            status_code=500, 
            detail="Plant metadata not loaded. Please check the server configuration."
        )
    
    try:
        # Check data integrity
        if not isinstance(plant_metadata, dict):
            print(f"Error: Unexpected metadata type: {type(plant_metadata)}")
            raise HTTPException(
                status_code=500,
                detail=f"Unexpected metadata format: {type(plant_metadata)}"
            )
        
        metadata_count = len(plant_metadata)
        print(f"Successfully serving metadata for {metadata_count} plants from JSON")
        
        # Return the existing JSON data
        return plant_metadata
        
    except Exception as e:
        print(f"Error in /api/plants endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing plant metadata: {str(e)}"
        )

@app.post("/api/classify", response_model=ClassificationResponse)
async def classify_image(files: List[UploadFile] = File(...)):
    """Classify one or more uploaded plant images and return top 5 predictions"""
    if not files:
        raise HTTPException(status_code=400, detail="No image files provided")
    
    # For compatibility, convert single file to list
    if not isinstance(files, list):
        files = [files]
    
    # Save the uploaded files temporarily
    file_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)
    
    try:
        # Use the new class-based validation method
        valid_file_paths, ood_results = retrieval_system.validate_images_class_based(
            file_paths, 
            threshold = 1.180,
            k=100,  # Initial number of neighbors to retrieve
            min_unique_classes=10  # Minimum number of unique classes to find
        )
        
        # If no valid images, return "not found"
        if not valid_file_paths:
            # Find the highest confidence OOD result if available
            if ood_results:
                sorted_ood = sorted(ood_results, key=lambda x: x.get("confidence", 0) 
                                if isinstance(x.get("confidence"), (int, float)) else 0, 
                                reverse=True)
                best_ood = sorted_ood[0]
                confidence = best_ood.get("confidence", 1.0)
            else:
                confidence = 1.0
                
            return ClassificationResponse(results=[
                ClassificationResult(
                    label="Không tồn tại trong cơ sở dữ liệu",
                    confidence=confidence,
                    image_path=None
                )
            ])
        
        # Process only valid images
        if len(valid_file_paths) > 1:
            top5 = classification_pipeline.process_multiple_images_topk(valid_file_paths, top_k=6)
        else:
            top5 = classification_pipeline.process_image_topk(valid_file_paths[0], top_k=6)
        
        highest_prob = max([prob for _, prob, _ in top5])

        if highest_prob < 0.1:
            return ClassificationResponse(results=[
                ClassificationResult(
                    label="Không tồn tại trong cơ sở dữ liệu",
                    confidence=1.,
                    image_path=None
                )
            ])
        # Format results
        results = []
        for label, prob, rep_img in top5:
            results.append(
                ClassificationResult(
                    label=label,
                    confidence=float(prob),
                    image_path=rep_img if rep_img else None
                )
            )
        
        return ClassificationResponse(results=results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image(s): {str(e)}")
    
    finally:
        # Clean up the temporary files
        for file_paths in file_paths:
            if os.path.exists(file_paths):
                os.remove(file_paths)

@app.post("/api/qa", response_model=QAResponse)
async def answer_question(request: QARequest):
    """Answer a question about plants, with session-based conversation management"""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        # Extract plain label from label string if in format "Label (99.99%)"
        label = None
        if request.label:
            import re
            match = re.match(r"^(.+?)\s\(\d+\.\d+%\)$", request.label)
            label = match.group(1) if match else request.label
        
        # Generate answer with session support
        print(f"🔍 Processing Q&A request (session: {request.session_id or 'new'})")
        answer = llm_qa.generate_answer(label, request.question, request.session_id)
        
        return QAResponse(answer=answer, session_id=request.session_id)
    
    except Exception as e:
        print(f"Error in Q&A endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/api/reset-conversation")
async def reset_conversation(request: ResetConversationRequest = None):
    """Reset conversation history for a specific session"""
    try:
        if not llm_qa:
            raise HTTPException(
                status_code=500, 
                detail="QA system not initialized"
            )
        
        session_id = request.session_id if request else None
        
        if session_id:
            llm_qa.reset_conversation(session_id)
            print(f"🔄 Reset conversation for session: {session_id[:8]}...")
            return {
                "status": "success",
                "message": f"Conversation history for session {session_id[:8]}... has been reset successfully",
                "session_id": session_id
            }
        else:
            return {
                "status": "info",
                "message": "No session ID provided. Sessions are automatically managed.",
                "session_id": None
            }
    except Exception as e:
        print(f"Error resetting conversation: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error resetting conversation: {str(e)}"
        )

@app.get("/api/session-stats", response_model=SessionStatsResponse)
async def get_session_stats():
    """Get current session statistics"""
    try:
        if not llm_qa:
            raise HTTPException(status_code=500, detail="QA system not initialized")
        
        active_sessions = llm_qa.session_manager.get_session_count()
        timeout_minutes = llm_qa.session_manager.session_timeout // 60
        
        return SessionStatsResponse(
            active_sessions=active_sessions,
            session_timeout_minutes=timeout_minutes
        )
    except Exception as e:
        print(f"Error getting session stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting session stats: {str(e)}")
        

app.mount("/plant-images", StaticFiles(directory=MODEL_CONFIG['labels_path']), name="plant_images") # Handled by reverse proxy
# # Serve static files (React frontend)
# app.mount("/", StaticFiles(directory="../frontend/build", html=True), name="static") # Handled by reverse proxy

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=9696, reload=True)