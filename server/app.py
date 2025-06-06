from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import time
import shutil
import glob
import datetime
from random import choice
import torch
import torchvision
from torchvision import transforms
from torch import nn
import numpy as np
import cv2
import face_recognition
from PIL import Image as pImage
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
from typing import List
import base64
import io

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files - FIXED: Added leading slash
app.mount("/uploaded_images", StaticFiles(directory="uploaded_images"), name="uploaded_images")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize = transforms.Normalize(
    mean=-1*np.divide(mean, std), std=np.divide([1, 1, 1], std))

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}

# Detects GPU in device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = torchvision.models.resnext50_32x4d(weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if (len(frames) == self.count):
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)  # Shape: (1, seq_len, C, H, W)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def allowed_video_file(filename):
    return filename.split('.')[-1].lower() in ALLOWED_VIDEO_EXTENSIONS

def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    final_model = ""
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    list_models = glob.glob(os.path.join("models", "*.pt"))

    for i in list_models:
        model_name.append(os.path.basename(i))

    for i in model_name:
        try:
            seq = i.split("_")[3]
            if (int(seq) == sequence_length):
                sequence_model.append(i)
        except:
            pass

    if len(sequence_model) > 1:
        accuracy = []
        for i in sequence_model:
            acc = i.split("_")[1]
            accuracy.append(acc)
        max_index = accuracy.index(max(accuracy))
        final_model = sequence_model[max_index]
    else:
        final_model = sequence_model[0] if sequence_model else None
    
    return final_model

def im_convert(tensor, video_file_name=""):
    """Convert tensor to image for visualization."""
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def generate_gradcam_heatmap(model, img, video_file_name=""):
    """Generate GradCAM heatmap showing areas of focus for deepfake detection."""
    # Assume model and img are already on GPU (device)
    
    # Forward pass
    fmap, logits = model(img)  # fmap shape: (batch_size * seq_len, channels, h, w)
    
    # Softmax on logits (keep on GPU)
    logits_softmax = sm(logits)
    confidence, prediction = torch.max(logits_softmax, 1)
    confidence_val = confidence.item() * 100
    pred_idx = prediction.item()
    
    # Move weights and fmap to CPU only when converting to numpy
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    
    # Use last frame feature map (last in batch dimension)
    fmap_last = fmap[-1].detach().cpu().numpy()
    nc, h, w = fmap_last.shape
    fmap_reshaped = fmap_last.reshape(nc, h*w)
    
    # Compute GradCAM heatmap
    heatmap_raw = np.dot(fmap_reshaped.T, weight_softmax[pred_idx, :].T)
    heatmap_raw -= heatmap_raw.min()
    heatmap_raw /= heatmap_raw.max()
    heatmap_img = np.uint8(255 * heatmap_raw.reshape(h, w))
    
    # Resize heatmap to model input size
    heatmap_resized = cv2.resize(heatmap_img, (im_size, im_size))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Convert original image tensor to numpy (must move to CPU)
    original_img = im_convert(img[:, -1, :, :, :])
    original_img_uint8 = (original_img * 255).astype(np.uint8)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_img_uint8, 0.6, heatmap_colored, 0.4, 0)
    
    # Save heatmap and overlay images
    os.makedirs(os.path.join("static", "heatmaps"), exist_ok=True)
    heatmap_filename = f"{video_file_name}_heatmap_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    heatmap_path = os.path.join("static", "heatmaps", heatmap_filename)
    cv2.imwrite(heatmap_path, overlay)
    
    # Matplotlib visualization
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original Frame')
    plt.axis('on')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet')
    plt.title('Attention Heatmap')
    plt.axis('on')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay[..., ::-1])  # convert BGR to RGB for matplotlib
    plt.title(f'Overlay - Prediction: {"REAL" if pred_idx == 1 else "FAKE"} ({confidence_val:.1f}%)')
    plt.axis('on')
    
    plt.tight_layout()
    
    plt_filename = f"{video_file_name}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt_path = os.path.join("static", "heatmaps", plt_filename)
    plt.savefig(plt_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'prediction': pred_idx,
        'confidence': confidence_val,
        'heatmap_path': f"/static/heatmaps/{heatmap_filename}",
        'analysis_path': f"/static/heatmaps/{plt_filename}"
    }


def predict_with_gradcam(model, img, video_file_name=""):
    """Enhanced prediction function with GradCAM visualization."""
    return generate_gradcam_heatmap(model, img, video_file_name)

@app.post("/api/upload")
async def api_upload_video(file: UploadFile = File(...), sequence_length: int = 20):
    if not allowed_video_file(file.filename):
        raise HTTPException(status_code=400, detail="Only video files are allowed")
    
    # Save the uploaded file
    file_ext = file.filename.split('.')[-1]
    saved_video_file = f'uploaded_video_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.{file_ext}'
    
    # Create uploaded_videos directory if it doesn't exist
    os.makedirs("uploaded_videos", exist_ok=True)
    file_path = os.path.join("uploaded_videos", saved_video_file)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the video
    result = await process_video(file_path, sequence_length)
    
    return {
        "status": "success",
        "result": result["output"],
        "confidence": result["confidence"],
        "accuracy": result["accuracy"],
        "frames_processed": sequence_length,
        "preprocessed_images": result["preprocessed_images"],
        "faces_cropped_images": result["faces_cropped_images"],
        "heatmap_image": result["heatmap_image"],
        "analysis_image": result["analysis_image"],
        "gradcam_explanation": result["gradcam_explanation"]
    }

async def process_video(video_file, sequence_length):
    try:
        # Validate video file exists
        if not os.path.exists(video_file):
            raise HTTPException(status_code=400, detail="Video file not found")

        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        
        # Create dataset
        video_dataset = ValidationDataset(
            path_to_videos, sequence_length=sequence_length, transform=train_transforms)
        
        # Load model to device
        model = Model(2).to(device)  # UPDATED for GPU
        model_filename = get_accurate_model(sequence_length)
        
        if not model_filename:
            raise HTTPException(
                status_code=500, 
                detail=f"No suitable model found for sequence length {sequence_length}"
            )
        
        model_path = os.path.join("models", model_filename)
        
        if not os.path.exists(model_path):
            raise HTTPException(
                status_code=500, 
                detail=f"Model file not found at {model_path}"
            )
        
        model.load_state_dict(torch.load(model_path, map_location=device))  # UPDATED for GPU
        model.eval()
        
        # Process frames for visualization
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        
        if not frames:
            raise HTTPException(status_code=400, detail="No frames could be read from the video")
        
        # Create directories if they don't exist
        os.makedirs(os.path.join("static", "uploaded_images"), exist_ok=True)
        
        # Save preprocessed images
        preprocessed_images = []
        for i in range(1, min(sequence_length + 1, len(frames))):
            try:
                frame = frames[i]
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = pImage.fromarray(image, 'RGB')
                image_name = f"{video_file_name_only}_preprocessed_{i}.png"
                image_path = os.path.join("static", "uploaded_images", image_name)
                img.save(image_path)
                preprocessed_images.append(f"/static/uploaded_images/{image_name}")
            except Exception as e:
                print(f"Error processing frame {i}: {str(e)}")
                continue
        
        # Face detection and cropping
        padding = 40
        faces_cropped_images = []
        faces_found = 0
        
        for i in range(1, min(sequence_length + 1, len(frames))):
            try:
                frame = frames[i]
                face_locations = face_recognition.face_locations(frame)
                
                if not face_locations:
                    continue
                    
                top, right, bottom, left = face_locations[0]
                frame_face = frame[
                    max(0, top-padding):min(frame.shape[0], bottom+padding),
                    max(0, left-padding):min(frame.shape[1], right+padding)
                ]
                image = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
                img = pImage.fromarray(image, 'RGB')
                image_name = f"{video_file_name_only}_cropped_faces_{i}.png"
                image_path = os.path.join("static", "uploaded_images", image_name)
                img.save(image_path)
                faces_found += 1
                faces_cropped_images.append(f"/static/uploaded_images/{image_name}")
            except Exception as e:
                print(f"Error processing face in frame {i}: {str(e)}")
                continue
        
        if faces_found == 0:
            raise HTTPException(status_code=400, detail="No faces detected in the video")
        
        # Make prediction with GradCAM
        try:
            # Move dataset tensor to device
            input_tensor = video_dataset[0].to(device)  # UPDATED for GPU
            
            gradcam_result = predict_with_gradcam(model, input_tensor, video_file_name_only)
            confidence = round(gradcam_result['confidence'], 1)
            output = "REAL" if gradcam_result['prediction'] == 1 else "FAKE"
            
            # Extract accuracy from model filename safely
            try:
                accuracy = model_filename.split("_")[1] if len(model_filename.split("_")) > 1 else "00"
                decimal = model_filename.split("_")[2] if len(model_filename.split("_")) > 2 else "00"
            except:
                accuracy = "00"
                decimal = "00"
            
            # Create explanation for GradCAM
            gradcam_explanation = {
                "description": "The heatmap shows areas where the AI model focused its attention when making the prediction.",
                "interpretation": {
                    "red_areas": "High attention - areas that strongly influenced the decision",
                    "yellow_areas": "Medium attention - moderately important areas", 
                    "blue_areas": "Low attention - areas with minimal influence on the decision"
                },
                "prediction_basis": f"The model classified this video as {output} with {confidence}% confidence based on the highlighted facial regions."
            }
            
            return {
                "preprocessed_images": preprocessed_images,
                "faces_cropped_images": faces_cropped_images,
                "output": output,
                "confidence": confidence,
                "accuracy": accuracy,
                "decimal": decimal,
                "heatmap_image": gradcam_result['heatmap_path'],
                "analysis_image": gradcam_result['analysis_path'],
                "gradcam_explanation": gradcam_explanation
            }
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error making prediction: {str(e)}"
            )
            
    except HTTPException:
        raise  # Re-raise HTTPExceptions
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing video: {str(e)}"
        )

#Extension Endpoint
@app.post("/predict")

async def predict_frames(data: dict):

    try:

        print("Received request to /predict endpoint")

        frames = data.get('frames', [])

        if not frames:

            print("No frames provided in request")

            raise HTTPException(status_code=400, detail="No frames provided")



        print(f"Processing {len(frames)} frames")

        sequence_length = 20

        processed_frames = []

        

        for i, frame_base64 in enumerate(frames[:sequence_length]):

            try:

                if ',' in frame_base64:

                    frame_base64 = frame_base64.split(',')[1]

                

                frame_data = base64.b64decode(frame_base64)

                frame = cv2.imdecode(

                    np.frombuffer(frame_data, np.uint8),

                    cv2.IMREAD_COLOR

                )


                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Attempt face detection but don't require it
                try:
                    faces = face_recognition.face_locations(frame)
                    if faces:
                        top, right, bottom, left = faces[0]
                        # Ensure we don't crop too aggressively
                        height, width = frame.shape[:2]
                        margin = int(min(width, height) * 0.1)  # 10% margin
                        
                        # Add margins and ensure bounds
                        top = max(0, top - margin)
                        bottom = min(height, bottom + margin)
                        left = max(0, left - margin)
                        right = min(width, right + margin)
                        
                        frame = frame[top:bottom, left:right, :]
                        print(f"Face detected in frame {i+1} with margins")
                    else:
                        # If no face detected, use the whole frame
                        print(f"No face detected in frame {i+1}, using full frame")
                except Exception as e:
                    print(f"Face detection error in frame {i+1}: {str(e)}, using full frame")
                
                # Resize frame if too large
                height, width = frame.shape[:2]
                max_dimension = 512  # Maximum dimension to process
                if height > max_dimension or width > max_dimension:
                    scale = max_dimension / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    print(f"Resized frame {i+1} to {new_width}x{new_height}")
                
                processed_frames.append(frame)

            except Exception as e:
                print(f"Error processing frame {i+1}: {str(e)}")
                continue



        if not processed_frames:

            print("No valid frames could be processed")

            raise HTTPException(status_code=400, detail="No valid frames could be processed")



        print(f"Successfully processed {len(processed_frames)} frames")



        frames_tensor = torch.stack([

            train_transforms(frame) for frame in processed_frames

        ])

        frames_tensor = frames_tensor.unsqueeze(0)



        model = Model(2).cpu()

        model_filename = get_accurate_model(sequence_length)

        

        if not model_filename:

            print(f"No suitable model found for sequence length {sequence_length}")

            raise HTTPException(

                status_code=500,

                detail=f"No suitable model found for sequence length {sequence_length}"

            )

        

        print(f"Using model: {model_filename}")

        

        # Extract model accuracy from filename

        try:

            parts = model_filename.split('_')

            accuracy = float(parts[1])  # Get the 87 from the filename

            print(f"Extracted accuracy: {accuracy}%")

            if accuracy <= 0 or accuracy > 100:

                print("Invalid accuracy value, using default")

                accuracy = 87.0

        except Exception as e:

            print(f"Error extracting accuracy: {str(e)}")

            accuracy = 87.0  # Fallback to known accuracy

            print(f"Using default accuracy: {accuracy}%")

        

        model_path = os.path.join("models", model_filename)

        print(f"Loading model from: {model_path}")

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        model.eval()



        with torch.no_grad():

            _, logits = model(frames_tensor)

            probabilities = sm(logits)

            _, prediction = torch.max(probabilities, 1)

            confidence = probabilities[:, int(prediction.item())].item() * 100

            

            # In your model, 1 is REAL and 0 is FAKE

            is_fake = prediction.item() == 0

            print(f"Prediction: {'FAKE' if is_fake else 'REAL'} with {confidence:.2f}% confidence")

            print(f"Model accuracy: {accuracy}%")



        response_data = {

            "is_fake": is_fake,

            "confidence": confidence,

            "frames_processed": len(processed_frames),

            "model_accuracy": accuracy

        }

        print(f"Sending response: {response_data}")

        return response_data



    except Exception as e:

        print(f"Error in predict_frames: {str(e)}")

        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test")
def test_endpoint():
    return {"status": "success", "message": "API is working!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
