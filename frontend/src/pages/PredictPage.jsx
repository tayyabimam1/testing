import React, { useState } from "react";
import Button from 'react-bootstrap/Button'
import "../App.css";

const UploadPage = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      const fileExtension = file.name.split(".").pop().toLowerCase();
      const allowedTypes = [
        "mp4", "gif", "webm", "avi", "3gp", "wmv", "flv", "mkv"
      ];

      if (allowedTypes.includes(fileExtension)) {
        setSelectedFile(file);
        setError(null);
        setResult(null); // Reset results when new file is selected
      } else {
        setError(
          "Please select a valid video file (mp4, webm, avi, 3gp, wmv, flv, mkv, gif)"
        );
        setSelectedFile(null);
      }
    }
  };

  const handleDetectDeepfake = async () => {
    if (!selectedFile) {
      setError("Please select a video file first");
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("sequence_length", 20);

    try {
      const response = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to process video");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "An error occurred while processing the video");
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setSelectedFile(null);
    setError(null);
    setResult(null);
    const fileInput = document.getElementById("video-upload");
    if (fileInput) fileInput.value = "";
  };

  return (
    <div className="custom-bg predict-page min-vh-100 d-flex flex-column justify-content-center">
      <div className="container py-5">
        <h1 className="text-white text-center display-2 mb-2">Upload Video</h1>
        <p className="text-center text-mute fs-4 mb-5">Upload a video to analyze for deepfake manipulation</p>
        
        {!result ? (
          <div className="upload-container mx-auto" style={{ maxWidth: '800px' }}>
            <div className="border-2 border-dashed rounded-4 p-5 mb-4 text-center position-relative" 
              style={{ 
                border: '2px solid rgba(19, 12, 231, 0.83)',
                background: 'rgba(255, 255, 255, 0.05)'
              }}>
              <div className="mb-4">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" 
                  strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" 
                  className="text-secondary mx-auto mb-3">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              
              <p className="text-secondary mb-4">Drag and drop your video here</p>
              <p className="text-mute small mb-4">Supports MP4, MOV, AVI, and WebM formats up to 500MB</p>
              
              <input
                type="file"
                className="form-control visually-hidden"
                id="video-upload"
                accept="video/*"
                onChange={handleFileSelect}
              />
              <Button 
                variant="" 
                onClick={() => document.getElementById('video-upload').click()}
                className="px-4 custom-btn">
                Select Video
              </Button>
            </div>

            {error && <div className="alert alert-danger">{error}</div>}
            {selectedFile && (
              <div className="alert alert-success">
                Selected file: {selectedFile.name}
              </div>
            )}

            <div className="d-flex gap-2 justify-content-center">
              <Button
                variant="primary"
                onClick={handleDetectDeepfake}
                disabled={!selectedFile || isLoading}
              >
                {isLoading ? "Processing..." : "Detect Deepfake"}
              </Button>
              <Button
                variant="secondary"
                onClick={resetForm}
                disabled={isLoading}
              >
                Clear
              </Button>
            </div>
          </div>
        ) : (
          <div className="card custom-card p-4 shadow-sm">
            <h2 className="text-center text-color mb-4">Detection Results</h2>
            
            <div className="mb-4">
              <h3 className="mb-3 text-color">
                Result: <span className={`${result.result?.toLowerCase() === 'real' ? 'text-success' : 'text-danger'} fw-bold`}>
                  {result.result}
                </span>
              </h3>
              <div className="d-flex align-items-center">
                <div className="text-mute">Confidence: {result.confidence}%</div>
              </div>
              <p className="mb-2 text-mute">Model Accuracy: {result.accuracy}%</p>
              <p className="mb-2 text-mute">Frames Processed: {result.frames_processed}</p>
            </div>

            {result.analysis_image && (
              <div>
                <h3 className="mb-3 text-color">AI Analysis Visualization</h3>
                <img 
                  src={`http://localhost:8000${result.analysis_image}`}
                  alt="GradCAM Analysis"
                  className="img-fluid mb-3"
                />
                {result.gradcam_explanation && (
                  <div>
                    <h4 className="mb-3 text-color">Understanding the Heatmap:</h4>
                    <p className="text-mute">{result.gradcam_explanation.description}</p>
                    <ul className="list-group mb-3">
                      <li className="list-group-item custom-color text-mute"><strong>Red areas:</strong> {result.gradcam_explanation.interpretation.red_areas}</li>
                      <li className="list-group-item custom-color text-mute"><strong>Yellow areas:</strong> {result.gradcam_explanation.interpretation.yellow_areas}</li>
                      <li className="list-group-item custom-color text-mute"><strong>Blue areas:</strong> {result.gradcam_explanation.interpretation.blue_areas}</li>
                    </ul>
                    <p className="text-mute"><strong>Prediction Basis:</strong> {result.gradcam_explanation.prediction_basis}</p>
                  </div>
                )}
              </div>
            )}

            <Button
              variant="primary"
              onClick={resetForm}
              className="mt-3"
            >
              Upload Another Video
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPage;
