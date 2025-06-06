// Configuration
const API_BASE = 'http://localhost:8000';
const API_ENDPOINT = `${API_BASE}/predict`;
const FRAME_SEQUENCE_LENGTH = 20;
const MIN_VIDEO_DIMENSION = 100; // px - ignore tiny videos

// Video platform selectors with comprehensive coverage
const VIDEO_SELECTORS = {
  'youtube.com': 'video.html5-main-video, video.video-stream, .html5-video-player video, #movie_player video',
  'instagram.com': 'video, video.html5-main-video, video.video-stream, video[src], video.tWeCl, video._ab1d, article video',
  // Comprehensive X.com selectors - fallback to generic 'video' for maximum compatibility
  'x.com': 'video[type="video/mp4"], video.css-1dbjc4n, video.html5-main-video, video.video-stream, video[src], video[preload], div[data-testid="videoPlayer"] video, div[data-testid="videoComponent"] video, video'
};

// State management
let currentVideo = null;
let isAnalyzing = false;
let extensionActive = false;
let videoRegistry = new Map(); // Track videos with unique IDs

// Generate unique ID for video elements
function generateVideoId(video) {
  // Create a unique identifier based on video properties and position
  const rect = video.getBoundingClientRect();
  const id = `video_${Math.round(rect.top)}_${Math.round(rect.left)}_${video.videoWidth}_${video.videoHeight}_${Date.now()}`;
  return id;
}

// Initialize extension state
function initializeExtension() {
  console.log('DeepSight: Initializing extension...');

  // Check if extension is active
  chrome.storage.local.get('isActive', (data) => {
    extensionActive = !!data.isActive;
    console.log('DeepSight: Extension active state:', extensionActive);
    if (extensionActive) {
      initializeDetection();
    } else {
      removeAllButtons();
    }
  });
}

// Remove all buttons and cleanup
function removeAllButtons() {
  console.log('DeepSight: Removing all buttons');
  
  // Remove all buttons
  const buttons = document.querySelectorAll('.deepdetect-video-button');
  buttons.forEach(button => button.remove());
  
  // Clear video registry
  videoRegistry.clear();
  
  // Process videos with deepdetect data
  document.querySelectorAll('video').forEach(video => {
    if (video.deepdetectButton) {
      video.deepdetectButton.remove();
    }
    
    // Clear data
    delete video.deepdetectButton;
    delete video.dataset.deepdetectInitialized;
    delete video.dataset.deepdetectId;
  });
}

// Listen for storage changes
chrome.storage.onChanged.addListener((changes, namespace) => {
  if (namespace === 'local' && changes.isActive) {
    extensionActive = changes.isActive.newValue;
    console.log('DeepSight: Extension active state changed to:', extensionActive);
    if (extensionActive) {
      initializeDetection();
    } else {
      removeAllButtons();
    }
  }
});

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('DeepSight: Received message:', request);
  
  if (request.action === 'activate') {
    console.log('DeepSight: Activating...');
    extensionActive = true;
    initializeDetection();
  } 
  else if (request.action === 'deactivate') {
    console.log('DeepSight: Deactivating...');
    extensionActive = false;
    removeAllButtons();
  }
  else if (request.action === 'analyzeVideoInPopup') {
    if (extensionActive) {
      // Find video by ID first, then fallback to src matching
      let video = null;
      
      if (request.videoId) {
        video = videoRegistry.get(request.videoId);
      } else if (request.videoSrc) {
        // Fallback to src matching
        video = document.querySelector(`video[src="${request.videoSrc}"]`);
        if (!video) {
          // Try currentSrc
          const videos = document.querySelectorAll('video');
          for (const v of videos) {
            if (v.currentSrc === request.videoSrc) {
              video = v;
              break;
            }
          }
        }
      }
      
      if (!video) {
        // Last resort: find the most likely video
        video = findMostLikelyVideo();
      }
      
      if (video) {
        console.log('DeepSight: Found video for analysis:', video);
        analyzeVideo(video).then(result => {
          chrome.runtime.sendMessage({
            action: 'analysisResult',
            result: result
          });
        }).catch(error => {
          console.error('DeepSight: Analysis failed:', error);
          chrome.runtime.sendMessage({
            action: 'analysisResult',
            result: { error: error.message }
          });
        });
      } else {
        console.error('DeepSight: Video not found for analysis');
        chrome.runtime.sendMessage({
          action: 'analysisResult',
          result: { error: 'Video not found' }
        });
      }
    }
  }
  else if (request.action === 'getVideoInfo') {
    // Find the most relevant video
    const video = findMostLikelyVideo();
    
    if (video) {
      console.log('DeepSight: Found video for info:', {
        src: video.src,
        currentSrc: video.currentSrc,
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight
      });
      
      // Create a thumbnail from the video
      const canvas = document.createElement('canvas');
      const width = video.videoWidth || 320;
      const height = video.videoHeight || 240;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      
      try {
        ctx.drawImage(video, 0, 0, width, height);
        const thumbnail = canvas.toDataURL('image/jpeg', 0.8);
        
        const videoId = video.dataset.deepdetectId || generateVideoId(video);
        if (!video.dataset.deepdetectId) {
          video.dataset.deepdetectId = videoId;
          videoRegistry.set(videoId, video);
        }
        
        sendResponse({
          hasVideo: true,
          videoInfo: {
            id: videoId,
            src: video.src || video.currentSrc || 'dynamic',
            thumbnail: thumbnail,
            width: width,
            height: height
          }
        });
      } catch (e) {
        console.error('Error creating thumbnail:', e);
        const videoId = video.dataset.deepdetectId || generateVideoId(video);
        if (!video.dataset.deepdetectId) {
          video.dataset.deepdetectId = videoId;
          videoRegistry.set(videoId, video);
        }
        
        sendResponse({ 
          hasVideo: true,
          videoInfo: {
            id: videoId,
            src: video.src || video.currentSrc || 'dynamic',
            thumbnail: null,
            width: width,
            height: height,
            error: 'Could not create thumbnail - CORS restriction'
          }
        });
      }
    } else {
      sendResponse({ hasVideo: false });
    }
    return true; // Keep the message channel open for async response
  }
  return true;
});

// Find the most likely video to analyze
function findMostLikelyVideo() {
  const videos = document.querySelectorAll('video');
  
  if (videos.length === 0) return null;
  if (videos.length === 1) return videos[0];
  
  // Prioritize videos that are:
  // 1. Currently playing
  // 2. Have reasonable dimensions
  // 3. Are visible
  
  let bestVideo = null;
  let bestScore = -1;
  
  videos.forEach(video => {
    let score = 0;
    
    // Playing video gets priority
    if (!video.paused) score += 10;
    
    // Has loaded data
    if (video.readyState >= 2) score += 5;
    
    // Reasonable size
    const width = video.videoWidth || video.offsetWidth;
    const height = video.videoHeight || video.offsetHeight;
    if (width >= MIN_VIDEO_DIMENSION && height >= MIN_VIDEO_DIMENSION) {
      score += 3;
    }
    
    // Visible
    const rect = video.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) score += 2;
    
    // Has current time (indicating it's been interacted with)
    if (video.currentTime > 0) score += 1;
    
    if (score > bestScore) {
      bestScore = score;
      bestVideo = video;
    }
  });
  
  return bestVideo;
}

// Main initialization
function initializeDetection() {
  if (!extensionActive) {
    console.log('DeepSight: Extension not active, skipping initialization');
    return;
  }
  
  console.log('DeepSight: Starting detection initialization...');

  // Add initial delay for YouTube to fully load video player
  const delay = window.location.hostname.includes('youtube.com') ? 1500 : 500;
  setTimeout(() => {
    // Process existing videos
    processVideos();

    // Set up observer for dynamically loaded videos with adaptive debouncing
    let observerTimeout;
    const observer = new MutationObserver((mutations) => {
      if (mutations.some(m => m.addedNodes.length > 0)) {
        clearTimeout(observerTimeout);
        // Use longer debounce for dynamic loading
        const debounceTime = window.location.hostname.includes('youtube.com') ? 2000 : 500;
        observerTimeout = setTimeout(() => {
          processVideos();
        }, debounceTime);
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }, delay);
}

// Process all videos on the page
function processVideos() {
  if (!extensionActive) {
    console.log('DeepSight: Extension not active, skipping video processing');
    return;
  }
  
  console.log('DeepSight: Processing videos...');
  
  // Get current domain selector
  const currentDomain = window.location.hostname;
  let selector = 'video'; // Default fallback to all videos
  
  for (const [domain, sel] of Object.entries(VIDEO_SELECTORS)) {
    if (currentDomain.includes(domain)) {
      selector = sel;
      console.log('DeepSight: Using selector for domain:', domain, 'selector:', sel);
      break;
    }
  }
  
  // Find all videos using the domain-specific selector
  const videos = document.querySelectorAll(selector);
  console.log('DeepSight: Found', videos.length, 'videos with selector:', selector);
  
  if (videos.length === 0 && currentDomain.includes('youtube.com')) {
    console.log('DeepSight: No videos found on YouTube, trying fallback selector');
    const fallbackVideos = document.querySelectorAll('video');
    console.log('DeepSight: Found', fallbackVideos.length, 'videos with fallback selector');
    if (fallbackVideos.length > 0) {
      videos = fallbackVideos;
    }
  }
  
  videos.forEach((video, index) => {
    if (!video.dataset.deepdetectInitialized) {
      console.log(`DeepSight: Setting up video ${index}:`, {
        src: video.src,
        currentSrc: video.currentSrc,
        readyState: video.readyState,
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight,
        duration: video.duration,
        offsetWidth: video.offsetWidth,
        offsetHeight: video.offsetHeight
      });
      
      // Generate and store unique ID
      const videoId = generateVideoId(video);
      video.dataset.deepdetectId = videoId;
      videoRegistry.set(videoId, video);
      
      addDetectButtonToVideo(video);
      video.dataset.deepdetectInitialized = 'true';
    }
  });
}

// Add detection button directly to video
function addDetectButtonToVideo(video) {
  if (!video || !extensionActive) return;
  
  console.log('DeepSight: Adding detect button to video');
  
  // Create a detect button
  const button = document.createElement('div');
  button.className = 'deepdetect-video-button';
  button.innerHTML = `
    <div class="deepdetect-button-icon">DS</div>
    <div class="deepdetect-button-text">Detect</div>
  `;
  
  // Create a wrapper if needed
  let wrapper = video.parentElement;
  const wrapperStyle = window.getComputedStyle(wrapper);
  if (wrapperStyle.position === 'static') {
    // Create a new wrapper
    wrapper = document.createElement('div');
    wrapper.style.cssText = `
      position: relative !important;
      display: inline-block !important;
      width: ${video.offsetWidth || 100}px !important;
      height: ${video.offsetHeight || 100}px !important;
    `;
    video.parentElement.insertBefore(wrapper, video);
    wrapper.appendChild(video);
  }
  
  // Style the button with important flags to override site styles
  button.style.cssText = `
    position: absolute !important;
    top: 12px !important;
    right: 12px !important;
    background: linear-gradient(135deg, #0A84FF, #5E5CE6) !important;
    color: white !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 13px !important;
    padding: 6px 12px !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    z-index: 2147483647 !important;
    cursor: pointer !important;
    box-shadow: 0 4px 12px rgba(10, 132, 255, 0.35) !important;
    opacity: 1 !important;
    pointer-events: all !important;
    user-select: none !important;
    transition: all 0.2s !important;
    backdrop-filter: blur(4px) !important;
    -webkit-backdrop-filter: blur(4px) !important;
  `;
  
  // Add icon styles
  const icon = button.querySelector('.deepdetect-button-icon');
  if (icon) {
    icon.style.cssText = `
      background: white !important;
      color: #0A84FF !important;
      border-radius: 6px !important;
      width: 20px !important;
      height: 20px !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      font-weight: bold !important;
      font-size: 11px !important;
      margin-right: 6px !important;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    `;
  }
  
  // Add hover effect
  button.addEventListener('mouseenter', () => {
    button.style.transform = 'scale(1.05) translateY(-2px)';
    button.style.boxShadow = '0 6px 16px rgba(10, 132, 255, 0.45) !important';
    button.style.opacity = '1';
  });
  
  button.addEventListener('mouseleave', () => {
    button.style.transform = 'scale(1) translateY(0)';
    button.style.boxShadow = '0 4px 12px rgba(10, 132, 255, 0.35) !important';
    button.style.opacity = '0.95';
  });
  
  // Add click effect
  button.addEventListener('mousedown', () => {
    button.style.transform = 'scale(0.98) translateY(1px)';
    button.style.boxShadow = '0 2px 8px rgba(10, 132, 255, 0.25) !important';
  });
  
  button.addEventListener('mouseup', () => {
    button.style.transform = 'scale(1.05) translateY(-2px)';
    button.style.boxShadow = '0 6px 16px rgba(10, 132, 255, 0.45) !important';
  });
  
  // Add click handler
  button.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!extensionActive) return;
    
    // Send video ID instead of src for better tracking
    const videoId = video.dataset.deepdetectId;
    chrome.runtime.sendMessage({ 
      action: 'openPopup', 
      videoId: videoId,
      videoSrc: video.src || video.currentSrc || 'dynamic'
    });
  });
  
  // Add the button to the wrapper
  wrapper.appendChild(button);
  
  // Initially set slightly transparent
  button.style.opacity = '0.95';
  
  // Store button reference
  video.deepdetectButton = button;
  
  // Update button position on video resize
  const resizeObserver = new ResizeObserver(() => {
    if (wrapper && video) {
      wrapper.style.width = `${video.offsetWidth}px`;
      wrapper.style.height = `${video.offsetHeight}px`;
    }
  });
  
  resizeObserver.observe(video);
}

// Enhanced analyze video function with platform-specific handling
async function analyzeVideo(video) {
  if (isAnalyzing || !extensionActive) {
    console.log('DeepSight: Already analyzing or extension inactive');
    return { error: 'Analysis already in progress or extension inactive' };
  }
  
  isAnalyzing = true;
  currentVideo = video;
  
  console.log('DeepSight: Starting video analysis:', {
    src: video.src,
    currentSrc: video.currentSrc,
    readyState: video.readyState,
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
    duration: video.duration,
    paused: video.paused,
    currentTime: video.currentTime
  });
  
  try {
    // Wait for video to be ready with platform-specific timeout
    const isXCom = window.location.hostname.includes('x.com');
    const loadTimeout = isXCom ? 3000 : 5000;
    
    if (video.readyState < 2) {
      console.log('DeepSight: Waiting for video to load...');
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          console.log('DeepSight: Video loading timeout, proceeding anyway');
          resolve(); // Don't reject, just proceed
        }, loadTimeout);
        
        const onLoadedData = () => {
          clearTimeout(timeout);
          video.removeEventListener('loadeddata', onLoadedData);
          resolve();
        };
        
        video.addEventListener('loadeddata', onLoadedData);
        
        // Try to load the video
        if (video.load) video.load();
      });
    }
    
    // Check if video has valid dimensions
    const width = video.videoWidth || video.offsetWidth;
    const height = video.videoHeight || video.offsetHeight;
    
    if (!width || !height || width < 50 || height < 50) {
      throw new Error(`Video has invalid dimensions: ${width}x${height}`);
    }
    
    console.log(`DeepSight: Video dimensions: ${width}x${height}`);
    
    // Choose frame capture method based on platform and video properties
    let frames;
    if (isXCom || !video.duration || isNaN(video.duration) || video.duration === Infinity) {
      console.log('DeepSight: Using alternative frame capture for X.com/streaming video');
      frames = await captureAlternativeFrames(video, FRAME_SEQUENCE_LENGTH);
    } else {
      console.log('DeepSight: Using standard frame capture');
      frames = await getVideoFrames(video, FRAME_SEQUENCE_LENGTH);
    }
    
    if (!frames || frames.length === 0) {
      throw new Error('No frames could be captured from video');
    }
    
    console.log(`DeepSight: Captured ${frames.length} frames, sending to API...`);
    
    // Send to API
    return await sendFramesToAPI(frames);
    
  } catch (error) {
    console.error('DeepSight: Analysis error:', error);
    return { error: error.message };
  } finally {
    isAnalyzing = false;
    currentVideo = null;
  }
}

// Alternative frame capture for videos without seeking capability (X.com)
async function captureAlternativeFrames(video, numFrames) {
  const frames = [];
  const captureInterval = 200; // ms between captures
  
  console.log('DeepSight: Using alternative frame capture method');
  
  // Try to get the video playing first
  const wasPlaying = !video.paused;
  if (video.paused) {
    try {
      await video.play();
      await new Promise(resolve => setTimeout(resolve, 500)); // Wait for playback to start
    } catch (e) {
      console.log('DeepSight: Could not start video playback, capturing static frames');
    }
  }
  
  for (let i = 0; i < numFrames; i++) {
    const frame = captureVideoFrame(video);
    if (frame) {
      frames.push(frame);
      console.log(`DeepSight: Alternative capture ${i + 1}/${numFrames}`);
    }
    
    // Wait between captures to get different frames
    if (i < numFrames - 1) {
      await new Promise(resolve => setTimeout(resolve, captureInterval));
    }
  }
  
  // Restore original playing state
  if (!wasPlaying && !video.paused) {
    video.pause();
  }
  
  console.log(`DeepSight: Alternative capture got ${frames.length} frames`);
  return frames;
}

// Send frames to API
async function sendFramesToAPI(frames) {
  console.log('DeepSight: Sending frames to API...');
  
  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ frames })
    });
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API request failed: ${response.status} - ${error}`);
    }
    
    const result = await response.json();
    console.log('DeepSight: API response:', result);
    return result;
  } catch (error) {
    console.error('DeepSight: API request failed:', error);
    throw error;
  }
}

// Helper function to capture video frame with enhanced error handling
function captureVideoFrame(video) {
  const canvas = document.createElement('canvas');
  const width = video.videoWidth || video.offsetWidth || 320;
  const height = video.videoHeight || video.offsetHeight || 240;
  
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  
  try {
    ctx.drawImage(video, 0, 0, width, height);
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);
    
    // Verify the frame was captured (not just a blank canvas)
    if (dataURL.length < 1000) {
      console.warn('DeepSight: Captured frame seems too small, might be blank');
      return null;
    }
    
    return dataURL;
  } catch (error) {
    console.error('DeepSight: Error capturing frame:', error);
    // This is likely a CORS error on X.com
    if (error.name === 'SecurityError') {
      console.error('DeepSight: CORS restriction - cannot capture video frames from this source');
    }
    return null;
  }
}

// Standard helper function to get video frames (YouTube/Instagram)
async function getVideoFrames(video, numFrames) {
  const frames = [];
  const currentTime = video.currentTime;
  const wasPlaying = !video.paused;
  
  console.log(`DeepSight: Capturing ${numFrames} frames from video with duration ${video.duration}`);
  
  try {
    if (wasPlaying) {
      video.pause();
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    const duration = video.duration;
    const interval = Math.max(duration / (numFrames + 1), 0.1); // Minimum 0.1s interval
    
    for (let i = 1; i <= numFrames; i++) {
      const targetTime = Math.min(i * interval, duration - 0.1);
      console.log(`DeepSight: Seeking to ${targetTime}s (${i}/${numFrames})`);
      
      video.currentTime = targetTime;
      
      // Wait for seek to complete
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          console.log(`DeepSight: Seek timeout at frame ${i}, continuing...`);
          resolve(); // Don't fail, just continue
        }, 3000);
        
        const onSeeked = () => {
          clearTimeout(timeout);
          video.removeEventListener('seeked', onSeeked);
          resolve();
        };
        
        video.addEventListener('seeked', onSeeked);
      });
      
      // Additional wait to ensure frame is ready
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const frame = captureVideoFrame(video);
      if (frame) {
        frames.push(frame);
        console.log(`DeepSight: Captured frame ${i}/${numFrames}`);
      } else {
        console.warn(`DeepSight: Failed to capture frame ${i}/${numFrames}`);
      }
    }

    console.log(`DeepSight: Successfully captured ${frames.length}/${numFrames} frames`);
    return frames;
    
  } catch (error) {
    console.error('DeepSight: Error in getVideoFrames:', error);
    throw error;
  } finally {
    // Restore original state
    try {
      video.currentTime = currentTime;
      if (wasPlaying) {
        await video.play().catch(e => console.log('DeepSight: Video play error:', e));
      }
    } catch (e) {
      console.warn('DeepSight: Error restoring video state:', e);
    }
  }
}

// Initialize immediately and after DOM content loaded
console.log('DeepSight: Content script loaded');
initializeExtension();

// Also initialize when DOM is ready with platform-specific timing
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    console.log('DeepSight: DOM content loaded');
    // Use longer delay for X.com's dynamic content
    const initDelay = window.location.hostname.includes('x.com') ? 2000 : 1000;
    setTimeout(initializeDetection, initDelay);
  });
} else {
  // DOM is already ready
  const initDelay = window.location.hostname.includes('x.com') ? 2000 : 1000;
  setTimeout(initializeDetection, initDelay);
}