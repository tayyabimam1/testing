// Create a connection to the background script
const port = chrome.runtime.connect({ name: "popup" });
const API_ENDPOINT = 'http://localhost:8000';

// Global variable to store current video info
let currentVideoInfo = null;
let serverStatus = false;

// Check if server is running
function checkServer() {
  const statusEl = document.createElement('div');
  statusEl.className = 'server-status';
  statusEl.style.position = 'absolute';
  statusEl.style.top = '4px';
  statusEl.style.right = '4px';
  statusEl.style.width = '10px';
  statusEl.style.height = '10px';
  statusEl.style.borderRadius = '50%';
  statusEl.style.background = '#555';
  document.body.appendChild(statusEl);
  
  fetch(`${API_ENDPOINT}/api/test`)
    .then(response => {
      if (response.ok) {
        console.log('DeepSight: Server is running');
        statusEl.style.background = '#34C759';
        serverStatus = true;
      } else {
        console.log('DeepSight: Server returned an error');
        statusEl.style.background = '#FF2D55';
        serverStatus = false;
      }
    })
    .catch(error => {
      console.log('DeepDetect: Server is not running', error);
      statusEl.style.background = '#FF2D55';
      serverStatus = false;
      
      // Show server not running message
      const resultEl = document.querySelector('.deepdetect-result');
      if (resultEl) {
        resultEl.textContent = 'Server not running';
        resultEl.className = 'deepdetect-result error';
      }
    });
}

// Helper function to safely send messages to content script
function sendMessageToContentScript(tabId, message, callback) {
  try {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      // Check if there was an error (like no content script listening)
      if (chrome.runtime.lastError) {
        console.log('Content script not ready:', chrome.runtime.lastError.message);
        if (callback) callback(null);
        return;
      }
      if (callback) callback(response);
    });
  } catch (error) {
    console.log('Error sending message to content script:', error);
    if (callback) callback(null);
  }
}

// Set extension as active when popup opens
chrome.storage.local.set({ isActive: true }, () => {
  console.log('Extension activated');
  
  // Check if server is running
  checkServer();
  
  // Get current tab
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      // Notify content script to activate (no response expected)
      sendMessageToContentScript(tabs[0].id, { action: "activate" });
      
      // Add a small delay before requesting video info to ensure content script is ready
      setTimeout(() => {
        // Request current video info
        sendMessageToContentScript(tabs[0].id, { action: "getVideoInfo" }, (response) => {
          if (response && response.hasVideo) {
            currentVideoInfo = response.videoInfo;
            updateUIWithVideo(response.videoInfo);
            setupAnalyzeButton();
          } else {
            // No video found or content script not ready
            updateUIWithVideo(null);
            setupAnalyzeButton();
          }
        });
      }, 100);
    }
  });
});

// Set extension as inactive when popup closes
window.addEventListener('unload', () => {
  chrome.storage.local.set({ isActive: false }, () => {
    console.log('Extension deactivated');
    // Notify content script (no response expected)
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        sendMessageToContentScript(tabs[0].id, { action: "deactivate" });
      }
    });
  });
  if (port) {
    port.disconnect();
  }
});

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analysisResult") {
    updateUIWithResult(message.result);
    // Send response to acknowledge receipt
    sendResponse({ received: true });
  }
  return true; // Keep message channel open for async response
});

// Setup analyze button click handler
function setupAnalyzeButton() {
  const analyzeBtn = document.querySelector('.deepdetect-analyze-btn');
  if (analyzeBtn) {
    // Remove any existing event listeners
    const newBtn = analyzeBtn.cloneNode(true);
    analyzeBtn.parentNode.replaceChild(newBtn, analyzeBtn);
    
    if (currentVideoInfo) {
      newBtn.disabled = false;
      newBtn.textContent = 'Detect DeepFake';
    } else {
      newBtn.disabled = true;
      newBtn.textContent = 'Select Video First';
    }
    
    newBtn.addEventListener('click', () => {
      if (!newBtn.disabled && currentVideoInfo) {
        // Check if server is running
        if (!serverStatus) {
          // Check server again
          fetch(`${API_ENDPOINT}/api/test`)
            .then(response => {
              if (response.ok) {
                serverStatus = true;
                startAnalysis();
              } else {
                showServerError();
              }
            })
            .catch(error => {
              showServerError();
            });
        } else {
          startAnalysis();
        }
      }
    });
  }
  
  function showServerError() {
    const resultEl = document.querySelector('.deepdetect-result');
    const alertEl = document.querySelector('.deepdetect-alert');
    if (resultEl) {
      resultEl.textContent = 'Server not running';
      resultEl.className = 'deepdetect-result error';
    }
    if (alertEl) {
      alertEl.textContent = 'DeepSight Alert: Start Python server first';
    }
  }
  
  function startAnalysis() {
    // Update UI to analyzing state
    const resultEl = document.querySelector('.deepdetect-result');
    const confidenceEl = document.querySelector('.confidence-value');
    const barEl = document.querySelector('.confidence-bar');
    const alertEl = document.querySelector('.deepdetect-alert');
    const analyzeBtn = document.querySelector('.deepdetect-analyze-btn');
    
    if (resultEl) resultEl.textContent = 'Analyzing...';
    if (resultEl) resultEl.className = 'deepdetect-result analyzing';
    if (confidenceEl) confidenceEl.textContent = '0%';
    if (barEl) barEl.style.width = '0%';
    if (barEl) barEl.className = 'confidence-bar';
    if (alertEl) alertEl.textContent = 'DeepDetect Alert: Analyzing video content';
    if (analyzeBtn) {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = 'Analyzing...';
      
      // Add a loading animation to the button
      analyzeBtn.style.background = 'linear-gradient(135deg, #0A84FF, #5E5CE6)';
      analyzeBtn.style.backgroundSize = '200% 200%';
      analyzeBtn.style.animation = 'gradient 1.5s ease infinite';
    }
    
    // Request analysis from content script
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        sendMessageToContentScript(tabs[0].id, {
          action: "analyzeVideoInPopup",
          videoSrc: currentVideoInfo.src
        });
      }
    });
  }
}

// Update UI when a video is selected
function updateUIWithVideo(videoInfo) {
  const previewEl = document.querySelector('.deepdetect-video-preview');
  const analyzeBtn = document.querySelector('.deepdetect-analyze-btn');
  const resultEl = document.querySelector('.deepdetect-result');
  
  if (videoInfo && videoInfo.thumbnail) {
    if (previewEl) {
      previewEl.innerHTML = `<img src="${videoInfo.thumbnail}" style="max-width: 100%; max-height: 100%; border-radius: 8px; object-fit: contain;">`;
    }
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Detect DeepFake';
    }
    if (resultEl) {
      resultEl.textContent = 'Ready to analyze';
      resultEl.className = 'deepdetect-result';
    }
  } else {
    if (previewEl) {
      previewEl.textContent = 'Select a video on the page to analyze';
    }
    if (analyzeBtn) {
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = 'Select Video First';
    }
    if (resultEl) {
      resultEl.textContent = 'No video selected';
      resultEl.className = 'deepdetect-result';
    }
  }
}

// Update UI with analysis results
function updateUIWithResult(result) {
  const resultEl = document.querySelector('.deepdetect-result');
  const confidenceEl = document.querySelector('.confidence-value');
  const barEl = document.querySelector('.confidence-bar');
  const bannerEl = document.querySelector('.deepdetect-banner');
  const alertEl = document.querySelector('.deepdetect-alert');
  const analyzeBtn = document.querySelector('.deepdetect-analyze-btn');
  
  // Reset button style
  if (analyzeBtn) {
    analyzeBtn.style.background = 'linear-gradient(135deg, #0A84FF, #5E5CE6)';
    analyzeBtn.style.backgroundSize = '100% 100%';
    analyzeBtn.style.animation = 'none';
  }
  
  if (result.error) {
    if (resultEl) {
      resultEl.textContent = `Error: ${result.error}`;
      resultEl.className = 'deepdetect-result error';
    }
    if (alertEl) alertEl.textContent = 'DeepDetect Alert: Analysis failed';
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Retry Analysis';
    }
    return;
  }
  
  const isFake = result.is_fake;
  const confidence = result.confidence?.toFixed(1) || 0;
  
  if (resultEl) {
    resultEl.textContent = isFake ? 'FAKE' : 'REAL';
    resultEl.className = `deepdetect-result ${isFake ? 'fake' : 'real'}`;
  }
  if (confidenceEl) confidenceEl.textContent = `${confidence}%`;
  if (barEl) {
    barEl.style.width = `${confidence}%`;
    barEl.className = `confidence-bar ${isFake ? 'fake' : 'real'}`;
  }
  if (bannerEl) bannerEl.className = `deepdetect-banner ${isFake ? 'fake' : 'real'}`;
  
  if (alertEl) {
    alertEl.textContent = isFake 
      ? `DeepSight Alert: This video contains AI generated or manipulated content (${confidence}% confidence)`
      : `DeepSight Alert: No AI manipulation detected (${confidence}% confidence)`;
  }
  
  if (analyzeBtn) {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Again';
  }
}