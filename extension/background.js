chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyzeVideo",
    title: "Analyze with DeepDetect",
    contexts: ["video"]  // Create context menu for video elements
  });
  
  // Initialize extension as inactive
  chrome.storage.local.set({ isActive: false });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyzeVideo") {
    // Check if extension is active
    chrome.storage.local.get('isActive', (data) => {
      if (data.isActive) {
        chrome.tabs.sendMessage(tab.id, {
          action: "analyzeVideo",
          videoSrc: info.srcUrl
        });
      } else {
        // Notify user to open extension first
        chrome.action.setBadgeText({ text: "!" });
        chrome.action.setBadgeBackgroundColor({ color: "#FF0000" });
        
        // Clear badge after 3 seconds
        setTimeout(() => {
          chrome.action.setBadgeText({ text: "" });
        }, 3000);
      }
    });
  }
});

// Reset extension active state when popup closes
chrome.runtime.onConnect.addListener(function(port) {
  if (port.name === "popup") {
    // Set extension as active when popup opens
    chrome.storage.local.set({ isActive: true });
    
    port.onDisconnect.addListener(function() {
      // Set extension as inactive when popup closes
      chrome.storage.local.set({ isActive: false });
    });
  }
});
