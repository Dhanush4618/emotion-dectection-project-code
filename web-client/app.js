class EmotionDetectionApp {
    constructor() {
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.ws = null;
        this.stream = null;
        this.isRunning = false;
        this.frameCount = 0;
        this.lastFpsTime = Date.now();
        this.fps = 0;
        this.processingInterval = null;
        this.consentAccepted = false;
        
        // DOM elements
        this.wsStatus = document.getElementById('wsStatus');
        this.cameraStatus = document.getElementById('cameraStatus');
        this.processingStatus = document.getElementById('processingStatus');
        this.fpsCounter = document.getElementById('fpsCounter');
        this.detectionsList = document.getElementById('detectionsList');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.consentBanner = document.getElementById('consentBanner');
        
        // Initialize
        this.init();
    }
    
    init() {
        // Set canvas size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());
        
        // Check if consent was already accepted
        if (localStorage.getItem('emotionDetectionConsent') === 'true') {
            this.consentAccepted = true;
            if (this.consentBanner) {
                this.consentBanner.style.display = 'none';
            }
        }
    }
    
    resizeCanvas() {
        const videoSection = this.video.parentElement;
        const rect = videoSection.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }
    
    async acceptConsent() {
        this.consentAccepted = true;
        localStorage.setItem('emotionDetectionConsent', 'true');
        if (this.consentBanner) {
            this.consentBanner.style.display = 'none';
        }
    }
    
    async startDetection() {
        if (!this.consentAccepted) {
            alert('Please accept the consent banner first.');
            return;
        }
        
        try {
            await this.startCamera();
            await this.connectWebSocket();
            this.startProcessing();
            this.updateUI('running');
        } catch (error) {
            console.error('Failed to start detection:', error);
            alert('Failed to start detection. Please check your camera permissions and try again.');
        }
    }
    
    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });
            
            this.video.srcObject = this.stream;
            this.cameraStatus.textContent = 'Connected';
            this.cameraStatus.className = 'status-value status-connected';
            
            // Wait for video to be ready
            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    resolve();
                };
            });
        } catch (error) {
            console.error('Camera access error:', error);
            throw new Error('Failed to access camera');
        }
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.wsStatus.textContent = 'Connected';
                this.wsStatus.className = 'status-value status-connected';
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(event);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.wsStatus.textContent = 'Disconnected';
                this.wsStatus.className = 'status-value status-disconnected';
                this.isRunning = false;
                this.updateUI('stopped');
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(new Error('WebSocket connection failed'));
            };
        });
    }
    
    startProcessing() {
        this.isRunning = true;
        this.processingStatus.textContent = 'Active';
        this.processingStatus.className = 'status-value status-connected';
        
        // Process frames every 200ms (5 FPS for server processing)
        this.processingInterval = setInterval(() => {
            if (this.isRunning && this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.captureAndSendFrame();
            }
        }, 200);
        
        // Update FPS counter
        this.fpsInterval = setInterval(() => {
            this.updateFps();
        }, 1000);
    }
    
    captureAndSendFrame() {
        if (!this.video.videoWidth || !this.video.videoHeight) return;
        
        // Draw video frame to canvas
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Convert canvas to base64
        const dataUrl = this.canvas.toDataURL('image/jpeg', 0.8);
        
        // Send to server
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(dataUrl);
        }
        
        this.frameCount++;
    }
    
    handleWebSocketMessage(event) {
        try {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                console.error('Server error:', data.error);
                return;
            }
            
            if (data.detections) {
                this.drawDetections(data.detections);
                this.updateDetectionsList(data.detections);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    drawDetections(detections) {
        // Clear previous drawings
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw video frame again
        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        detections.forEach(detection => {
            const [x1, y1, x2, y2] = detection.bbox;
            const emotion = detection.emotion;
            
            if (emotion.error) return;
            
            // Draw bounding box
            this.ctx.strokeStyle = '#00ff00';
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            
            // Draw emotion label
            this.ctx.fillStyle = '#00ff00';
            this.ctx.font = 'bold 18px Arial';
            this.ctx.textAlign = 'center';
            
            const label = `${emotion.emotion} (${Math.round(emotion.confidence * 100)}%)`;
            const textWidth = this.ctx.measureText(label).width;
            
            // Draw background for text
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            this.ctx.fillRect(x1 - 5, y1 - 30, textWidth + 10, 25);
            
            // Draw text
            this.ctx.fillStyle = '#00ff00';
            this.ctx.fillText(label, x1 + textWidth / 2, y1 - 10);
        });
    }
    
    updateDetectionsList(detections) {
        if (detections.length === 0) {
            this.detectionsList.innerHTML = `
                <div class="loading">
                    <i class="fas fa-search"></i> No faces detected...
                </div>
            `;
            return;
        }
        
        const detectionsHtml = detections.map((detection, index) => {
            const emotion = detection.emotion;
            if (emotion.error) return '';
            
            const confidencePercent = Math.round(emotion.confidence * 100);
            const emotionEmoji = this.getEmotionEmoji(emotion.emotion);
            const isDemo = emotion.demo_mode ? ' (Demo)' : '';
            
            return `
                <div class="detection-item">
                    <div class="emotion-label">
                        <span class="emoji">${emotionEmoji}</span>
                        <span>${emotion.emotion}${isDemo}</span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                    </div>
                    <div class="confidence-text">
                        <span><i class="fas fa-percentage"></i> ${confidencePercent}%</span>
                        <span><i class="fas fa-clock"></i> ${emotion.inference_time_ms}ms</span>
                    </div>
                </div>
            `;
        }).join('');
        
        this.detectionsList.innerHTML = detectionsHtml;
    }
    
    getEmotionEmoji(emotion) {
        const emojiMap = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'disgust': '🤢',
            'neutral': '😐'
        };
        return emojiMap[emotion] || '😐';
    }
    
    updateFps() {
        const now = Date.now();
        this.fps = Math.round((this.frameCount * 1000) / (now - this.lastFpsTime));
        this.fpsCounter.innerHTML = `<i class="fas fa-tachometer-alt"></i> FPS: ${this.fps}`;
        this.frameCount = 0;
        this.lastFpsTime = now;
    }
    
    stopDetection() {
        this.isRunning = false;
        
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
        
        if (this.fpsInterval) {
            clearInterval(this.fpsInterval);
            this.fpsInterval = null;
        }
        
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.updateUI('stopped');
    }
    
    updateUI(state) {
        if (state === 'running') {
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
        } else {
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            this.cameraStatus.textContent = 'Not Started';
            this.cameraStatus.className = 'status-value status-disconnected';
            this.processingStatus.textContent = 'Idle';
            this.processingStatus.className = 'status-value status-disconnected';
            this.wsStatus.textContent = 'Disconnected';
            this.wsStatus.className = 'status-value status-disconnected';
            this.fpsCounter.innerHTML = '<i class="fas fa-tachometer-alt"></i> FPS: 0';
            
            // Clear canvas
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Clear detections list
            this.detectionsList.innerHTML = `
                <div class="loading">
                    <i class="fas fa-search"></i> No faces detected yet...
                </div>
            `;
        }
    }
    
    resetApp() {
        this.stopDetection();
        this.consentAccepted = false;
        localStorage.removeItem('emotionDetectionConsent');
        if (this.consentBanner) {
            this.consentBanner.style.display = 'block';
        }
    }
}

// Initialize app when page loads
let app;

document.addEventListener('DOMContentLoaded', () => {
    app = new EmotionDetectionApp();
});

// Global functions for HTML buttons
function acceptConsent() {
    if (app) app.acceptConsent();
}

function startDetection() {
    if (app) app.startDetection();
}

function stopDetection() {
    if (app) app.stopDetection();
}

function resetApp() {
    if (app) app.resetApp();
} 