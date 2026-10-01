// NOTE: Change this URL to your live Render/Railway backend URL once deployed.
const API_URL = "https://chroma-proof.onrender.com"; 

async function init() {
    // 1. Load Kit Options
    try {
        const kitRes = await fetch(`${API_URL}/api/kits`);
        const kits = await kitRes.json();
        const kitSelect = document.getElementById("kitSelect");
        kitSelect.innerHTML = kits.map(k => `<option value="${k.id}">${k.name}</option>`).join("");
    } catch (e) {
        console.error("Failed to load kits. API offline?");
    }

    // 2. Setup Camera
    const video = document.getElementById('webcam');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
        video.srcObject = stream;
    } catch (err) {
        console.warn("Camera access denied or unavailable.");
    }

    // --- SHARED PROCESSING FUNCTION ---
    // Both the camera snapshot and the gallery upload route through here
    async function processBlob(blob, sourceText) {
        const formData = new FormData();
        formData.append('image', blob, 'capture.jpg');
        formData.append('operator_id', document.getElementById('operatorId').value);
        formData.append('kit_id', document.getElementById('kitSelect').value);
        formData.append('location', `GPS: Browser Enabled (${sourceText})`);

        const captureBtn = document.getElementById('btnCapture');
        const galleryBtn = document.getElementById('btnGallery');
        
        // Show loading state
        captureBtn.innerText = "Processing...";
        galleryBtn.innerText = "Processing...";
        
        try {
            const res = await fetch(`${API_URL}/api/process`, { method: "POST", body: formData });
            const data = await res.json();
            
            if (data.status === "SUCCESS") {
                displayResult(data.record);
            } else {
                alert("Analysis failed. Quality threshold might not be met.");
            }
        } catch (err) {
            alert("Processing failed. Check API connection.");
        } finally {
            // Restore buttons
            captureBtn.innerText = "Snap & Analyze";
            galleryBtn.innerText = "📁 Upload from Gallery";
        }
    }

    // 3. Handle Live Camera Capture
    document.getElementById('btnCapture').addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        canvas.toBlob((blob) => {
            processBlob(blob, "Live Capture");
        }, 'image/jpeg');
    });

    // 4. Handle Gallery Upload
    const galleryInput = document.getElementById('galleryInput');
    
    // Trigger the hidden file input when the user clicks the stylized button
    document.getElementById('btnGallery').addEventListener('click', () => {
        galleryInput.click();
    });

    // When the user selects a photo from their camera roll, process it
    galleryInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            processBlob(file, "Gallery Upload");
            event.target.value = ""; // Reset the input so they can upload the same file again if needed
        }
    });
}

function displayResult(record) {
    const box = document.getElementById('resultBox');
    box.style.display = 'block';
    box.className = `result-box ${record.result}`;
    
    box.innerHTML = `
        <h2>${record.result}</h2>
        <p><strong>Confidence:</strong> ${record.confidence}</p>
        <p><strong>Explanation:</strong> ${record.explanation}</p>
        <div class="meta-data">
            <p>ID: ${record.record_id}</p>
            <p>Hash: ${record.image_sha256}</p>
        </div>
    `;
}

// Start app
init();
