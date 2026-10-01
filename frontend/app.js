// NOTE: Change this URL to your Render/Railway backend URL once deployed.
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
        alert("Camera access denied or unavailable.");
    }

    // 3. Handle Capture & Analysis
    document.getElementById('btnCapture').addEventListener('click', () => {
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'capture.jpg');
            formData.append('operator_id', document.getElementById('operatorId').value);
            formData.append('kit_id', document.getElementById('kitSelect').value);
            formData.append('location', "GPS: Browser Enabled");

            document.getElementById('btnCapture').innerText = "Processing...";
            
            try {
                const res = await fetch(`${API_URL}/api/process`, { method: "POST", body: formData });
                const data = await res.json();
                
                if (data.status === "SUCCESS") {
                    displayResult(data.record);
                }
            } catch (err) {
                alert("Processing failed. Check API connection.");
            } finally {
                document.getElementById('btnCapture').innerText = "Snap & Analyze";
            }
        }, 'image/jpeg');
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
