const canvas = document.getElementById('fashionCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;

function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predictedCategory').innerText = '-';
    document.getElementById('confidenceVal').innerText = 'Draw or pick a garment';
    for (let i = 0; i < 10; i++) {
        document.getElementById(`bar-${i}`).style.width = '0%';
        document.getElementById(`prob-${i}`).innerText = '0%';
    }
}

canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    draw(e);
});
canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('mousemove', draw);

// Touch events
canvas.addEventListener('touchstart', (e) => {
    drawing = true;
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.arc(touch.clientX - rect.left, touch.clientY - rect.top, 12, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
});
canvas.addEventListener('touchend', () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('touchmove', (e) => {
    if (!drawing) return;
    e.preventDefault();
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    ctx.lineWidth = 22;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#ffffff';
    ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
});

function draw(e) {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineWidth = 22;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#ffffff';

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function getPixels28x28() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    const imgData = tempCtx.getImageData(0, 0, 28, 28);
    const pixels = [];
    for (let i = 0; i < imgData.data.length; i += 4) {
        pixels.push(imgData.data[i]);
    }
    return pixels;
}

async function classifyDrawing() {
    const pixels = getPixels28x28();
    document.getElementById('confidenceVal').innerText = 'Analyzing neural network...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pixels: pixels })
        });
        const res = await response.json();
        if (res.error) {
            alert('Error: ' + res.error);
            return;
        }

        document.getElementById('predictedCategory').innerText = res.category;
        document.getElementById('confidenceVal').innerText = `Confidence: ${res.confidence}`;

        for (let i = 0; i < 10; i++) {
            const p = res.probabilities[i];
            const pct = (p * 100).toFixed(1);
            document.getElementById(`bar-${i}`).style.width = pct + '%';
            document.getElementById(`prob-${i}`).innerText = pct + '%';
        }
    } catch (err) {
        console.error(err);
        alert('Prediction failed.');
    }
}

async function loadSampleCategory(catId) {
    clearCanvas();
    try {
        const res = await fetch(`/sample/${catId}`);
        const data = await res.json();
        if (data.pixels) {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            const imgData = tempCtx.createImageData(28, 28);

            for (let i = 0; i < 784; i++) {
                const val = data.pixels[i];
                imgData.data[i * 4] = val;
                imgData.data[i * 4 + 1] = val;
                imgData.data[i * 4 + 2] = val;
                imgData.data[i * 4 + 3] = 255;
            }
            tempCtx.putImageData(imgData, 0, 0);

            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);

            classifyDrawing();
        }
    } catch (err) {
        console.error(err);
    }
}

clearCanvas();
