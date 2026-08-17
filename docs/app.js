// ═══════════════════════════════════════════════════════════════
// Draw & Guess — Canvas Drawing + ONNX Runtime Web Inference
// ═══════════════════════════════════════════════════════════════

const canvas = document.getElementById('drawing-canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear-btn');
const predictionsPanel = document.getElementById('predictions-panel');
const statusText = document.getElementById('status-text');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

let isDrawing = false;
let session = null;
let labels = [];
let debounceTimer = null;
let hasDrawn = false;
let inputName = '';

// ─── Initialize Canvas ────────────────────────────────────────
function initCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 4;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'black';
}
initCanvas();

// ─── Load Model and Labels ────────────────────────────────────
async function loadModelAndLabels() {
    try {
        statusText.textContent = 'Loading model and labels...';

        // Load labels
        const labelsRes = await fetch('labels.json');
        labels = await labelsRes.json();

        // Load ONNX model (with external weights file)
        session = await ort.InferenceSession.create('model.onnx', {
            externalData: [{ path: 'model.onnx.data', data: 'model.onnx.data' }]
        });
        inputName = session.inputNames[0];

        // Warm-up: run a dummy prediction
        const warmupData = new Float32Array(64 * 64).fill(0);
        const warmupTensor = new ort.Tensor('float32', warmupData, [1, 64, 64, 1]);
        await session.run({ [inputName]: warmupTensor });

        loadingOverlay.classList.add('hidden');
        statusText.textContent = 'Model loaded. Ready to guess!';
        statusText.style.color = '#4f8cff';
    } catch (e) {
        console.error('Error loading model:', e);
        loadingText.textContent = 'Error loading model. Check console.';
        statusText.textContent = 'Failed to load model.';
        statusText.style.color = 'red';
    }
}
loadModelAndLabels();

// ─── Drawing Events ───────────────────────────────────────────
function startDrawing(e) {
    isDrawing = true;
    hasDrawn = true;
    draw(e);
}

function stopDrawing() {
    if (!isDrawing) return;
    isDrawing = false;
    ctx.beginPath();

    // Debounced inference
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
        if (session) predict();
    }, 150);
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    let x, y;

    if (e.type.includes('touch')) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Mouse events
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch events
canvas.addEventListener('touchstart', startDrawing, { passive: false });
canvas.addEventListener('touchmove', draw, { passive: false });
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchcancel', stopDrawing);

// ─── Clear Canvas ─────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
    initCanvas();
    hasDrawn = false;
    predictionsPanel.innerHTML = '<div class="placeholder">Draw something!</div>';
});

// ─── Inference ────────────────────────────────────────────────
async function predict() {
    if (!session || !labels.length || !hasDrawn) return;

    // 1. Resize canvas to 64x64 on offscreen canvas
    const offscreen = document.createElement('canvas');
    offscreen.width = 64;
    offscreen.height = 64;
    const offCtx = offscreen.getContext('2d');
    offCtx.fillStyle = 'white';
    offCtx.fillRect(0, 0, 64, 64);
    offCtx.drawImage(canvas, 0, 0, 64, 64);

    // 2. Get pixel data, convert to grayscale, invert, normalize
    const imgData = offCtx.getImageData(0, 0, 64, 64);
    const data = imgData.data;
    const floatData = new Float32Array(64 * 64);

    for (let i = 0; i < floatData.length; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];
        const avg = (r + g + b) / 3;
        // Invert: white canvas (255) -> 0, black strokes (0) -> 1
        floatData[i] = (255 - avg) / 255.0;
    }

    // 3. Create ONNX tensor [1, 64, 64, 1]
    const inputTensor = new ort.Tensor('float32', floatData, [1, 64, 64, 1]);

    // 4. Run inference
    const results = await session.run({ [inputName]: inputTensor });
    const outputName = session.outputNames[0];
    const predictions = results[outputName].data;

    // 5. Get top 3
    const top3 = getTopK(predictions, 3);
    renderPredictions(top3);
}

function getTopK(predictions, k) {
    const arr = Array.from(predictions).map((prob, index) => ({ prob, index }));
    arr.sort((a, b) => b.prob - a.prob);
    return arr.slice(0, k);
}

function renderPredictions(top3) {
    predictionsPanel.innerHTML = '';

    top3.forEach((item, i) => {
        const label = labels[item.index] || 'Unknown';
        const displayLabel = label.charAt(0).toUpperCase() + label.slice(1);
        const probPct = (item.prob * 100).toFixed(1);

        const html = `
            <div class="prediction-item">
                <div class="prediction-header">
                    <span class="prediction-label">#${i + 1} ${displayLabel}</span>
                    <span class="prediction-score">${probPct}%</span>
                </div>
                <div class="bar-container">
                    <div class="bar-fill" style="width: ${probPct}%"></div>
                </div>
            </div>
        `;
        predictionsPanel.innerHTML += html;
    });
}
