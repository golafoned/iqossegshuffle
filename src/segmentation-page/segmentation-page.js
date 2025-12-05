const MODEL_SIZE = 224;
const MODEL_URL = "/iqos_seg_224_v3.onnx";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const segmentationMaskColorPickerEl = document.getElementById('segmentationMaskColorPicker');

let session = null;
let webcamStream = null;
let animationId = null;
let lastFrameTime = 0;

let currentMaskColor = '#00ff00';
let currentConfidence = 0.5;

// Offscreen canvas for model input resizing
const modelCanvas = document.createElement("canvas");
modelCanvas.width = MODEL_SIZE;
modelCanvas.height = MODEL_SIZE;
const modelCtx = modelCanvas.getContext("2d", { willReadFrequently: true });

function hexToRgbA(hex) {
    if (/^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$/.test(hex)) {
        let r = parseInt(hex.slice(1, 3), 16);
        let g = parseInt(hex.slice(3, 5), 16);
        let b = parseInt(hex.slice(5, 7), 16);
        let a = 255;
        if (hex.length === 9) {
            a = parseInt(hex.slice(7, 9), 16);
        }
        return [r, g, b, a];
    }
    throw new Error('Bad Hex');
}
// ONNX Runtime is loaded from CDN as global 'ort' object

// Configure WASM files location
function configureWasm(ort, ep) {
    if (ep === "wasm") {
        // Load WASM files from same CDN (version 1.19.2 for better compatibility)
        ort.env.wasm.wasmPaths =
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
        // Enable multi-threading for better performance
        ort.env.wasm.numThreads = Math.min(
            4,
            navigator.hardwareConcurrency || 4
        );
        ort.env.wasm.simd = true;
    }
}

// Convert ImageData to NCHW Float32Array [0..1]
function imageToTensorNCHW(imageData) {
    const src = imageData.data;
    const hw = MODEL_SIZE * MODEL_SIZE;
    const out = new Float32Array(1 * 3 * hw);

    let rOff = 0;
    let gOff = hw;
    let bOff = 2 * hw;

    for (let i = 0, p = 0; i < hw; i++, p += 4) {
        out[rOff++] = src[p] / 255.0;
        out[gOff++] = src[p + 1] / 255.0;
        out[bOff++] = src[p + 2] / 255.0;
    }
    return out;
}

// Global buffers for post-processing to avoid GC
const ppSize = MODEL_SIZE * MODEL_SIZE;
const ppVisited = new Uint8Array(ppSize);
const ppStack = new Int32Array(ppSize);
const ppLabels = new Int32Array(ppSize);
const smoothedMask = new Float32Array(ppSize); // Buffer for temporal smoothing

// Post-processing: Keep only the largest connected component
function filterLargestBlob(mask, width, height) {
    // Reset visited buffer
    ppVisited.fill(0);

    let currentLabel = 0;
    const counts = [];

    for (let i = 0; i < ppSize; i++) {
        if (mask[i] > currentConfidence && ppVisited[i] === 0) {
            let stackPtr = 0;
            ppStack[stackPtr++] = i;
            ppVisited[i] = 1;
            ppLabels[i] = currentLabel;
            let count = 0;

            while (stackPtr > 0) {
                const idx = ppStack[--stackPtr];
                count++;

                const x = idx % width;
                const y = (idx / width) | 0;

                // Check 4 neighbors
                // Right
                if (x < width - 1) {
                    const n = idx + 1;
                    if (mask[n] > 0.5 && ppVisited[n] === 0) {
                        ppVisited[n] = 1;
                        ppLabels[n] = currentLabel;
                        ppStack[stackPtr++] = n;
                    }
                }
                // Left
                if (x > 0) {
                    const n = idx - 1;
                    if (mask[n] > 0.5 && ppVisited[n] === 0) {
                        ppVisited[n] = 1;
                        ppLabels[n] = currentLabel;
                        ppStack[stackPtr++] = n;
                    }
                }
                // Down
                if (y < height - 1) {
                    const n = idx + width;
                    if (mask[n] > 0.5 && ppVisited[n] === 0) {
                        ppVisited[n] = 1;
                        ppLabels[n] = currentLabel;
                        ppStack[stackPtr++] = n;
                    }
                }
                // Up
                if (y > 0) {
                    const n = idx - width;
                    if (mask[n] > 0.5 && ppVisited[n] === 0) {
                        ppVisited[n] = 1;
                        ppLabels[n] = currentLabel;
                        ppStack[stackPtr++] = n;
                    }
                }
            }

            counts.push({ label: currentLabel, count: count });
            currentLabel++;
        }
    }

    if (counts.length === 0) {
        // No blobs found, zero out everything
        mask.fill(0);
        return;
    }

    // Find largest blob
    let bestLabel = -1;
    let maxC = 0;
    for (let c of counts) {
        if (c.count > maxC) {
            maxC = c.count;
            bestLabel = c.label;
        }
    }

    // Filter: Zero out everything that is not the best blob
    for (let i = 0; i < ppSize; i++) {
        if (ppVisited[i] === 1) {
            if (ppLabels[i] !== bestLabel) {
                mask[i] = 0;
            }
        } else {
            mask[i] = 0; // Clean up low probability noise
        }
    }
}

// Create mask overlay image data
function createMaskOverlay(mask, alpha = 0.5) {
    const img = new ImageData(MODEL_SIZE, MODEL_SIZE);
    const d = img.data;
    const [r, g, b, a] = hexToRgbA(currentMaskColor);
    for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
        const m = mask[i];
        if (m > 0.5) {
            // Red overlay
            d[p] = r; // R
            d[p + 1] = g; // G
            d[p + 2] = b; // B
            d[p + 3] = a * alpha; // A
        } else {
            d[p + 3] = 0; // Transparent
        }
    }
    return img;
}

// Run inference on current canvas content
async function runInference(sourceElement) {
    if (!session) return;

    const t0 = performance.now();

    // Draw source to model canvas (resize to 224x224)
    modelCtx.drawImage(sourceElement, 0, 0, MODEL_SIZE, MODEL_SIZE);

    const imageData = modelCtx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE);
    const inputData = imageToTensorNCHW(imageData);

    const inputTensor = new ort.Tensor("float32", inputData, [
        1,
        3,
        MODEL_SIZE,
        MODEL_SIZE,
    ]);

    const outputs = await session.run({ input: inputTensor });
    const maskTensor = outputs.mask;
    const mask = maskTensor.data;

    const flat =
        mask.length === MODEL_SIZE * MODEL_SIZE
            ? mask
            : mask.subarray(0, MODEL_SIZE * MODEL_SIZE);

    // Post-processing: Keep only largest blob to remove artifacts
    filterLargestBlob(flat, MODEL_SIZE, MODEL_SIZE);

    // Temporal Smoothing: Blend current frame with previous frames
    // Higher alpha = faster reaction, Lower alpha = smoother but more lag
    const alpha = 0.6;
    for (let i = 0; i < ppSize; i++) {
        smoothedMask[i] = (flat[i] * alpha) + (smoothedMask[i] * (1 - alpha));
    }

    // Create overlay on model canvas
    const overlayData = createMaskOverlay(smoothedMask);

    // Put the mask overlay onto the model canvas
    modelCtx.putImageData(overlayData, 0, 0);

    // Draw the mask overlay from modelCanvas to main canvas (same dimensions as video)
    // This ensures the mask covers the entire screen properly
    if (webcamStream) {
        const videoAspect = video.videoWidth / video.videoHeight;
        const canvasAspect = canvas.width / canvas.height;
        
        let drawWidth, drawHeight, offsetX, offsetY;
        
        if (videoAspect > canvasAspect) {
            drawHeight = canvas.height;
            drawWidth = drawHeight * videoAspect;
            offsetX = (canvas.width - drawWidth) / 2;
            offsetY = 0;
        } else {
            drawWidth = canvas.width;
            drawHeight = drawWidth / videoAspect;
            offsetX = 0;
            offsetY = (canvas.height - drawHeight) / 2;
        }
        
        ctx.drawImage(modelCanvas, offsetX, offsetY, drawWidth, drawHeight);
    } else {
        // For static images, just draw to full canvas
        ctx.drawImage(modelCanvas, 0, 0, canvas.width, canvas.height);
    }

    const t1 = performance.now();
    return (t1 - t0).toFixed(1);
}

// Webcam loop
async function webcamLoop() {
    if (!webcamStream) return;

    const now = performance.now();
    const elapsed = now - lastFrameTime;

    // Resize canvas to match viewport (full screen)
    if (canvas.width !== window.innerWidth || canvas.height !== window.innerHeight) {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    // Draw full resolution video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Run inference (pass video element as source)
    await runInference(video);

    lastFrameTime = now;
    animationId = requestAnimationFrame(webcamLoop);
}

// Start webcam
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "environment" },
            audio: false,
        });
        video.srcObject = webcamStream;
        await video.play();

        statusEl.textContent = "";
        statusEl.classList.add("success");

        lastFrameTime = performance.now();
        webcamLoop();
    } catch (err) {
        statusEl.textContent = `Camera error: ${err.message}`;
        statusEl.classList.add("error");
    }
}

segmentationMaskColorPickerEl.addEventListener('change', (event) => {
    currentMaskColor = event.target.value;
});

segmentationMaskColorPickerEl.value = currentMaskColor;

// Handle preset color buttons
document.querySelectorAll('.preset-color').forEach(button => {
    button.addEventListener('click', (event) => {
        const color = event.target.getAttribute('data-color');
        currentMaskColor = color;
        segmentationMaskColorPickerEl.value = color;
    });
});

// Initialize canvas to full screen
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Initialize
async function main() {
    try {
        statusEl.textContent = "Loading...";

        // Wait for CDN script to load
        while (typeof ort === "undefined") {
            await new Promise((resolve) => setTimeout(resolve, 100));
        }

        const ep = "wasm";
        configureWasm(ort, ep);

        session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: [ep],
        });

        // Auto-start camera
        await startWebcam();
    } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.classList.add("error");
        console.error(err);
    }
}

main();
