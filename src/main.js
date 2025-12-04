const MODEL_SIZE = 224;
const MODEL_URL = "/iqos_seg_224_v3.onnx";

const fileInput = document.getElementById("fileInput");
const webcamBtn = document.getElementById("webcamBtn");
const stopBtn = document.getElementById("stopBtn");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const epEl = document.getElementById("ep");
const fpsEl = document.getElementById("fps");
const inferenceTimeEl = document.getElementById("inferenceTime");

const videoSourceSelector = document.getElementById('videosource');

let session = null;
let ep = "wasm";
let webcamStream = null;
let animationId = null;
let lastFrameTime = 0;

// Offscreen canvas for model input resizing
const modelCanvas = document.createElement("canvas");
modelCanvas.width = MODEL_SIZE;
modelCanvas.height = MODEL_SIZE;
const modelCtx = modelCanvas.getContext("2d", { willReadFrequently: true });

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
        if (mask[i] > 0.5 && ppVisited[i] === 0) {
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
    for (let i = 0, p = 0; i < mask.length; i++, p += 4) {
        const m = mask[i];
        if (m > 0.5) {
            // Red overlay
            d[p] = 255; // R
            d[p + 1] = 0; // G
            d[p + 2] = 0; // B
            d[p + 3] = 255 * alpha; // A
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

    // Put the mask overlay onto the model canvas (replacing the resized video frame)
    modelCtx.putImageData(overlayData, 0, 0);

    // Draw the mask from modelCanvas to main canvas (upscale to source resolution)
    // Note: main canvas should already have the source frame drawn
    ctx.drawImage(modelCanvas, 0, 0, canvas.width, canvas.height);

    const t1 = performance.now();
    const inferenceTime = (t1 - t0).toFixed(1);
    inferenceTimeEl.textContent = `${inferenceTime}ms`;

    return inferenceTime;
}

// Webcam loop
async function webcamLoop() {
    if (!webcamStream) return;

    const now = performance.now();
    const elapsed = now - lastFrameTime;

    // Resize canvas to match video resolution if needed
    if (
        video.videoWidth &&
        (canvas.width !== video.videoWidth ||
            canvas.height !== video.videoHeight)
    ) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }

    // Draw full resolution video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Run inference (pass video element as source)
    await runInference(video);

    // Calculate FPS
    if (elapsed > 0) {
        const fps = Math.round(1000 / elapsed);
        fpsEl.textContent = fps;
    }

    lastFrameTime = now;
    animationId = requestAnimationFrame(webcamLoop);
}

// Start webcam
async function startWebcam() {
    try {
        const selectedDeviceId = videoSourceSelector.value;
        const videoConstraints = selectedDeviceId && selectedDeviceId !== ''
            ? { deviceId: selectedDeviceId, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" };
        
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraints,
            audio: false,
        });
        video.srcObject = webcamStream;
        video.classList.remove("hidden");

        await video.play();

        webcamBtn.classList.add("hidden");
        stopBtn.classList.remove("hidden");
        fileInput.disabled = true;

        statusEl.textContent = "Webcam active - running inference...";
        statusEl.classList.add("success");

        lastFrameTime = performance.now();
        webcamLoop();
    } catch (err) {
        statusEl.textContent = `Webcam error: ${err.message}`;
        statusEl.classList.add("error");
    }
}

async function initializeCameras() {
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
    tempStream.getTracks().forEach(track => track.stop());

    await enumerateCameraDevices();
}

async function enumerateCameraDevices() {
    let cameras = await navigator.mediaDevices.enumerateDevices();

    videoSourceSelector.innerHTML = '';

    for (let i = 0; i < cameras.length; i++) {
        let camera = cameras[i];
        if (camera.kind === 'videoinput') {
            let option = document.createElement('option');
            option.value = camera.deviceId;
            option.text = camera.label || `Camera ${i + 1}`;
            videoSourceSelector.appendChild(option);
        }
    }
}

// Stop webcam
function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach((track) => track.stop());
        webcamStream = null;
    }
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }

    video.classList.add("hidden");
    webcamBtn.classList.remove("hidden");
    stopBtn.classList.add("hidden");
    fileInput.disabled = false;

    statusEl.textContent = "Webcam stopped";
    statusEl.classList.remove("success");
    fpsEl.textContent = "-";
}

// Handle file upload
async function handleFileUpload(file) {
    if (!file) return;

    const bmp = await createImageBitmap(file);

    // Resize canvas to match image resolution
    canvas.width = bmp.width;
    canvas.height = bmp.height;

    // Draw full resolution image
    ctx.drawImage(bmp, 0, 0);

    statusEl.textContent = "Running inference...";
    await runInference(bmp);
    statusEl.textContent = "Inference complete!";
    statusEl.classList.add("success");
}

// Initialize
async function main() {
    try {
        statusEl.textContent = "Loading ONNX Runtime...";

        // Wait for CDN script to load
        while (typeof ort === "undefined") {
            await new Promise((resolve) => setTimeout(resolve, 100));
        }

        await initializeCameras();

        const ep = "wasm";
        epEl.textContent = ep.toUpperCase();

        // Configure WASM paths
        configureWasm(ort, ep);

        statusEl.textContent = "Loading model...";

        session = await ort.InferenceSession.create(MODEL_URL, {
            executionProviders: [ep],
        });

        statusEl.textContent =
            "✅ Model loaded! Upload an image or start webcam.";
        statusEl.classList.add("success");

        // Event listeners
        fileInput.addEventListener("change", async () => {
            const file = fileInput.files?.[0];
            if (file) await handleFileUpload(file);
        });

        webcamBtn.addEventListener("click", startWebcam);
        stopBtn.addEventListener("click", stopWebcam);
        
        // Handle camera selection change
        videoSourceSelector.addEventListener("change", async () => {
            if (webcamStream) {
                stopWebcam();
                await startWebcam();
            }
        });
    } catch (err) {
        statusEl.textContent = `❌ Error: ${err.message}`;
        statusEl.classList.add("error");
        console.error(err);
    }
}

main();
