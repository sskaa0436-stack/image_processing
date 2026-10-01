// ==========================================
// 3. VARIABEL & INISIALISASI MODEL AI
// ==========================================
let model = null;
let faceDetector = null;
let faceLandmarker = null;
let lastFaceSeenAt = 0;
const FACE_RESET_TIMEOUT_MS = 3000;
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const overlayMsg = document.getElementById('overlay-msg');
const overlayText = document.getElementById('overlay-text');
const resultBox = document.getElementById('resultBox');
const totalDetectionsEl = document.getElementById('total-detections');
const faceCountEl = document.getElementById('face-count');
const objectCountEl = document.getElementById('object-count');
const statsBodyEl = document.getElementById('stats-body');
const faceTableBodyEl = document.getElementById('face-table-body');
const faceChartCanvas = document.getElementById('face-chart');
const faceChartStatusEl = document.getElementById('face-chart-status');
const faceChartCurrentEl = document.getElementById('face-chart-current');
const faceChartCtx = faceChartCanvas ? faceChartCanvas.getContext('2d') : null;

const categoryCounts = new Map();
let totalDetections = 0;
let hasTrackedFace = false;
let lastFaceSampleAt = 0;
const faceHistory = [];

function renderCategoryStats() {
    const rows = Array.from(categoryCounts.entries()).sort((a, b) => b[1] - a[1]);
    const maxValue = rows.length > 0 ? Math.max(...rows.map(([, value]) => value)) : 1;

    if (!statsBodyEl) return;

    if (rows.length === 0) {
        statsBodyEl.innerHTML = '<tr><td colspan="3">Belum ada data</td></tr>';
        return;
    }

    statsBodyEl.innerHTML = rows.map(([category, count]) => {
        const percent = (count / maxValue) * 100;
        return `
            <tr>
                <td>${category}</td>
                <td>${count}</td>
                <td class="chart-cell">
                    <div class="bar-track"><span class="bar-fill" style="width: ${percent}%"></span></div>
                </td>
            </tr>
        `;
    }).join('');
}

function updateStatCards() {
    const faceTotal = categoryCounts.get('wajah') || 0;
    const objectTotal = Array.from(categoryCounts.entries())
        .filter(([label]) => label !== 'wajah')
        .reduce((sum, [, count]) => sum + count, 0);

    if (totalDetectionsEl) totalDetectionsEl.textContent = String(totalDetections);
    if (faceCountEl) faceCountEl.textContent = String(faceTotal);
    if (objectCountEl) objectCountEl.textContent = String(objectTotal);
}

function registerDetectedCategories(detectedTargets, faceCount = 0) {
    categoryCounts.clear();

    if (faceCount > 0) {
        categoryCounts.set('wajah', faceCount);
    }

    detectedTargets.forEach(item => {
        const key = item.class || 'unknown';
        categoryCounts.set(key, (categoryCounts.get(key) || 0) + 1);
    });

    totalDetections = Array.from(categoryCounts.values()).reduce((sum, value) => sum + value, 0);
    renderCategoryStats();
    updateStatCards();
}

async function initTensorFlow() {
    if (typeof tf === 'undefined') {
        throw new Error('TensorFlow.js belum dimuat.');
    }

    await tf.ready();

    try {
        if (tf.setBackend) {
            await tf.setBackend('webgl');
        }
    } catch (backendError) {
        console.warn('WebGL backend unavailable, using default backend:', backendError);
    }
}

async function loadCocoSsdModel() {
    if (typeof cocoSsd === 'undefined') {
        throw new Error('COCO-SSD belum dimuat.');
    }

    return cocoSsd.load({ base: 'lite_mobilenet_v2' });
}

async function initFaceLandmarker() {
    try {
        const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm');
        const fileset = await vision.FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );
        const options = {
            baseOptions: {
                modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                delegate: 'GPU'
            },
            runningMode: 'VIDEO',
            numFaces: 5,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false
        };
        try {
            faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, options);
        } catch (gpuError) {
            console.warn('GPU MediaPipe tidak tersedia, memakai CPU:', gpuError);
            delete options.baseOptions.delegate;
            faceLandmarker = await vision.FaceLandmarker.createFromOptions(fileset, options);
        }
        console.log('MediaPipe Face Landmarker siap digunakan.');
    } catch (error) {
        console.warn('MediaPipe Face Landmarker gagal dimuat, mencoba FaceDetector:', error);
        if (typeof FaceDetector !== 'undefined') {
            try {
                faceDetector = new FaceDetector({ fastMode: true, maxDetectedFaces: 5 });
            } catch (fallbackError) {
                console.warn('FaceDetector fallback tidak tersedia:', fallbackError);
            }
        }
    }
}

async function loadModel() {
    if (overlayText) overlayText.textContent = 'Memuat Model AI (COCO-SSD + Face Landmarks)...';
    try {
        await initTensorFlow();
        model = await loadCocoSsdModel();
        await initFaceLandmarker();
        console.log('Model AI berhasil dimuat:', model);
        if (overlayMsg) overlayMsg.style.display = 'none';
    } catch (err) {
        console.error('Gagal memuat model:', err);
        if (overlayText) overlayText.textContent = 'Gagal memuat Model AI. Periksa koneksi atau browser.';
    }
}

function hideCameraOverlay() {
    if (overlayMsg) {
        overlayMsg.style.display = 'none';
    }
}

function showCameraOverlay(message) {
    if (overlayMsg) {
        overlayMsg.style.display = 'flex';
    }
    if (overlayText && message) {
        overlayText.textContent = message;
    }
}

// ==========================================
// 4. LABEL NORMALIZATION & THROTTLING CONFIG
// ==========================================
const OBJECT_CONFIDENCE_THRESHOLD = 0.60;

const LABEL_ALIAS = {
    person: 'orang',
    bicycle: 'sepeda',
    car: 'mobil',
    motorcycle: 'sepeda motor',
    airplane: 'pesawat',
    bus: 'bus',
    train: 'kereta',
    truck: 'truk',
    boat: 'perahu',
    'traffic light': 'lampu lalu lintas',
    'fire hydrant': 'hidran',
    'stop sign': 'rambu berhenti',
    'parking meter': 'meter parkir',
    bench: 'bangku',
    bird: 'burung',
    cat: 'kucing',
    dog: 'anjing',
    horse: 'kuda',
    sheep: 'domba',
    cow: 'sapi',
    elephant: 'gajah',
    bear: 'beruang',
    zebra: 'zebra',
    giraffe: 'jerapah',
    backpack: 'ransel',
    umbrella: 'payung',
    handbag: 'tas tangan',
    tie: 'dasi',
    suitcase: 'koper',
    frisbee: 'frisbee',
    skis: 'ski',
    snowboard: 'papan seluncur salju',
    'sports ball': 'bola',
    kite: 'layang-layang',
    'baseball bat': 'tongkat bisbol',
    'baseball glove': 'sarung tangan bisbol',
    skateboard: 'skateboard',
    surfboard: 'papan selancar',
    'tennis racket': 'raket tenis',
    bottle: 'botol',
    'wine glass': 'gelas anggur',
    cup: 'gelas',
    fork: 'garpu',
    knife: 'pisau',
    spoon: 'sendok',
    bowl: 'mangkuk',
    banana: 'pisang',
    apple: 'apel',
    sandwich: 'roti lapis',
    orange: 'jeruk',
    broccoli: 'brokoli',
    carrot: 'wortel',
    'hot dog': 'hot dog',
    pizza: 'pizza',
    donut: 'donat',
    cake: 'kue',
    chair: 'kursi',
    couch: 'sofa',
    'potted plant': 'tanaman pot',
    bed: 'tempat tidur',
    'dining table': 'meja makan',
    toilet: 'toilet',
    tv: 'televisi',
    laptop: 'laptop',
    mouse: 'mouse',
    remote: 'remote',
    keyboard: 'keyboard',
    'cell phone': 'smartphone',
    'cellphone': 'smartphone',
    'phone': 'smartphone',
    microwave: 'microwave',
    oven: 'oven',
    toaster: 'pemanggang roti',
    sink: 'wastafel',
    refrigerator: 'kulkas',
    book: 'buku',
    clock: 'jam',
    vase: 'vas',
    scissors: 'gunting',
    'teddy bear': 'boneka beruang',
    'hair drier': 'pengering rambut',
    'toothbrush': 'sikat gigi'
};

function getFriendlyLabel(className) {
    const label = String(className || '').trim();
    const key = label.toLowerCase();
    return LABEL_ALIAS[key] || label || 'objek';
}

function getFriendlyItemName(item) {
    const className = item && (item.class || item.name || item.rawClass || '');
    return getFriendlyLabel(className);
}

function normalizeStoredItemNames(value) {
    let normalized = String(value || '');
    const aliases = Object.entries(LABEL_ALIAS).sort(([left], [right]) => right.length - left.length);

    aliases.forEach(([rawLabel, friendlyLabel]) => {
        let searchFrom = 0;
        let matchIndex = normalized.toLowerCase().indexOf(rawLabel, searchFrom);

        while (matchIndex !== -1) {
            const matchEnd = matchIndex + rawLabel.length;
            const leftChar = normalized[matchIndex - 1] || '';
            const rightChar = normalized[matchEnd] || '';
            const leftBoundary = !/[a-z0-9]/i.test(leftChar);
            const rightBoundary = !/[a-z0-9]/i.test(rightChar);

            if (leftBoundary && rightBoundary) {
                normalized = normalized.slice(0, matchIndex) + friendlyLabel + normalized.slice(matchEnd);
                searchFrom = matchIndex + friendlyLabel.length;
            } else {
                searchFrom = matchEnd;
            }
            matchIndex = normalized.toLowerCase().indexOf(rawLabel, searchFrom);
        }
    });

    return normalized;
}

let lastSaveTime = 0;
let isSaveInProgress = false;
const SAVE_THROTTLE_MS = 3000; // Simpan maksimal setiap 3 detik

function getColorNameFromRGB(r, g, b) {
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const delta = max - min;

    if (delta < 25) {
        if (max < 80) return 'hitam';
        if (max > 200) return 'putih';
        return 'abu-abu';
    }

    const hue = (() => {
        if (max === r) return ((g - b) / delta) % 6;
        if (max === g) return (b - r) / delta + 2;
        return (r - g) / delta + 4;
    })() * 60;

    if (hue >= 330 || hue < 20) return 'merah';
    if (hue < 55) return 'kuning';
    if (hue < 150) return 'hijau';
    if (hue < 210) return 'biru';
    if (hue < 260) return 'ungu';
    if (hue < 330) return 'pink';
    return 'merah';
}

function getObjectSizeFromArea(area, frameArea) {
    const ratio = area / frameArea;
    if (ratio < 0.02) return 'kecil';
    if (ratio < 0.08) return 'sedang';
    return 'besar';
}

function getObjectShape(width, height) {
    const ratio = width / height;
    if (Math.abs(ratio - 1) <= 0.2) return 'bundar';
    if (ratio > 1.35) return 'persegi panjang';
    if (ratio < 0.75) return 'tegak';
    return 'oval';
}

function analyzeObjectPattern(x, y, width, height) {
    const sampleCanvas = document.createElement('canvas');
    const sampleCtx = sampleCanvas.getContext('2d');
    sampleCanvas.width = Math.max(1, Math.floor(width));
    sampleCanvas.height = Math.max(1, Math.floor(height));

    sampleCtx.drawImage(video, x, y, width, height, 0, 0, sampleCanvas.width, sampleCanvas.height);
    const { data } = sampleCtx.getImageData(0, 0, sampleCanvas.width, sampleCanvas.height);

    let totalR = 0;
    let totalG = 0;
    let totalB = 0;
    let count = 0;

    for (let i = 0; i < data.length; i += 16) {
        const alpha = data[i + 3];
        if (alpha === 0) continue;

        totalR += data[i];
        totalG += data[i + 1];
        totalB += data[i + 2];
        count += 1;
    }

    const avgR = count > 0 ? totalR / count : 0;
    const avgG = count > 0 ? totalG / count : 0;
    const avgB = count > 0 ? totalB / count : 0;

    const area = width * height;
    const frameArea = (video.videoWidth || 1) * (video.videoHeight || 1);

    return {
        color: getColorNameFromRGB(avgR, avgG, avgB),
        size: getObjectSizeFromArea(area, frameArea),
        shape: getObjectShape(width, height),
        pattern: `${getColorNameFromRGB(avgR, avgG, avgB)}-${getObjectSizeFromArea(area, frameArea)}-${getObjectShape(width, height)}`
    };
}

function summarizePatterns(items) {
    const patternMap = new Map();

    items.forEach(item => {
        const label = getFriendlyItemName(item);
        const key = `${label} (${item.color}, ${item.size}, ${item.shape})`;
        patternMap.set(key, (patternMap.get(key) || 0) + 1);
    });

    return Array.from(patternMap.entries())
        .map(([pattern, count]) => `${pattern} x${count}`)
        .join(' | ');
}

// ==========================================
// 5. FUNGSI SIMPAN KE DATABASE & REAL-TIME LISTENER
// ==========================================
async function saveToDatabase(data) {
    try {
        const detections = data.map(item => ({
            ...item,
            class: getFriendlyItemName(item)
        }));
        const payload = await window.databaseService.addDetection(detections);
        console.log('Data tersimpan ke database:', payload);
        return true;
    } catch (err) {
        console.error('Gagal menyimpan ke database:', err);
        return false;
    }
}

function listenToDatabase() {
    window.databaseService.subscribe(records => {
        const dataList = document.getElementById('data-list');
        if (!dataList) return;
        dataList.replaceChildren();

        records.forEach(data => {
            const div = document.createElement('div');
            div.className = 'log-item';
            const time = document.createElement('span');
            time.className = 'time';
            const timestamp = data.timestamp && typeof data.timestamp.toDate === 'function'
                ? data.timestamp.toDate()
                : new Date(data.timestamp || Number.NaN);
            time.textContent = Number.isNaN(timestamp.getTime())
                ? 'Baru saja'
                : timestamp.toLocaleTimeString('id-ID');

            const items = document.createElement('span');
            items.className = 'items';
            items.textContent = normalizeStoredItemNames(data.items || data.pattern_summary || 'Deteksi tersimpan');
            div.append(time, items);
            dataList.appendChild(div);
        });

        const logCountEl = document.getElementById('log-count');
        if (logCountEl) logCountEl.textContent = `${records.length} items`;

        if (records.length === 0) {
            dataList.innerHTML = '<div class="empty-state">Belum ada objek terdeteksi</div>';
        }
    });
}

// ==========================================
// 6. KAMERA & DETECTION LOOP
// ==========================================
let isCameraOn = false;
let isAIDetecting = false;

async function startCamera() {
    if (isCameraOn && video.srcObject) {
        isAIDetecting = true;
        if (resultBox) resultBox.innerHTML = 'Kamera dan AI sudah aktif.';
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });

        video.srcObject = stream;
        isCameraOn = true;

        await new Promise(resolve => {
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                resolve();
            };
        });

        if (video) {
            video.style.transform = 'none';
            video.style.webkitTransform = 'none';
            video.style.zIndex = '1';
        }

        if (canvas) {
            canvas.style.zIndex = '2';
        }

        isAIDetecting = true;
        hideCameraOverlay();
        void detectLoop();
    } catch (err) {
        console.error('Gagal akses kamera:', err);
        alert('Akses kamera ditolak atau browser tidak mendukung.');
    }
}

function stopCamera() {
    isCameraOn = false;
    isAIDetecting = false;
    hasTrackedFace = false;
    lastFaceSeenAt = 0;
    resetFaceDisplayState();
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (overlayMsg) {
        overlayMsg.style.display = 'flex';
    }
    if (overlayText) {
        overlayText.textContent = 'Kamera dimatikan. Tekan Start Camera untuk menyalakan ulang.';
    }
    if (resultBox) resultBox.innerHTML = 'Kamera dimatikan.';
}

function stopAI() {
    isAIDetecting = false;
    if (resultBox) resultBox.innerHTML = 'AI dihentikan. Tekan Start Camera untuk menjalankannya kembali.';
}

async function detectObjectsWithCocoSsd() {
    if (!model || !video || video.readyState < 2) {
        return [];
    }

    return model.detect(video);
}

async function detectFacesAccurately() {
    if (!video || video.readyState < 2) {
        return [];
    }

    try {
        if (faceLandmarker) {
            const result = faceLandmarker.detectForVideo(video, performance.now());
            return (result.faceLandmarks || []).map(landmarks => {
                const points = landmarks.map(point => ({
                    x: point.x * video.videoWidth,
                    y: point.y * video.videoHeight
                }));
                const xs = points.map(point => point.x);
                const ys = points.map(point => point.y);
                const xMin = Math.min(...xs);
                const yMin = Math.min(...ys);
                const xMax = Math.max(...xs);
                const yMax = Math.max(...ys);

                return {
                    landmarks: points,
                    boundingBox: { xMin, yMin, width: xMax - xMin, height: yMax - yMin }
                };
            });
        }

        if (faceDetector) {
            const detections = await faceDetector.detect(video);
            return detections.map(face => ({
                boundingBox: {
                    xMin: face.boundingBox.x,
                    yMin: face.boundingBox.y,
                    width: face.boundingBox.width,
                    height: face.boundingBox.height
                },
                landmarks: null
            }));
        }
    } catch (error) {
        console.warn('Deteksi wajah gagal:', error);
    }

    return [];
}

function estimateFaceMetrics(faceBox, frameWidth, frameHeight) {
    const width = Math.max(1, faceBox.width || 0);
    const height = Math.max(1, faceBox.height || 0);
    const area = width * height;
    const frameArea = Math.max(1, frameWidth * frameHeight);
    const relativeSize = area / frameArea;

    let distance = 'Jauh';
    if (relativeSize > 0.18) distance = 'Dekat';
    else if (relativeSize > 0.08) distance = 'Sedang';

    return {
        width,
        height,
        area,
        relativeSize,
        distance
    };
}

function drawLandmarkPath(points, indexes) {
    const pathPoints = indexes.map(index => points[index]).filter(Boolean);
    if (pathPoints.length < 2) return;

    ctx.beginPath();
    ctx.moveTo(pathPoints[0].x, pathPoints[0].y);
    pathPoints.slice(1).forEach(point => ctx.lineTo(point.x, point.y));
    ctx.stroke();
}

function drawFaceAxes(face) {
    if (!ctx || !video) return;

    const faceBox = face.boundingBox;
    const x = faceBox.xMin;
    const y = faceBox.yMin;
    const width = faceBox.width;
    const height = faceBox.height;
    const centerX = x + width / 2;
    const centerY = y + height / 2;

    ctx.save();
    ctx.strokeStyle = '#5cf2c2';
    ctx.lineWidth = 2;
    ctx.setLineDash([7, 5]);
    ctx.strokeRect(x, y, width, height);

    ctx.beginPath();
    ctx.moveTo(x, centerY);
    ctx.lineTo(x + width, centerY);
    ctx.moveTo(centerX, y);
    ctx.lineTo(centerX, y + height);
    ctx.strokeStyle = 'rgba(255, 209, 102, 0.8)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.stroke();

    if (face.landmarks) {
        ctx.strokeStyle = '#5cf2c2';
        ctx.lineWidth = 1.5;
        drawLandmarkPath(face.landmarks, [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]);
        drawLandmarkPath(face.landmarks, [33, 160, 158, 133, 153, 144, 33]);
        drawLandmarkPath(face.landmarks, [362, 385, 387, 263, 373, 380, 362]);
        drawLandmarkPath(face.landmarks, [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]);
        drawLandmarkPath(face.landmarks, [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]);

        ctx.fillStyle = 'rgba(255, 209, 102, 0.9)';
        face.landmarks.forEach((point, index) => {
            if (index % 3 !== 0) return;
            ctx.beginPath();
            ctx.arc(point.x, point.y, 1.25, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    ctx.fillStyle = '#102b28';
    ctx.fillRect(x, Math.max(0, y - 25), 142, 22);
    ctx.fillStyle = '#5cf2c2';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText(`X ${centerX.toFixed(0)}  Y ${centerY.toFixed(0)}`, x + 7, Math.max(16, y - 9));
    ctx.restore();
}

function renderFaceTable(faces) {
    if (!faceTableBodyEl) return;
    if (faces.length === 0) {
        faceTableBodyEl.innerHTML = '<tr><td colspan="5">Wajah hilang, data direset.</td></tr>';
        return;
    }

    faceTableBodyEl.innerHTML = faces.map((face, index) => {
        const box = face.boundingBox;
        const metrics = estimateFaceMetrics(box, video.videoWidth, video.videoHeight);
        const centerX = box.xMin + box.width / 2;
        const centerY = box.yMin + box.height / 2;
        return `<tr><td>${index + 1}</td><td>${centerX.toFixed(0)} px</td><td>${centerY.toFixed(0)} px</td><td>${metrics.width.toFixed(0)} x ${metrics.height.toFixed(0)} px</td><td>${metrics.distance}</td></tr>`;
    }).join('');
}

function drawFaceHistory() {
    if (!faceChartCanvas || !faceChartCtx) return;

    const bounds = faceChartCanvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const pixelRatio = window.devicePixelRatio || 1;
    faceChartCanvas.width = Math.round(bounds.width * pixelRatio);
    faceChartCanvas.height = Math.round(bounds.height * pixelRatio);
    faceChartCtx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    const width = bounds.width;
    const height = bounds.height;
    const padding = { top: 10, right: 8, bottom: 10, left: 20 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const maxCount = Math.max(2, ...faceHistory.map(sample => sample.count));

    faceChartCtx.clearRect(0, 0, width, height);
    faceChartCtx.font = '10px Segoe UI, sans-serif';
    faceChartCtx.textAlign = 'right';
    faceChartCtx.fillStyle = '#8fa2a1';
    faceChartCtx.strokeStyle = 'rgba(205, 226, 223, 0.14)';
    faceChartCtx.lineWidth = 1;

    for (let tick = 0; tick <= 2; tick += 1) {
        const y = padding.top + (plotHeight * tick) / 2;
        const value = Math.round(maxCount * (1 - tick / 2));
        faceChartCtx.beginPath();
        faceChartCtx.moveTo(padding.left, y);
        faceChartCtx.lineTo(width - padding.right, y);
        faceChartCtx.stroke();
        faceChartCtx.fillText(String(value), padding.left - 6, y + 3);
    }

    if (!faceHistory.length) return;
    const gradient = faceChartCtx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, 'rgba(92, 242, 194, 0.30)');
    gradient.addColorStop(1, 'rgba(92, 242, 194, 0.01)');

    const coordinates = faceHistory.map((sample, index) => ({
        x: padding.left + (faceHistory.length === 1 ? plotWidth : (plotWidth * index) / (faceHistory.length - 1)),
        y: padding.top + plotHeight * (1 - sample.count / maxCount)
    }));

    faceChartCtx.beginPath();
    faceChartCtx.moveTo(coordinates[0].x, height - padding.bottom);
    coordinates.forEach(point => faceChartCtx.lineTo(point.x, point.y));
    faceChartCtx.lineTo(coordinates.at(-1).x, height - padding.bottom);
    faceChartCtx.closePath();
    faceChartCtx.fillStyle = gradient;
    faceChartCtx.fill();

    faceChartCtx.beginPath();
    coordinates.forEach((point, index) => {
        if (index === 0) faceChartCtx.moveTo(point.x, point.y);
        else faceChartCtx.lineTo(point.x, point.y);
    });
    faceChartCtx.strokeStyle = '#5cf2c2';
    faceChartCtx.lineWidth = 2.5;
    faceChartCtx.lineJoin = 'round';
    faceChartCtx.lineCap = 'round';
    faceChartCtx.stroke();

    const latest = coordinates.at(-1);
    faceChartCtx.beginPath();
    faceChartCtx.arc(latest.x, latest.y, 3.5, 0, Math.PI * 2);
    faceChartCtx.fillStyle = '#ffd166';
    faceChartCtx.fill();
}

function updateFaceHistory(faceCount) {
    const now = Date.now();
    if (now - lastFaceSampleAt < 1000) return;
    lastFaceSampleAt = now;
    faceHistory.push({ time: now, count: faceCount });
    if (faceHistory.length > 30) faceHistory.shift();
    if (faceChartCurrentEl) faceChartCurrentEl.textContent = String(faceCount);
    if (faceChartStatusEl) {
        let status = 'WAITING';
        if (faceCount > 0) status = 'LIVE';
        else if (hasTrackedFace) status = 'LOST';
        faceChartStatusEl.textContent = status;
        faceChartStatusEl.classList.toggle('is-live', faceCount > 0);
    }
    drawFaceHistory();
}

function resetFaceDisplayState() {
    renderFaceTable([]);
    if (faceChartCurrentEl) faceChartCurrentEl.textContent = '0';
    if (faceChartStatusEl) {
        faceChartStatusEl.textContent = 'WAITING';
        faceChartStatusEl.classList.remove('is-live');
    }
}

async function detectLoop() {
    if (!isAIDetecting || !isCameraOn || !model || video.paused || video.ended) return;

    try {
        const faceDetections = await detectFacesAccurately();
        const predictions = await detectObjectsWithCocoSsd();
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (faceDetections.length > 0) {
            lastFaceSeenAt = Date.now();
            hasTrackedFace = true;
            renderFaceTable(faceDetections);

            faceDetections.forEach(face => {
                drawFaceAxes(face);
            });
        } else if (hasTrackedFace && Date.now() - lastFaceSeenAt > FACE_RESET_TIMEOUT_MS) {
            resetFaceDisplayState();
            hasTrackedFace = false;
        }
        updateFaceHistory(faceDetections.length);

        let detectedTargets = [];

        predictions.forEach(pred => {
            const [x, y, w, h] = pred.bbox;
            const confidence = pred.score;
            if (confidence < OBJECT_CONFIDENCE_THRESHOLD) return;

            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            const pattern = analyzeObjectPattern(x, y, w, h);
            const friendlyClass = getFriendlyLabel(pred.class);
            const label = `${friendlyClass} ${(confidence * 100).toFixed(0)}% | ${pattern.color}`;
            ctx.fillStyle = '#10b981';
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x, y > 25 ? y - 25 : 0, textWidth + 10, 25);

            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 14px sans-serif';
            ctx.fillText(label, x + 5, y > 25 ? y - 7 : 18);

            detectedTargets.push({
                class: friendlyClass,
                rawClass: pred.class,
                confidence: Number.parseFloat((confidence * 100).toFixed(2)),
                color: pattern.color,
                size: pattern.size,
                shape: pattern.shape,
                pattern: pattern.pattern
            });
        });

        registerDetectedCategories(detectedTargets, faceDetections.length);

        if (resultBox) {
            if (detectedTargets.length > 0 || faceDetections.length > 0) {
                const detectedSummary = detectedTargets
                    .map(d => `${getFriendlyItemName(d)} (${d.color}, ${d.size}, ${d.shape}, ${d.confidence}%)`)
                    .join(', ');
                const faceSummary = faceDetections.length > 0
                    ? faceDetections.map(face => {
                        const box = face.boundingBox;
                        const centerX = box.xMin + box.width / 2;
                        const centerY = box.yMin + box.height / 2;
                        const metrics = estimateFaceMetrics(box, video.videoWidth, video.videoHeight);
                        return `X:${centerX.toFixed(0)} Y:${centerY.toFixed(0)} (${metrics.width.toFixed(0)}x${metrics.height.toFixed(0)} px, ${metrics.distance})`;
                    }).join(' | ')
                    : 'Tidak ada wajah';
                const patternSummary = summarizePatterns(detectedTargets);
                resultBox.innerHTML = '<b>Deteksi:</b> ' + (detectedSummary || 'Tidak ada objek yang dikenali') + '<br><b>Wajah:</b> ' + faceSummary + '<br><b>Pola:</b> ' + (patternSummary || 'Belum ada pola');
            } else {
                resultBox.innerHTML = 'Mencari objek...';
            }
        }

        if (detectedTargets.length > 0 && !isSaveInProgress && (Date.now() - lastSaveTime) >= SAVE_THROTTLE_MS) {
            lastSaveTime = Date.now();
            isSaveInProgress = true;
            void saveToDatabase(detectedTargets).finally(() => {
                isSaveInProgress = false;
            });
        }
    } catch (e) {
        console.error('Error pada loop deteksi:', e);
    }

    if (isAIDetecting) {
        requestAnimationFrame(detectLoop);
    }
}

// ==========================================
// 7. INISIALISASI UTAMA APLIKASI
// ==========================================
function initApp() {
    console.log('Memulai inisialisasi aplikasi...');
    renderCategoryStats();
    drawFaceHistory();
    listenToDatabase();
    void loadModel();
}

document.addEventListener('DOMContentLoaded', initApp);