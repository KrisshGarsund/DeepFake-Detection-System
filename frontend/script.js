document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.section');
    const navTriggers = document.querySelectorAll('.nav-trigger');
    const tabTriggers = document.querySelectorAll('.tab-trigger');
    
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewSection = document.getElementById('preview-section');
    const mediaPreview = document.getElementById('media-preview');
    const uploadInstruction = document.getElementById('upload-instruction');
    const modalityText = document.getElementById('modality-text');
    
    const analyzeBtn = document.getElementById('analyze-btn');
    const cancelBtn = document.getElementById('cancel-btn');
    const resetBtn = document.getElementById('reset-btn');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results');
    
    const gaugeFill = document.getElementById('gauge-fill');
    const confidenceText = document.getElementById('confidence-text');
    const finalVerdict = document.getElementById('final-verdict');
    const verdictDesc = document.getElementById('verdict-desc');
    const insightsList = document.getElementById('insights-list');

    // --- State ---
    let currentSection = 'home';
    let currentModality = 'image';
    let currentFile = null;
    const API_URL = 'http://127.0.0.1:8000/api/analyze';

    // --- Navigation Logic ---
    function switchSection(sectionId) {
        sections.forEach(s => s.classList.remove('active'));
        navLinks.forEach(l => l.classList.remove('active'));
        
        document.getElementById(sectionId).classList.add('active');
        const activeLink = document.querySelector(`.nav-link[data-section="${sectionId}"]`);
        if (activeLink) activeLink.classList.add('active');
        
        currentSection = sectionId;
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    navLinks.forEach(link => {
        link.addEventListener('click', () => switchSection(link.dataset.section));
    });

    navTriggers.forEach(trigger => {
        trigger.addEventListener('click', () => switchSection(trigger.dataset.section));
    });

    // --- Modality (Sub-tabs) Logic ---
    function switchModality(modality) {
        currentModality = modality;
        tabTriggers.forEach(t => t.classList.remove('active'));
        document.querySelector(`.tab-trigger[data-modality="${modality}"]`).classList.add('active');
        
        // Update UI
        modalityText.textContent = modality;
        const acceptTypes = {
            'image': 'image/*',
            'video': 'video/*',
            'audio': 'audio/*'
        };
        fileInput.accept = acceptTypes[modality];
        
        // Reset upload state if modality changes
        resetUploadState();
    }

    tabTriggers.forEach(tab => {
        tab.addEventListener('click', () => switchModality(tab.dataset.modality));
    });

    function resetUploadState() {
        currentFile = null;
        fileInput.value = '';
        dropArea.classList.remove('hidden');
        previewSection.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loader.classList.add('hidden');
        mediaPreview.innerHTML = '';
        document.getElementById('upload-panel').classList.remove('hidden');
    }

    // --- Upload Logic ---
    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, e => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    dropArea.addEventListener('drop', e => handleFiles(e.dataTransfer.files));

    function handleFiles(files) {
        if (!files.length) return;
        currentFile = files[0];
        
        displayPreview(currentFile);
        dropArea.classList.add('hidden');
        previewSection.classList.remove('hidden');
    }

    function displayPreview(file) {
        mediaPreview.innerHTML = '';
        const url = URL.createObjectURL(file);
        
        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            img.src = url;
            img.style.maxWidth = '100%';
            mediaPreview.appendChild(img);
        } else if (file.type.startsWith('video/')) {
            const video = document.createElement('video');
            video.src = url;
            video.controls = true;
            video.style.width = '100%';
            mediaPreview.appendChild(video);
        } else if (file.type.startsWith('audio/')) {
            mediaPreview.innerHTML = `
                <div style="padding: 3rem; text-align: center;">
                    <i class="fa-solid fa-waveform" style="font-size: 4rem; color: var(--accent-glow); margin-bottom: 1rem;"></i>
                    <audio src="${url}" controls style="width: 100%;"></audio>
                </div>
            `;
        }
    }

    cancelBtn.addEventListener('click', resetUploadState);
    resetBtn.addEventListener('click', resetUploadState);

    // --- Analysis Logic ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        previewSection.classList.add('hidden');
        loader.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Model inference failed');
            const result = await response.json();
            
            showResults(result);
        } catch (err) {
            console.error(err);
            alert('Analysis Error: ' + err.message);
            resetUploadState();
        }
    });

    function showResults(data) {
        loader.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        const isFake = data.prediction === 'FAKE';
        const confidence = data.confidence;
        const color = isFake ? 'var(--danger)' : 'var(--success)';
        
        // Update Badge
        const badge = document.getElementById('modality-badge');
        if (badge) badge.textContent = data.modality.toUpperCase();
        
        // Gauge Update
        gaugeFill.setAttribute('stroke-dasharray', `${confidence * 100}, 100`);
        gaugeFill.className = `circle ${isFake ? 'red' : 'green'}`;
        
        // Text update
        confidenceText.textContent = `${Math.round(confidence * 100)}%`;
        finalVerdict.textContent = isFake ? 'AI Generated' : 'Authentic Media';
        finalVerdict.style.color = color;
        verdictDesc.textContent = data.explanation || (isFake ? 'This media shows strong signs of AI manipulation.' : 'No significant signs of AI manipulation detected.');

        // Insights
        insightsList.innerHTML = `
            <li style="display: flex; justify-content: space-between; padding: 0.8rem 0; border-bottom: 1px solid var(--panel-border);">
                <span style="color: var(--text-secondary);">Modality</span>
                <span style="font-weight: 600;">${data.modality.toUpperCase()}</span>
            </li>
            <li style="display: flex; justify-content: space-between; padding: 0.8rem 0; border-bottom: 1px solid var(--panel-border);">
                <span style="color: var(--text-secondary);">Model Trust</span>
                <span style="font-weight: 600;">${Math.round(data.confidence * 100)}%</span>
            </li>
            <li style="display: flex; justify-content: space-between; padding: 0.8rem 0;">
                <span style="color: var(--text-secondary);">Verdict</span>
                <span style="font-weight: 600; color: ${color};">${data.prediction}</span>
            </li>
        `;
        
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
});
