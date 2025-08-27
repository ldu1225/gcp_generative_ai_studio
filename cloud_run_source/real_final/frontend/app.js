document.addEventListener('DOMContentLoaded', () => {
    let lastGeneratedImage = { url: null, gcsUri: null };

    // --- Unified Error Handler ---
    const handleApiError = async (response) => {
        if (!response.ok) {
            const result = await response.json().catch(() => ({ error: `Server returned status ${response.status}` }));
            throw new Error(result.error || `An unknown server error occurred.`);
        }
        return response.json();
    };

    // --- Tab Navigation ---
    const tabLinks = document.querySelectorAll('.tab-link');
    const tabContents = document.querySelectorAll('.tab-content');
    tabLinks.forEach(link => {
        link.addEventListener('click', () => {
            const tabId = link.dataset.tab;
            if (link.classList.contains('active')) return;
            
            localStorage.setItem('activeTab', tabId);

            tabLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
        });
    });

    const savedTab = localStorage.getItem('activeTab');
    if (savedTab && document.querySelector(`.tab-link[data-tab="${savedTab}"]`)) {
        document.querySelector(`.tab-link[data-tab="${savedTab}"]`).click();
    }

    function setupFileUpload(uploadId, previewContainerId, previewImgId, dropZoneSelector) {
        const uploadElement = document.getElementById(uploadId);
        const previewContainer = document.getElementById(previewContainerId);
        const previewImage = document.getElementById(previewImgId);
        const dropZone = document.querySelector(dropZoneSelector);
        if(!uploadElement) return;

        uploadElement.addEventListener('change', () => handleFile(uploadElement.files[0]));
        dropZone.addEventListener('dragover', e => e.preventDefault());
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (file) {
                uploadElement.files = e.dataTransfer.files;
                handleFile(file);
            }
        });
        function handleFile(file) {
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImage.src = e.target.result;
                    previewContainer.classList.remove('hidden');
                    dropZone.classList.add('has-preview');
                };
                reader.readAsDataURL(file);
            }
        }
    }

    document.querySelectorAll('.btn-enhance').forEach(btn => {
        btn.addEventListener('click', async () => {
            const targetId = btn.dataset.target;
            const textarea = document.getElementById(targetId);
            const originalText = textarea.value;
            if (!originalText) { alert('Please enter a prompt first.'); return; }
            
            btn.classList.add('enhancing');
            btn.disabled = true;

            try {
                const response = await fetch('/api/enhance_prompt', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: originalText })
                });
                const result = await handleApiError(response);
                textarea.value = result.enhancedPrompt;
            } catch (error) {
                alert('Error enhancing prompt: ' + error.message);
            } finally {
                setTimeout(() => {
                    btn.classList.remove('enhancing');
                    btn.disabled = false;
                }, 600);
            }
        });
    });

    // --- Image Edit Tab ---
    const editForm = document.getElementById('edit-form');
    const upscaleBtnStandalone = document.getElementById('upscale-btn-standalone');
    const imageEditUpload = document.getElementById('image-edit-upload');
    
    setupFileUpload('image-edit-upload', 'image-edit-preview-container', 'image-edit-preview', '#edit-form .drop-zone');
    
    editForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const loading = document.getElementById('image-edit-loading');
        const editResults = document.getElementById('image-edit-results');
        
        loading.classList.remove('hidden');
        editResults.classList.add('hidden');

        try {
            const response = await fetch('/api/edit', { method: 'POST', body: new FormData(editForm) });
            const result = await handleApiError(response);
            document.getElementById('original-image').src = result.originalImageUrl;
            document.getElementById('edited-image').src = result.editedImageUrl;
            editResults.classList.remove('hidden');
        } catch (error) {
            alert('Error editing image: ' + error.message);
        } finally {
            loading.classList.add('hidden');
        }
    });

    upscaleBtnStandalone.addEventListener('click', async () => {
        if (imageEditUpload.files.length === 0) {
            alert('Please upload an image to upscale.');
            return;
        }
        const loading = document.getElementById('image-edit-loading');
        const upscaleResults = document.getElementById('upscale-results');

        loading.classList.remove('hidden');
        upscaleResults.classList.add('hidden');
        
        const formData = new FormData();
        formData.append('image', imageEditUpload.files[0]);

        try {
            const response = await fetch('/api/upscale_image', { method: 'POST', body: formData });
            const result = await handleApiError(response);
            document.getElementById('upscaled-image').src = result.upscaledUrl;
            upscaleResults.classList.remove('hidden');
        } catch (error) {
            alert('Error upscaling image: ' + error.message);
        } finally {
            loading.classList.add('hidden');
        }
    });

    // --- Image Generate Tab ---
    const imageGenForm = document.getElementById('image-gen-form');
    imageGenForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const loading = document.getElementById('image-gen-loading');
        const results = document.getElementById('image-gen-results');
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        try {
            const prompt = new FormData(imageGenForm).get('prompt');
            const response = await fetch('/api/generate_image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            });
            const result = await handleApiError(response);
            document.getElementById('generated-image').src = result.imageUrl;
            lastGeneratedImage = { url: result.imageUrl, gcsUri: result.gcsUri };
            results.classList.remove('hidden');
        } catch (error) {
            alert('Error generating image: ' + error.message);
        } finally {
            loading.classList.add('hidden');
        }
    });

    // --- Video Generation Tab ---
    const videoForm = document.getElementById('video-gen-form');
    setupFileUpload('video-gen-upload', 'video-gen-preview-container', 'video-gen-preview', '#video-gen-form .drop-zone');
    videoForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const loading = document.getElementById('video-gen-loading');
        const results = document.getElementById('video-gen-results');
        
        loading.classList.remove('hidden');
        results.classList.add('hidden');

        try {
            const response = await fetch('/api/generate_video', { method: 'POST', body: new FormData(videoForm) });
            const result = await handleApiError(response);
            document.getElementById('generated-video').src = result.videoUrl;
            results.classList.remove('hidden');
        } catch (error) {
            alert('Error generating video: ' + error.message);
        } finally {
            loading.classList.add('hidden');
        }
    });

    // --- Music Generation Tab ---
    const musicGenForm = document.getElementById('music-gen-form');
    musicGenForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const loading = document.getElementById('music-gen-loading');
        const results = document.getElementById('music-gen-results');
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        try {
            const prompt = new FormData(musicGenForm).get('prompt');
            const response = await fetch('/api/generate_music', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt })
            });
            const result = await handleApiError(response);
            document.getElementById('generated-music').src = result.audioUrl;
            results.classList.remove('hidden');
        } catch (error) {
            alert('Error generating music: ' + error.message);
        } finally {
            loading.classList.add('hidden');
        }
    });

    // --- Voice Generation Tab ---
    const voiceGenForm = document.getElementById('voice-gen-form');
    voiceGenForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const loading = document.getElementById('voice-gen-loading');
        const results = document.getElementById('voice-gen-results');
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        try {
            const formData = new FormData(voiceGenForm);
            const prompt = formData.get('prompt');
            const voice = formData.get('voice');
            const response = await fetch('/api/generate_voice', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, voice })
            });
            const result = await handleApiError(response);
            document.getElementById('generated-voice').src = result.audioUrl;
            results.classList.remove('hidden');
        } catch (error) {
            alert('Error generating voice: ' + error.message);
        } finally {
            loading.classList.add('hidden');
        }
    });

    // --- Ad Campaign Simulator Tab ---
    const campaignGenerateBtn = document.getElementById('campaign-generate-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const campaignPrompt = document.getElementById('campaign-prompt');

    campaignGenerateBtn.addEventListener('click', async () => {
        if (!campaignPrompt.value) { alert("Please describe the ad image."); return; }
        
        const placeholder = document.getElementById('ad-image-placeholder');
        const canvas = document.getElementById('campaign-canvas');
        const ctx = canvas.getContext('2d');
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        placeholder.classList.remove('hidden');
        placeholder.textContent = "Generating image...";
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/api/generate_image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: campaignPrompt.value })
            });
            const result = await handleApiError(response);
            lastGeneratedImage = { url: result.imageUrl, gcsUri: result.gcsUri, imageData: result.imageData };
            
            const img = new Image();
            img.onload = () => {
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                ctx.drawImage(img, 0, 0);
                placeholder.classList.add('hidden');
                analyzeBtn.disabled = false;
            };
            img.onerror = () => {
                alert('Failed to load image onto canvas. This might be a CORS issue. Please check the browser console.');
                placeholder.textContent = "Failed to load image.";
            };
            img.src = result.imageData; 
        } catch (error) {
            alert('Error generating ad image: ' + error.message);
            placeholder.textContent = "Image generation failed. Please try again.";
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!lastGeneratedImage.gcsUri) { return; }
        const loading = document.getElementById('analysis-loading');
        const resultsGrid = document.getElementById('analysis-results-grid');
        const resultsContent = document.getElementById('analysis-results-content');
        const annotatedImage = document.getElementById('annotated-image');
        const policyBox = document.getElementById('hsad-policy-box');
        const geminiResultsBox = document.getElementById('gemini-analysis-results');

        loading.classList.remove('hidden');
        resultsGrid.classList.add('hidden');
        geminiResultsBox.classList.add('hidden');

        try {
            const response = await fetch('/api/analyze_image', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ gcsUri: lastGeneratedImage.gcsUri })
            });
            const result = await handleApiError(response);
            
            annotatedImage.src = result.annotatedImageUrl;
            
            // HSAD Policy Box
            let policyHtml = `<h4>HSAD 광고 콘텐츠 정책</h4><ul>`;
            for(const [key, value] of Object.entries(result.hsadPolicyDefinition)) {
                policyHtml += `<li><strong>${key}:</strong> ${value}</li>`;
            }
            policyHtml += `</ul>`;
            const policy = result.hsadPolicy;
            const statusClass = policy.status === '승인' ? 'status-approved' : 'status-rejected';
            policyHtml += `<p><strong>심사 상태: <span class="${statusClass}">${policy.status}</span></strong> (${policy.reason})</p>`;
            policyBox.innerHTML = policyHtml;

            // Vision API Box (Simplified)
            let resultsHtml = `<h4>Vision API 상세 분석</h4>`;
            const objectNames = result.objects.map(obj => obj.name).join(', ');
            resultsHtml += `<p><strong>탐지된 객체:</strong> ${objectNames || '없음'}</p>`;
            resultsHtml += `<p><strong>감지된 텍스트:</strong> ${result.text || '없음'}</p>`;
            resultsHtml += `<h6>콘텐츠 안전성 평가</h6><div class="safety-grid">
`;
            for (const [key, value] of Object.entries(result.safety)) {
                resultsHtml += `<div class="safety-item"><span>${key.charAt(0).toUpperCase() + key.slice(1)}:</span> <span>${value}</span></div>`;
            }
            resultsHtml += `</div>`;
            resultsContent.innerHTML = resultsHtml;

            // Gemini Analysis Box
            let geminiHtml = `<h4>Gemini 광고 크리에이티브 분석</h4>`;
            if (result.geminiLabels && result.geminiLabels.length > 0) {
                geminiHtml += `<h6>AI 추천 라벨</h6><div class="label-grid">
`;
                result.geminiLabels.forEach(label => {
                    geminiHtml += `<span class="gemini-label">${label.ko} (${label.en})</span>`;
                });
                geminiHtml += `</div>`;
            }
            if (result.geminiAnalysis) {
                geminiHtml += `<h6>AI 종합 분석</h6>`;
                let formattedText = result.geminiAnalysis
                    .replace(/</g, "&lt;").replace(/>/g, "&gt;")
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\n/g, '<br>');
                geminiHtml += `<p>${formattedText}</p>`;
            }
            geminiResultsBox.innerHTML = geminiHtml;
            
            resultsGrid.classList.remove('hidden');
            geminiResultsBox.classList.remove('hidden');

        } catch (error) {
            alert('Error analyzing image: ' + error.message);
        } finally {
            loading.classList.add('hidden');
        }
    });

    // --- Fine-Tuning Simulator Tab ---
    const finetuneUpload = document.getElementById('finetune-upload');
    const finetuneGallery = document.getElementById('finetune-preview-gallery');
    const finetuneBtn = document.getElementById('finetune-generate-btn');

    finetuneUpload.addEventListener('change', () => {
        finetuneGallery.innerHTML = '';
        for (const file of finetuneUpload.files) {
            const reader = new FileReader();
            reader.onload = e => {
                const img = document.createElement('img');
                img.src = e.target.result;
                finetuneGallery.appendChild(img);
            };
            reader.readAsDataURL(file);
        }
    });

    finetuneBtn.addEventListener('click', async () => {
        const prompt = document.getElementById('finetune-prompt-input').value;
        if (finetuneUpload.files.length === 0) { alert('Please upload style images.'); return; }
        if (!prompt) { alert('Please describe your product.'); return; }

        const loading = document.getElementById('finetune-loading');
        const placeholder = document.getElementById('finetune-placeholder');
        const image = document.getElementById('finetune-image');
        
        const formData = new FormData();
        formData.append('prompt', prompt);
        for (const file of finetuneUpload.files) {
            formData.append('files', file);
        }

        loading.classList.remove('hidden');
        placeholder.classList.add('hidden');
        image.classList.add('hidden');

        try {
            const response = await fetch('/api/simulate_finetuning', { method: 'POST', body: formData });
            const result = await handleApiError(response);
            image.src = result.imageUrl;
            image.classList.remove('hidden');
        } catch (error) {
            alert('Error generating fine-tuned image: ' + error.message);
            placeholder.classList.remove('hidden');
        } finally {
            loading.classList.add('hidden');
        }
    });
});