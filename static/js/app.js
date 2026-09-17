/**
 * FloraScan AI — Interactive Frontend Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const dropzoneIdle = document.getElementById('dropzone-idle');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const previewName = document.getElementById('preview-name');
    const previewSize = document.getElementById('preview-size');
    const removeBtn = document.getElementById('remove-btn');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeBtnText = document.getElementById('analyze-btn-text');
    
    // LLM & History Elements
    const historySection = document.getElementById('history-section');
    const historyList = document.getElementById('history-list');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatMessages = document.getElementById('chat-messages');
    const llmRecBox = document.getElementById('llm-recommendation-box');
    const llmRecContent = document.getElementById('llm-recommendation-content');
    
    // Sample Chips
    const sampleBlightBtn = document.getElementById('sample-blight-btn');
    const sampleHealthyBtn = document.getElementById('sample-healthy-btn');
    const sampleChips = document.querySelectorAll('.sample-chip');

    // States
    const emptyState = document.getElementById('empty-state');
    const loadingState = document.getElementById('loading-state');
    const resultState = document.getElementById('result-state');

    // Result Elements
    const diagnosisBanner = document.getElementById('diagnosis-banner');
    const resPlantName = document.getElementById('res-plant-name');
    const resDiseaseName = document.getElementById('res-disease-name');
    const resRawClass = document.getElementById('res-raw-class');
    const resSeverityPill = document.getElementById('res-severity-pill');
    const resConfidenceCircle = document.getElementById('res-confidence-circle');
    const resConfidenceVal = document.getElementById('res-confidence-val');
    const resStatusCallout = document.getElementById('res-status-callout');
    const resStatusTitle = document.getElementById('res-status-title');
    const resDescription = document.getElementById('res-description');
    const resTreatment = document.getElementById('res-treatment');
    const predictionsList = document.getElementById('predictions-list');
    const resetBtn = document.getElementById('reset-btn');
    const copyBtn = document.getElementById('copy-btn');
    const copyBtnText = document.getElementById('copy-btn-text');
    const sendReportEmailBtn = document.getElementById('send-report-email-btn');
    const emailStatusToast = document.getElementById('email-status-toast');
    const emailLatestBtn = document.getElementById('email-latest-btn');
    const historyEmailAlert = document.getElementById('history-email-alert');

    // Active State Trackers
    let currentFile = null;
    let currentSampleName = null;
    let lastResult = null;

    // --- File Input & Drag/Drop Event Listeners ---
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });

        dropzone.addEventListener('click', () => {
            if (!currentFile && !currentSampleName) {
                fileInput.click();
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelect(e.target.files[0]);
            }
        });

        // Drag & Drop
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            if (dt.files && dt.files[0]) {
                handleFileSelect(dt.files[0]);
            }
        });

        removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            clearSelectedImage();
        });

        // --- Preset Sample Handlers ---
        sampleBlightBtn.addEventListener('click', () => {
            selectPresetSample('sample_blight.jpg', sampleBlightBtn);
        });

        sampleHealthyBtn.addEventListener('click', () => {
            selectPresetSample('sample_healthy.jpg', sampleHealthyBtn);
        });
    }

    function selectPresetSample(sampleFilename, chipElement) {
        sampleChips.forEach(chip => chip.classList.remove('active'));
        chipElement.classList.add('active');

        currentFile = null;
        currentSampleName = sampleFilename;

        previewImage.src = `/static/samples/${sampleFilename}`;
        previewName.textContent = sampleFilename === 'sample_blight.jpg' ? 'Tomato Blight Sample' : 'Healthy Leaf Sample';
        previewSize.textContent = 'Preset Asset';

        dropzoneIdle.style.display = 'none';
        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
    }

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (JPG, PNG, WEBP).');
            return;
        }

        sampleChips.forEach(chip => chip.classList.remove('active'));
        currentFile = file;
        currentSampleName = null;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewName.textContent = file.name;
            previewSize.textContent = formatBytes(file.size);

            dropzoneIdle.style.display = 'none';
            previewContainer.style.display = 'block';
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    function clearSelectedImage() {
        currentFile = null;
        currentSampleName = null;
        fileInput.value = '';
        previewImage.src = '';
        dropzoneIdle.style.display = 'flex';
        previewContainer.style.display = 'none';
        analyzeBtn.disabled = true;
        sampleChips.forEach(chip => chip.classList.remove('active'));
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    // --- Classification API Call ---
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            if (!currentFile && !currentSampleName) return;

            // Show Loading State
            setAppState('loading');
            analyzeBtn.disabled = true;

            const sendEmail = document.getElementById('send-email-cb')?.checked || false;

            try {
                let response;
                if (currentFile) {
                    const formData = new FormData();
                    formData.append('file', currentFile);
                    formData.append('send_email', sendEmail);
                    response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                } else if (currentSampleName) {
                    response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ sample: currentSampleName, send_email: sendEmail })
                    });
                }

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || 'Diagnostic evaluation failed.');
                }

                lastResult = data;
                renderDiagnosisResults(data);
                setAppState('result');
            } catch (error) {
                console.error('Inference error:', error);
                alert(`Classification error: ${error.message}`);
                setAppState('empty');
            } finally {
                analyzeBtn.disabled = false;
            }
        });
    }

    // --- Switch UI States ---
    function setAppState(state) {
        if (emptyState) emptyState.style.display = 'none';
        if (loadingState) loadingState.style.display = 'none';
        if (resultState) resultState.style.display = 'none';

        if (state === 'empty' && emptyState) {
            emptyState.style.display = 'block';
        } else if (state === 'loading' && loadingState) {
            loadingState.style.display = 'block';
        } else if (state === 'result' && resultState) {
            resultState.style.display = 'block';
        }
    }

    // --- Render Results ---
    function renderDiagnosisResults(data) {
        const isHealthy = data.is_healthy;
        const confidence = data.confidence;

        // Headline & Banner Theme
        resPlantName.textContent = data.plant;
        resDiseaseName.textContent = data.disease;
        resRawClass.textContent = data.predicted_class;

        if (isHealthy) {
            diagnosisBanner.classList.remove('is-diseased');
            resStatusCallout.classList.remove('is-diseased');
            resStatusTitle.textContent = 'Specimen Appears Healthy';
            resSeverityPill.textContent = 'Healthy';
            resSeverityPill.className = 'severity-pill severity-none';
        } else {
            diagnosisBanner.classList.add('is-diseased');
            resStatusCallout.classList.add('is-diseased');
            resStatusTitle.textContent = 'Pathogenic Infection Detected';
            resSeverityPill.textContent = data.severity;
            resSeverityPill.className = `severity-pill severity-${data.severity.toLowerCase()}`;
        }

        resDescription.textContent = data.description || 'Foliar pattern classified.';
        resTreatment.textContent = data.treatment || 'Consult an agricultural specialist.';

        // Populate Neural Telemetry Details
        const telem = data.telemetry || {};
        const telemModelFile = document.getElementById('telem-model-file');
        const telemLatency = document.getElementById('telem-latency');
        const telemTensor = document.getElementById('telem-tensor');
        if (telemModelFile) telemModelFile.textContent = telem.model_file ? `${telem.model_file} (${telem.model_size_mb} MB)` : 'best_plant__model_float32.tflite';
        if (telemLatency) telemLatency.textContent = telem.inference_time_ms ? `${telem.inference_time_ms} ms` : '20 ms';
        if (telemTensor) telemTensor.textContent = telem.input_tensor_shape ? `[${telem.input_tensor_shape.join(', ')}]` : '[1, 224, 224, 3]';

        // Animate Circular Confidence Meter
        animateConfidenceRing(confidence, isHealthy);

        // Render Top 5 Predictions Bar Chart
        renderProbabilityBars(data.top_predictions);
        
        // Handle Auto-Email Delivery Status
        if (data.email_status && emailStatusToast) {
            emailStatusToast.style.display = 'block';
            if (data.email_status.success) {
                emailStatusToast.style.background = 'rgba(16, 185, 129, 0.12)';
                emailStatusToast.style.borderColor = '#10b981';
                emailStatusToast.style.color = '#34d399';
                emailStatusToast.innerHTML = `<strong>✓ Report Dispatched:</strong> ${data.email_status.message}`;
            } else {
                emailStatusToast.style.background = 'rgba(239, 68, 68, 0.12)';
                emailStatusToast.style.borderColor = '#ef4444';
                emailStatusToast.style.color = '#f87171';
                emailStatusToast.innerHTML = `<strong>⚠️ Email Notice:</strong> ${data.email_status.message}`;
            }
        } else if (emailStatusToast) {
            emailStatusToast.style.display = 'none';
        }

        // Fetch LLM Recommendation
        fetchRecommendation(data.plant, data.disease);
    }

    async function fetchRecommendation(plant, disease) {
        if (!llmRecBox || !llmRecContent) return;
        llmRecBox.style.display = 'flex';
        llmRecContent.innerHTML = `<div class="llm-loading">
            <div class="spinner-orbit" style="width:20px;height:20px;border-width:2px;margin:0;"></div>
            <span>Analyzing agronomic protocol with Llama3...</span>
        </div>`;
        
        try {
            const res = await fetch('/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ plant, disease })
            });
            const data = await res.json();
            if (data.success) {
                llmRecContent.textContent = data.recommendation;
            } else {
                llmRecContent.textContent = "Unable to generate recommendation.";
            }
        } catch (e) {
            llmRecContent.textContent = "Error connecting to AI service.";
        }
    }

    function animateConfidenceRing(confidencePct, isHealthy) {
        const circumference = 2 * Math.PI * 50; // r=50 -> ~314.159
        const targetOffset = circumference - (confidencePct / 100) * circumference;

        // Reset
        resConfidenceCircle.style.strokeDashoffset = circumference;

        // Animate stroke
        setTimeout(() => {
            resConfidenceCircle.style.strokeDashoffset = targetOffset;
        }, 50);

        // Counter Animation
        let startVal = 0;
        const duration = 1000;
        const stepTime = 20;
        const totalSteps = duration / stepTime;
        const increment = confidencePct / totalSteps;

        const timer = setInterval(() => {
            startVal += increment;
            if (startVal >= confidencePct) {
                resConfidenceVal.textContent = `${confidencePct.toFixed(1)}%`;
                clearInterval(timer);
            } else {
                resConfidenceVal.textContent = `${startVal.toFixed(1)}%`;
            }
        }, stepTime);
    }

    function renderProbabilityBars(predictions) {
        predictionsList.innerHTML = '';
        if (!predictions || !predictions.length) return;

        predictions.forEach((pred, index) => {
            const row = document.createElement('div');
            row.className = 'pred-row';

            const meta = document.createElement('div');
            meta.className = 'pred-meta';

            const label = document.createElement('span');
            label.className = 'pred-label';
            label.textContent = `${pred.plant} — ${pred.disease}`;

            const pct = document.createElement('span');
            pct.className = 'pred-pct';
            pct.textContent = `${pred.confidence.toFixed(2)}%`;

            meta.appendChild(label);
            meta.appendChild(pct);

            const barBg = document.createElement('div');
            barBg.className = 'pred-bar-bg';

            const barFill = document.createElement('div');
            barFill.className = `pred-bar-fill ${index === 0 ? 'highlight' : ''}`;

            barBg.appendChild(barFill);
            row.appendChild(meta);
            row.appendChild(barBg);
            predictionsList.appendChild(row);

            // Animate bar width
            setTimeout(() => {
                barFill.style.width = `${Math.max(pred.confidence, 1.5)}%`;
            }, 60 * (index + 1));
        });
    }

    // --- Reset & Copy Actions ---
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            clearSelectedImage();
            setAppState('empty');
        });

        copyBtn.addEventListener('click', () => {
            if (!lastResult) return;

            const text = `FloraScan AI Pathology Report
----------------------------
Plant: ${lastResult.plant}
Diagnosis: ${lastResult.disease}
Status: ${lastResult.is_healthy ? 'Healthy' : 'Diseased'}
Severity: ${lastResult.severity}
Confidence: ${lastResult.confidence}%
Raw Class: ${lastResult.predicted_class}

Description:
${lastResult.description}

Recommended Action:
${lastResult.treatment}
`;

            navigator.clipboard.writeText(text).then(() => {
                copyBtnText.textContent = 'Copied!';
                setTimeout(() => {
                    copyBtnText.textContent = 'Copy Diagnosis';
                }, 2000);
            }).catch(() => {
                alert('Failed to copy to clipboard.');
            });
        });
    }

    // --- Send Email Report from Dashboard Results ---
    if (sendReportEmailBtn) {
        sendReportEmailBtn.addEventListener('click', async () => {
            const predId = lastResult?.prediction_id || null;
            await triggerEmailSend(predId, sendReportEmailBtn, emailStatusToast);
        });
    }

    // --- Send Latest Email from History Page Header ---
    if (emailLatestBtn) {
        emailLatestBtn.addEventListener('click', async () => {
            await triggerEmailSend(null, emailLatestBtn, historyEmailAlert);
        });
    }

    // Generic helper to trigger email sending
    async function triggerEmailSend(predictionId, buttonElement, toastElement) {
        const originalHtml = buttonElement ? buttonElement.innerHTML : '';
        if (buttonElement) {
            buttonElement.disabled = true;
            buttonElement.innerHTML = `<span>⏳ Sending report...</span>`;
        }

        if (toastElement) {
            toastElement.style.display = 'block';
            toastElement.style.background = 'rgba(255, 255, 255, 0.05)';
            toastElement.style.borderColor = 'var(--border-subtle)';
            toastElement.style.color = 'var(--text-secondary)';
            toastElement.innerHTML = `<span>⏳ Preparing and dispatching agronomic report via Resend...</span>`;
        }

        try {
            const res = await fetch('/api/send-email-report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prediction_id: predictionId })
            });
            const data = await res.json();

            if (data.success) {
                if (buttonElement) {
                    buttonElement.innerHTML = `<span>✓ Sent!</span>`;
                    setTimeout(() => {
                        buttonElement.innerHTML = originalHtml;
                        buttonElement.disabled = false;
                    }, 3500);
                }
                if (toastElement) {
                    toastElement.style.display = 'block';
                    toastElement.style.background = 'rgba(16, 185, 129, 0.12)';
                    toastElement.style.borderColor = '#10b981';
                    toastElement.style.color = '#34d399';
                    toastElement.innerHTML = `<strong>✓ Success:</strong> ${data.message}`;
                } else {
                    alert(`✓ Success: ${data.message}`);
                }
            } else {
                throw new Error(data.error || 'Failed to send email.');
            }
        } catch (err) {
            if (buttonElement) {
                buttonElement.innerHTML = `<span>⚠️ Send Failed</span>`;
                setTimeout(() => {
                    buttonElement.innerHTML = originalHtml;
                    buttonElement.disabled = false;
                }, 3500);
            }
            if (toastElement) {
                toastElement.style.display = 'block';
                toastElement.style.background = 'rgba(239, 68, 68, 0.12)';
                toastElement.style.borderColor = '#ef4444';
                toastElement.style.color = '#f87171';
                toastElement.innerHTML = `<strong>⚠️ Delivery Error:</strong> ${err.message}`;
            } else {
                alert(`⚠️ Delivery Error: ${err.message}`);
            }
        }
    }

    // --- LLM Chat Functionality ---
    async function sendChatMessage() {
        const msg = chatInput.value.trim();
        if (!msg) return;
        
        // Append User Msg
        appendChatMessage('user', msg);
        chatInput.value = '';
        
        // Disable input
        chatInput.disabled = true;
        chatSendBtn.disabled = true;
        
        try {
            const res = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg })
            });
            const data = await res.json();
            if (data.success) {
                appendChatMessage('ai', data.response);
            } else {
                appendChatMessage('ai', "Error processing request.");
            }
        } catch (e) {
            appendChatMessage('ai', "Error connecting to server.");
        } finally {
            chatInput.disabled = false;
            chatSendBtn.disabled = false;
            chatInput.focus();
        }
    }

    if (chatSendBtn) {
        chatSendBtn.addEventListener('click', sendChatMessage);
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }

    function appendChatMessage(role, content) {
        if (!chatMessages) return;
        const div = document.createElement('div');
        div.className = `chat-msg ${role}`;
        div.textContent = content;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // --- Load History ---
    async function loadHistory() {
        try {
            const res = await fetch('/history');
            const data = await res.json();
            if (data.success) {
                // Populate History
                if (historyList) {
                    historyList.innerHTML = '';
                    if (data.predictions && data.predictions.length > 0) {
                        if (historySection) historySection.style.display = 'block';
                        data.predictions.forEach(p => {
                            const item = document.createElement('div');
                            item.className = 'history-item';
                            item.style.display = 'flex';
                            item.style.justifyContent = 'space-between';
                            item.style.alignItems = 'center';
                            item.style.gap = '1rem';
                            item.innerHTML = `
                                <div style="flex: 1;">
                                    <div class="plant" style="font-weight: 600; color: var(--text-primary); font-size: 1rem;">${p.plant}</div>
                                    <div class="disease" style="font-size: 0.88rem; color: var(--text-secondary); margin-top: 0.15rem;">${p.disease} <span style="color: var(--primary); font-weight: 500;">(${p.confidence.toFixed(1)}%)</span></div>
                                    <div class="date" style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem;">${new Date(p.timestamp).toLocaleString()}</div>
                                </div>
                                <div>
                                    <button type="button" class="btn btn-secondary send-hist-email-btn" data-id="${p.id}" style="padding: 0.45rem 0.85rem; font-size: 0.8rem; background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); color: #34d399; display: inline-flex; align-items: center; gap: 0.35rem; cursor: pointer;">
                                        <span>✉️ Email Report</span>
                                    </button>
                                </div>
                            `;
                            historyList.appendChild(item);
                        });

                        // Attach handlers to each history item's email button
                        historyList.querySelectorAll('.send-hist-email-btn').forEach(btn => {
                            btn.addEventListener('click', async (e) => {
                                e.stopPropagation();
                                const predId = parseInt(btn.getAttribute('data-id'), 10);
                                await triggerEmailSend(predId, btn, historyEmailAlert);
                            });
                        });
                    } else {
                        historyList.innerHTML = '<div class="history-empty" style="text-align:center;padding:2rem;color:var(--text-muted);opacity:0.7;">No predictions yet. Analyze a leaf to get started.</div>';
                    }
                }
                
                // Populate Chat
                if (chatMessages && data.chat_history && data.chat_history.length > 0) {
                    chatMessages.innerHTML = '';
                    data.chat_history.forEach(msg => {
                        appendChatMessage(msg.role, msg.content);
                    });
                }
            }
        } catch (e) {
            console.error("Failed to load history", e);
        }
    }
    
    // Init on load
    loadHistory();
});
