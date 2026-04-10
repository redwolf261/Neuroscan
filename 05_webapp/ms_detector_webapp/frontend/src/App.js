import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import './App.css';

function App() {
  const [flairFile, setFlairFile] = useState(null);
  const [t1File, setT1File] = useState(null);
  const [t2File, setT2File] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [mode, setMode] = useState('multi'); // 'multi' or 'single'

  // Fetch model info on mount
  React.useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await axios.get('http://localhost:5000/model-info');
        setModelInfo(response.data);
      } catch (err) {
        console.error('Failed to fetch model info:', err);
      }
    };
    fetchModelInfo();
  }, []);

  // Multi-modal dropzones
  const onDropFlair = useCallback((acceptedFiles) => {
    console.log('FLAIR files dropped:', acceptedFiles);
    if (acceptedFiles.length > 0) {
      setFlairFile(acceptedFiles[0]);
      setResult(null);
      setError(null);
    }
  }, []);

  const onDropT1 = useCallback((acceptedFiles) => {
    console.log('T1 files dropped:', acceptedFiles);
    if (acceptedFiles.length > 0) {
      setT1File(acceptedFiles[0]);
      setResult(null);
      setError(null);
    }
  }, []);

  const onDropT2 = useCallback((acceptedFiles) => {
    console.log('T2 files dropped:', acceptedFiles);
    if (acceptedFiles.length > 0) {
      setT2File(acceptedFiles[0]);
      setResult(null);
      setError(null);
    }
  }, []);

  // Single file dropzone
  const onDropSingle = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFlairFile(acceptedFiles[0]); // Reuse flairFile for single mode
      setResult(null);
      setError(null);
    }
  }, []);

  const dropzoneConfig = {
    accept: {
      'application/octet-stream': ['.nii', '.nii.gz'],
      'application/gzip': ['.nii.gz'],
      'application/x-gzip': ['.nii.gz']
    },
    maxFiles: 1,
    noKeyboard: true,
    multiple: false,
    // Custom validator to allow .nii and .nii.gz files regardless of MIME type
    validator: (file) => {
      const validExtensions = ['.nii', '.nii.gz'];
      const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
      if (!hasValidExtension) {
        return {
          code: "invalid-file-type",
          message: "Only .nii or .nii.gz files are accepted"
        };
      }
      return null;
    }
  };

  const flairDropzone = useDropzone({ 
    ...dropzoneConfig, 
    onDrop: onDropFlair,
    onDropRejected: (fileRejections) => {
      console.log('FLAIR files rejected:', fileRejections);
      if (fileRejections.length > 0) {
        const errors = fileRejections[0].errors.map(e => e.message).join(', ');
        setError(`FLAIR file rejected: ${errors}`);
      }
    },
    disabled: false
  });
  const t1Dropzone = useDropzone({ 
    ...dropzoneConfig, 
    onDrop: onDropT1,
    onDropRejected: (fileRejections) => {
      console.log('T1 files rejected:', fileRejections);
      if (fileRejections.length > 0) {
        const errors = fileRejections[0].errors.map(e => e.message).join(', ');
        setError(`T1 file rejected: ${errors}`);
      }
    },
    disabled: false
  });
  const t2Dropzone = useDropzone({ 
    ...dropzoneConfig, 
    onDrop: onDropT2,
    onDropRejected: (fileRejections) => {
      console.log('T2 files rejected:', fileRejections);
      if (fileRejections.length > 0) {
        const errors = fileRejections[0].errors.map(e => e.message).join(', ');
        setError(`T2 file rejected: ${errors}`);
      }
    },
    disabled: false
  });
  const singleDropzone = useDropzone({ 
    ...dropzoneConfig, 
    onDrop: onDropSingle,
    onDropRejected: (fileRejections) => {
      console.log('Single file rejected:', fileRejections);
      if (fileRejections.length > 0) {
        const errors = fileRejections[0].errors.map(e => e.message).join(', ');
        setError(`File rejected: ${errors}`);
      }
    },
    disabled: false
  });

  const handleUpload = async () => {
    if (mode === 'multi' && (!flairFile || !t1File || !t2File)) {
      setError('Please upload all three modalities (FLAIR, T1, T2)');
      return;
    }
    if (mode === 'single' && !flairFile) {
      setError('Please upload a file');
      return;
    }

    setUploading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    
    if (mode === 'multi') {
      formData.append('flair_file', flairFile);
      formData.append('t1_file', t1File);
      formData.append('t2_file', t2File);
    } else {
      formData.append('file', flairFile);
    }

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.success) {
        setResult(response.data.result);
      } else {
        setError(response.data.error || 'Prediction failed');
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to connect to server. Please ensure the backend is running.');
    } finally {
      setUploading(false);
    }
  };

  const resetUpload = () => {
    setFlairFile(null);
    setT1File(null);
    setT2File(null);
    setResult(null);
    setError(null);
  };

  const getSeverityColor = (severity) => {
    switch(severity?.toLowerCase()) {
      case 'minimal/none': return 'severity-minimal';
      case 'mild': return 'severity-mild';
      case 'moderate': return 'severity-moderate';
      case 'severe': return 'severity-severe';
      default: return '';
    }
  };

  const FileUploadBox = ({ dropzone, file, setFile, label, icon }) => {
    const fileInputRef = React.useRef(null);
    
    const handleFileSelect = (e) => {
      const selectedFile = e.target.files[0];
      console.log(`${label} file selected:`, selectedFile);
      if (selectedFile) {
        setFile(selectedFile);
        setResult(null);
        setError(null);
      }
    };
    
    const handleBoxClick = () => {
      console.log(`${label} box clicked`);
      fileInputRef.current?.click();
    };
    
    return (
      <div className="upload-box">
        <label className="upload-label">
          <span className="modality-icon">{icon}</span>
          {label}
        </label>
        <div
          onClick={handleBoxClick}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            const droppedFile = e.dataTransfer.files[0];
            console.log(`${label} file dropped:`, droppedFile);
            if (droppedFile) {
              setFile(droppedFile);
              setResult(null);
              setError(null);
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          className={`dropzone-small ${file ? 'has-file' : ''}`}
        >
          <input 
            ref={fileInputRef}
            type="file"
            accept=".nii,.nii.gz"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />
          {file ? (
            <div className="file-info-compact">
              <span className="file-icon-small">✓</span>
              <div className="file-details-small">
                <p className="file-name-small">{file.name}</p>
                <p className="file-size-small">{(file.size / (1024 * 1024)).toFixed(1)} MB</p>
              </div>
              <button 
                className="btn-remove-small" 
                onClick={(e) => { 
                  e.stopPropagation(); 
                  e.preventDefault();
                  console.log(`${label} file removed`);
                  setFile(null); 
                }}
                type="button"
              >
                ✕
              </button>
            </div>
          ) : (
            <div className="dropzone-content-small">
              <span className="upload-icon-small">☁️</span>
              <p>Drop {label} here</p>
              <small>or click to browse</small>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <span className="logo-icon">🧠</span>
            <h1>MS Lesion Detector</h1>
          </div>
          <p className="tagline">AI-Powered Multiple Sclerosis Lesion Detection from Multi-Modal MRI</p>
          {modelInfo && (
            <div className="model-badge">
              <span className="badge-highlight">{(modelInfo.performance.val_dice * 100).toFixed(1)}%</span> Dice Score
              <span className="badge-separator">|</span>
              <span className="badge-highlight">{(modelInfo.performance.recall * 100).toFixed(1)}%</span> Recall
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="container">
          {!result ? (
            <>
              {/* Mode Selector */}
              <div className="mode-selector">
                <button 
                  className={`mode-btn ${mode === 'multi' ? 'active' : ''}`}
                  onClick={() => { setMode('multi'); resetUpload(); }}
                >
                  <span className="mode-icon">🎯</span>
                  <div className="mode-info">
                    <strong>Multi-Modal (Recommended)</strong>
                    <small>FLAIR + T1 + T2 - Best Accuracy</small>
                  </div>
                </button>
                <button 
                  className={`mode-btn ${mode === 'single' ? 'active' : ''}`}
                  onClick={() => { setMode('single'); resetUpload(); }}
                >
                  <span className="mode-icon">📄</span>
                  <div className="mode-info">
                    <strong>Single Modality</strong>
                    <small>Any modality - Quick Test</small>
                  </div>
                </button>
              </div>

              {/* Upload Section */}
              <div className="upload-section">
                <h2>Upload MRI Scans</h2>
                <p className="subtitle">
                  {mode === 'multi' 
                    ? 'Upload FLAIR, T1, and T2 MRI scans (.nii or .nii.gz format)'
                    : 'Upload any MRI scan (.nii or .nii.gz format)'}
                </p>
                <div className="info-banner">
                  <span className="info-icon">ℹ️</span>
                  <p>
                    <strong>Important:</strong> This model is trained on <strong>HUMAN pediatric MS brain MRI scans</strong>. 
                    Only upload genuine human brain medical MRI scans in NIfTI format (.nii or .nii.gz). 
                    Do not upload: (1) Converted regular images (JPG, PNG), (2) Animal/veterinary brain scans, or (3) Non-brain MRI scans. 
                    Use test samples from the <code>testing/</code> folder (P1-P9 human patients) for demo purposes.
                  </p>
                </div>

                {mode === 'multi' ? (
                  <div className="multi-upload-grid">
                    <FileUploadBox 
                      dropzone={flairDropzone} 
                      file={flairFile} 
                      setFile={setFlairFile} 
                      label="FLAIR" 
                      icon="🔵"
                    />
                    <FileUploadBox 
                      dropzone={t1Dropzone} 
                      file={t1File} 
                      setFile={setT1File} 
                      label="T1-Weighted" 
                      icon="🟢"
                    />
                    <FileUploadBox 
                      dropzone={t2Dropzone} 
                      file={t2File} 
                      setFile={setT2File} 
                      label="T2-Weighted" 
                      icon="🟠"
                    />
                  </div>
                ) : (
                  <div
                    onClick={() => {
                      console.log('Single mode box clicked');
                      document.getElementById('single-file-input').click();
                    }}
                    onDrop={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      const droppedFile = e.dataTransfer.files[0];
                      console.log('Single file dropped:', droppedFile);
                      if (droppedFile) {
                        setFlairFile(droppedFile);
                        setResult(null);
                        setError(null);
                      }
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                    }}
                    className={`dropzone ${flairFile ? 'has-file' : ''}`}
                  >
                    <input 
                      id="single-file-input"
                      type="file"
                      accept=".nii,.nii.gz"
                      onChange={(e) => {
                        const selectedFile = e.target.files[0];
                        console.log('Single file selected:', selectedFile);
                        if (selectedFile) {
                          setFlairFile(selectedFile);
                          setResult(null);
                          setError(null);
                        }
                      }}
                      style={{ display: 'none' }}
                    />
                    {flairFile ? (
                      <div className="file-info">
                        <span className="file-icon">📄</span>
                        <div className="file-details">
                          <p className="file-name">{flairFile.name}</p>
                          <p className="file-size">{(flairFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                        </div>
                        <button className="btn-remove" onClick={(e) => { 
                          e.stopPropagation(); 
                          console.log('Single file removed');
                          resetUpload(); 
                        }}>
                          ✕
                        </button>
                      </div>
                    ) : (
                      <div className="dropzone-content">
                        <span className="upload-icon">☁️</span>
                        <p className="dropzone-text">Drag & drop your MRI scan here</p>
                        <p className="dropzone-subtext">or click to browse</p>
                        <p className="file-formats">Supported: .nii, .nii.gz</p>
                      </div>
                    )}
                  </div>
                )}

                {((mode === 'multi' && flairFile && t1File && t2File) || (mode === 'single' && flairFile)) && (
                  <button
                    className="btn-primary btn-analyze"
                    onClick={handleUpload}
                    disabled={uploading}
                  >
                    {uploading ? (
                      <>
                        <span className="spinner"></span>
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <span>🔍</span>
                        Analyze Scans
                      </>
                    )}
                  </button>
                )}

                {error && (
                  <div className="error-message">
                    <span className="error-icon">⚠️</span>
                    <p>{error}</p>
                  </div>
                )}
              </div>

              {/* Info Cards */}
              <div className="info-cards">
                <div className="info-card">
                  <span className="card-icon">⚡</span>
                  <h3>Fast Analysis</h3>
                  <p>Get results in ~2 seconds with our optimized AI model</p>
                </div>
                <div className="info-card">
                  <span className="card-icon">🎯</span>
                  <h3>High Accuracy</h3>
                  <p>{modelInfo ? `${(modelInfo.performance.val_dice * 100).toFixed(1)}%` : '83.99%'} Dice score - Production quality</p>
                </div>
                <div className="info-card">
                  <span className="card-icon">🔒</span>
                  <h3>Private & Secure</h3>
                  <p>Your data is processed locally and never stored</p>
                </div>
              </div>
            </>
          ) : (
            /* Results Section */
            <div className="results-section">
              <div className="results-header">
                <h2>Analysis Results</h2>
                <button className="btn-secondary" onClick={resetUpload}>
                  ← Upload New Scan
                </button>
              </div>

              {/* Main Result Card */}
              <div className={`result-card ${result.has_ms ? 'positive' : 'negative'}`}>
                <div className="result-icon">
                  {result.has_ms ? '⚠️' : '✅'}
                </div>
                <h3 className="result-title">
                  {result.has_ms ? 'MS Lesions Detected' : 'No Significant Lesions Detected'}
                </h3>
                <p className="result-subtitle">
                  {result.has_ms
                    ? 'The scan shows evidence of lesions consistent with Multiple Sclerosis'
                    : 'The scan does not show significant lesions indicative of Multiple Sclerosis'}
                </p>
                {result.severity && (
                  <div className={`severity-badge ${getSeverityColor(result.severity)}`}>
                    Severity: <strong>{result.severity}</strong>
                  </div>
                )}
                {result.warning && (
                  <div className="warning-badge">
                    ⚠️ {result.warning}
                  </div>
                )}
              </div>

              {/* Metrics Grid */}
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Average Confidence</div>
                  <div className="metric-value">{result.confidence}%</div>
                  <div className="metric-bar">
                    <div
                      className="metric-bar-fill"
                      style={{ width: `${result.confidence}%` }}
                    ></div>
                  </div>
                </div>

                {result.max_confidence && (
                  <div className="metric-card">
                    <div className="metric-label">Peak Confidence</div>
                    <div className="metric-value">{result.max_confidence}%</div>
                    <div className="metric-description">maximum prediction</div>
                  </div>
                )}

                {result.num_lesions !== undefined && (
                  <div className="metric-card">
                    <div className="metric-label">Lesion Count</div>
                    <div className="metric-value">{result.num_lesions}</div>
                    <div className="metric-description">detected regions</div>
                  </div>
                )}

                {result.lesion_volume_ml && (
                  <div className="metric-card">
                    <div className="metric-label">Lesion Volume</div>
                    <div className="metric-value">{result.lesion_volume_ml} mL</div>
                    <div className="metric-description">total volume</div>
                  </div>
                )}

                <div className="metric-card">
                  <div className="metric-label">Lesion Load</div>
                  <div className="metric-value">{result.lesion_volume_percentage}%</div>
                  <div className="metric-description">of brain volume</div>
                </div>

                <div className="metric-card">
                  <div className="metric-label">Total Voxels</div>
                  <div className="metric-value">{result.total_lesion_voxels.toLocaleString()}</div>
                  <div className="metric-description">lesion voxels</div>
                </div>

                <div className="metric-card">
                  <div className="metric-label">Scan Dimensions</div>
                  <div className="metric-value dimension">
                    {result.scan_dimensions.join(' × ')}
                  </div>
                  <div className="metric-description">original size</div>
                </div>

                <div className="metric-card">
                  <div className="metric-label">Processed Size</div>
                  <div className="metric-value dimension">
                    {result.processed_dimensions.join(' × ')}
                  </div>
                  <div className="metric-description">model input</div>
                </div>
              </div>

              {/* Disclaimer */}
              <div className="disclaimer">
                <h4>⚕️ Medical Disclaimer</h4>
                <p>
                  This AI system is designed to assist medical professionals and should not be used as the sole basis for diagnosis.
                  Always consult with a qualified healthcare provider for proper medical evaluation and diagnosis.
                  The results provided are for research and educational purposes. Model performance: {modelInfo ? `${(modelInfo.performance.val_dice * 100).toFixed(1)}% Dice, ${(modelInfo.performance.recall * 100).toFixed(1)}% Recall` : '83.99% Dice, 91.64% Recall'}.
                </p>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-content">
          <p>
            Powered by <strong>{modelInfo ? modelInfo.model_name : 'HybridMiniSwin2.5D-ResNet with CSRF and 2.5D-MAE'}</strong>
          </p>
          <p className="footer-stats">
            {modelInfo ? (
              <>
                Dice: {(modelInfo.performance.val_dice * 100).toFixed(2)}% | 
                Recall: {(modelInfo.performance.recall * 100).toFixed(2)}% | 
                Precision: {(modelInfo.performance.precision * 100).toFixed(2)}% | 
                Trained: {modelInfo.training.segmentation_epochs} epochs (early stopped)
              </>
            ) : (
              'Model: 83.99% Dice Score | 91.64% Recall | 77.60% Precision'
            )}
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
