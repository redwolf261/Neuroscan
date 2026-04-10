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
    if (acceptedFiles.length > 0) {
      setFlairFile(acceptedFiles[0]);
      setResult(null);
      setError(null);
    }
  }, []);

  const onDropT1 = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setT1File(acceptedFiles[0]);
      setResult(null);
      setError(null);
    }
  }, []);

  const onDropT2 = useCallback((acceptedFiles) => {
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
      'application/octet-stream': ['.nii', '.nii.gz']
    },
    maxFiles: 1
  };

  const flairDropzone = useDropzone({ ...dropzoneConfig, onDrop: onDropFlair });
  const t1Dropzone = useDropzone({ ...dropzoneConfig, onDrop: onDropT1 });
  const t2Dropzone = useDropzone({ ...dropzoneConfig, onDrop: onDropT2 });
  const singleDropzone = useDropzone({ ...dropzoneConfig, onDrop: onDropSingle });

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
              <span className="badge-highlight">{modelInfo.performance.val_dice * 100}%</span> Dice Score
              <span className="badge-separator">|</span>
              <span className="badge-highlight">{modelInfo.performance.recall * 100}%</span> Recall
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

                {mode === 'multi' ? (
                  <div className="multi-upload-grid">
                    {/* FLAIR Upload */}
                    <div className="upload-box">
                      <label className="upload-label">
                        <span className="modality-icon">🔵</span>
                        FLAIR
                      </label>
                      <div
                        {...flairDropzone.getRootProps()}
                        className={`dropzone-small ${flairDropzone.isDragActive ? 'active' : ''} ${flairFile ? 'has-file' : ''}`}
                      >
                        <input {...flairDropzone.getInputProps()} />
                        {flairFile ? (
                          <div className="file-info-compact">
                            <span className="file-icon-small">✓</span>
                            <div className="file-details-small">
                              <p className="file-name-small">{flairFile.name}</p>
                              <p className="file-size-small">{(flairFile.size / (1024 * 1024)).toFixed(1)} MB</p>
                            </div>
                            <button className="btn-remove-small" onClick={(e) => { e.stopPropagation(); setFlairFile(null); }}>
                              ✕
                            </button>
                          </div>
                        ) : (
                          <div className="dropzone-content-small">
                            <span className="upload-icon-small">☁️</span>
                            <p>Drop FLAIR here</p>
                            <small>or click to browse</small>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* T1 Upload */}
                    <div className="upload-box">
                      <label className="upload-label">
                        <span className="modality-icon">🟢</span>
                        T1-Weighted
                      </label>
                      <div
                        {...t1Dropzone.getRootProps()}
                        className={`dropzone-small ${t1Dropzone.isDragActive ? 'active' : ''} ${t1File ? 'has-file' : ''}`}
                      >
                        <input {...t1Dropzone.getInputProps()} />
                        {t1File ? (
                          <div className="file-info-compact">
                            <span className="file-icon-small">✓</span>
                            <div className="file-details-small">
                              <p className="file-name-small">{t1File.name}</p>
                              <p className="file-size-small">{(t1File.size / (1024 * 1024)).toFixed(1)} MB</p>
                            </div>
                            <button className="btn-remove-small" onClick={(e) => { e.stopPropagation(); setT1File(null); }}>
                              ✕
                            </button>
                          </div>
                        ) : (
                          <div className="dropzone-content-small">
                            <span className="upload-icon-small">☁️</span>
                            <p>Drop T1 here</p>
                            <small>or click to browse</small>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* T2 Upload */}
                    <div className="upload-box">
                      <label className="upload-label">
                        <span className="modality-icon">🟠</span>
                        T2-Weighted
                      </label>
                      <div
                        {...t2Dropzone.getRootProps()}
                        className={`dropzone-small ${t2Dropzone.isDragActive ? 'active' : ''} ${t2File ? 'has-file' : ''}`}
                      >
                        <input {...t2Dropzone.getInputProps()} />
                        {t2File ? (
                          <div className="file-info-compact">
                            <span className="file-icon-small">✓</span>
                            <div className="file-details-small">
                              <p className="file-name-small">{t2File.name}</p>
                              <p className="file-size-small">{(t2File.size / (1024 * 1024)).toFixed(1)} MB</p>
                            </div>
                            <button className="btn-remove-small" onClick={(e) => { e.stopPropagation(); setT2File(null); }}>
                              ✕
                            </button>
                          </div>
                        ) : (
                          <div className="dropzone-content-small">
                            <span className="upload-icon-small">☁️</span>
                            <p>Drop T2 here</p>
                            <small>or click to browse</small>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div
                    {...singleDropzone.getRootProps()}
                    className={`dropzone ${singleDropzone.isDragActive ? 'active' : ''} ${flairFile ? 'has-file' : ''}`}
                  >
                    <input {...singleDropzone.getInputProps()} />
                    {flairFile ? (
                      <div className="file-info">
                        <span className="file-icon">📄</span>
                        <div className="file-details">
                          <p className="file-name">{flairFile.name}</p>
                          <p className="file-size">{(flairFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                        </div>
                        <button className="btn-remove" onClick={(e) => { e.stopPropagation(); resetUpload(); }}>
                          ✕
                        </button>
                      </div>
                    ) : singleDropzone.isDragActive ? (
                      <p className="dropzone-text">📥 Drop your file here</p>
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
                    </div>
                  ) : isDragActive ? (
                    <p className="dropzone-text">📥 Drop your file here</p>
                  ) : (
                    <div className="dropzone-content">
                      <span className="upload-icon">☁️</span>
                      <p className="dropzone-text">Drag & drop your MRI scan here</p>
                      <p className="dropzone-subtext">or click to browse</p>
                      <p className="file-formats">Supported: .nii, .nii.gz</p>
                    </div>
                  )}
                </div>

                {file && (
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
                        Analyze Scan
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
                  <p>Get results in seconds with our optimized AI model</p>
                </div>
                <div className="info-card">
                  <span className="card-icon">🎯</span>
                  <h3>High Accuracy</h3>
                  <p>74.5% Dice score on validation dataset</p>
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
                  ← Upload Another Scan
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
              </div>

              {/* Metrics Grid */}
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">Confidence</div>
                  <div className="metric-value">{result.confidence}%</div>
                  <div className="metric-bar">
                    <div
                      className="metric-bar-fill"
                      style={{ width: `${result.confidence}%` }}
                    ></div>
                  </div>
                </div>

                <div className="metric-card">
                  <div className="metric-label">Lesion Volume</div>
                  <div className="metric-value">{result.lesion_volume_percentage}%</div>
                  <div className="metric-description">of brain volume</div>
                </div>

                <div className="metric-card">
                  <div className="metric-label">Total Lesion Voxels</div>
                  <div className="metric-value">{result.total_lesion_voxels.toLocaleString()}</div>
                  <div className="metric-description">detected voxels</div>
                </div>

                <div className="metric-card">
                  <div className="metric-label">Scan Dimensions</div>
                  <div className="metric-value dimension">
                    {result.scan_dimensions.join(' × ')}
                  </div>
                  <div className="metric-description">original size</div>
                </div>
              </div>

              {/* Disclaimer */}
              <div className="disclaimer">
                <h4>⚕️ Medical Disclaimer</h4>
                <p>
                  This AI system is designed to assist medical professionals and should not be used as the sole basis for diagnosis.
                  Always consult with a qualified healthcare provider for proper medical evaluation and diagnosis.
                  The results provided are for research and educational purposes.
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
            Powered by <strong>HybridMiniSwin3D</strong> (PATH 3 BALANCED) | Trained on PediMS Dataset
          </p>
          <p className="footer-stats">
            Model: 74.5% Dice Score | 200 Training Epochs | RTX 2050 Optimized
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
