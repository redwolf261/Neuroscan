# MS Detector Web App - Quick Setup Script
# Run this script to set up the entire project

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MS Detector - Quick Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
Write-Host "Checking Docker installation..." -ForegroundColor Yellow
try {
    docker --version | Out-Null
    Write-Host "✓ Docker is installed" -ForegroundColor Green
    $useDocker = Read-Host "Would you like to use Docker? (y/n)"
} catch {
    Write-Host "✗ Docker is not installed" -ForegroundColor Red
    $useDocker = "n"
}

if ($useDocker -eq "y") {
    # Docker setup
    Write-Host ""
    Write-Host "Setting up with Docker..." -ForegroundColor Yellow
    
    # Create models directory
    if (!(Test-Path ".\models")) {
        New-Item -ItemType Directory -Path ".\models" | Out-Null
        Write-Host "✓ Created models directory" -ForegroundColor Green
    }
    
    # Check for model file
    $modelPath = "G:\My Drive\NeuroScan_PEDiMS_v2\deployment\model_deployable.pth"
    if (Test-Path $modelPath) {
        Write-Host "Copying model file..." -ForegroundColor Yellow
        Copy-Item $modelPath ".\models\" -Force
        Write-Host "✓ Model file copied" -ForegroundColor Green
    } else {
        Write-Host "⚠ Model file not found at: $modelPath" -ForegroundColor Yellow
        Write-Host "Please ensure the model file is in the .\models\ directory" -ForegroundColor Yellow
    }
    
    # Build and start containers
    Write-Host ""
    Write-Host "Building and starting Docker containers..." -ForegroundColor Yellow
    docker-compose up --build -d
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Access the application at:" -ForegroundColor White
    Write-Host "  Frontend: http://localhost" -ForegroundColor Cyan
    Write-Host "  Backend API: http://localhost:5000" -ForegroundColor Cyan
    Write-Host "  Health Check: http://localhost:5000/health" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To view logs: docker-compose logs -f" -ForegroundColor Yellow
    Write-Host "To stop: docker-compose down" -ForegroundColor Yellow
    
} else {
    # Manual setup
    Write-Host ""
    Write-Host "Setting up manually..." -ForegroundColor Yellow
    
    # Backend setup
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Backend Setup" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    Push-Location backend
    
    # Create virtual environment
    if (!(Test-Path ".\venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    }
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
    
    # Install dependencies
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
    
    # Create models directory
    if (!(Test-Path ".\models")) {
        New-Item -ItemType Directory -Path ".\models" | Out-Null
        Write-Host "✓ Created models directory" -ForegroundColor Green
    }
    
    # Check for model file
    $modelPath = "G:\My Drive\NeuroScan_PEDiMS_v2\deployment\model_deployable.pth"
    if (Test-Path $modelPath) {
        Write-Host "Copying model file..." -ForegroundColor Yellow
        Copy-Item $modelPath ".\models\" -Force
        Write-Host "✓ Model file copied" -ForegroundColor Green
    } else {
        Write-Host "⚠ Model file not found at: $modelPath" -ForegroundColor Yellow
        Write-Host "Please copy the model file to: backend\models\" -ForegroundColor Yellow
    }
    
    Pop-Location
    
    # Frontend setup
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Frontend Setup" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    
    Push-Location frontend
    
    # Check if node is installed
    try {
        node --version | Out-Null
        Write-Host "✓ Node.js is installed" -ForegroundColor Green
        
        # Install dependencies
        Write-Host "Installing Node dependencies..." -ForegroundColor Yellow
        npm install
        Write-Host "✓ Dependencies installed" -ForegroundColor Green
        
    } catch {
        Write-Host "✗ Node.js is not installed" -ForegroundColor Red
        Write-Host "Please install Node.js from: https://nodejs.org/" -ForegroundColor Yellow
    }
    
    Pop-Location
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Setup Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To start the application:" -ForegroundColor White
    Write-Host ""
    Write-Host "1. Start Backend (in backend directory):" -ForegroundColor Cyan
    Write-Host "   .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "   python app.py" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "2. Start Frontend (in frontend directory, new terminal):" -ForegroundColor Cyan
    Write-Host "   npm start" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Access the application at:" -ForegroundColor White
    Write-Host "  Frontend: http://localhost:3000" -ForegroundColor Cyan
    Write-Host "  Backend API: http://localhost:5000" -ForegroundColor Cyan
}

Write-Host ""
