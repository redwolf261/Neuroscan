"""
Test script for MS Detection API
Tests health check and basic connectivity
"""

import requests
import sys

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Device: {data.get('device')}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        print("   Start server with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing /model-info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved!")
            print(f"   Model: {data.get('model_name')}")
            print(f"   Architecture: {data.get('architecture')}")
            print(f"   Input channels: {data.get('input_channels')}")
            print(f"   Modalities: {data.get('modalities')}")
            print(f"   Input size: {data.get('input_size')}")
            print(f"   Device: {data.get('device')}")
            
            training = data.get('training', {})
            print(f"\n   Training info:")
            print(f"     MAE epochs: {training.get('mae_epochs')}")
            print(f"     Seg epochs: {training.get('segmentation_epochs')}")
            print(f"     Best epoch: {training.get('best_epoch')}")
            
            performance = data.get('performance', {})
            if performance:
                print(f"\n   Performance metrics:")
                print(f"     Val Dice: {performance.get('val_dice', 0):.4f} ({performance.get('val_dice', 0)*100:.2f}%)")
                print(f"     Recall: {performance.get('recall', 0):.4f} ({performance.get('recall', 0)*100:.2f}%)")
                print(f"     Precision: {performance.get('precision', 0):.4f} ({performance.get('precision', 0)*100:.2f}%)")
                print(f"     F1 Score: {performance.get('f1_score', 0):.4f} ({performance.get('f1_score', 0)*100:.2f}%)")
            
            return True
        else:
            print(f"❌ Model info failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict_validation():
    """Test predict endpoint validation (without actual files)"""
    print("\n🔍 Testing /predict endpoint validation...")
    try:
        # Test without files (should fail gracefully)
        response = requests.post(f"{BASE_URL}/predict", timeout=5)
        
        if response.status_code == 400:
            data = response.json()
            print("✅ Validation working correctly!")
            print(f"   Expected error: {data.get('error')}")
            return True
        else:
            print(f"⚠️  Unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("="*70)
    print("🏥 MS Detection API Test Suite")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Model Info", test_model_info()))
    results.append(("Predict Validation", test_predict_validation()))
    
    # Summary
    print("\n" + "="*70)
    print("📊 Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✨ All tests passed! API is ready for use.")
        print("\n📖 Next steps:")
        print("   1. Test with actual NIfTI files using inference.py")
        print("   2. Test multi-modal upload (3 files)")
        print("   3. Test single-modal upload (1 file)")
        print("   4. Check API_DOCUMENTATION.md for usage examples")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the server logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
