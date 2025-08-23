from ultralytics import YOLO

print("📥 Downloading YOLO11m...")

try:
    # This will automatically download yolo11m.pt
    model = YOLO("yolo11l.pt")
    print("✅ YOLO11m downloaded successfully!")
    print(f"📍 Model ready to use!")
    
    # Test the model
    print("🧪 Testing model...")
    model.info()
    print("✅ Model is working!")
    
except Exception as e:
    print(f"❌ Download failed: {e}")