# Test script for Gold-YOLO Jittor implementation

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jittor as jt
import numpy as np

from configs.gold_yolo_s import get_config
from models.yolo import build_model


def test_model_structure():
    """Test Gold-YOLO-s model structure and forward pass"""
    print("🐾 Testing Gold-YOLO-s Jittor implementation...")

    # Get configuration
    config = get_config()
    print(f"✅ Config loaded: {config.model.type}")

    # Build model
    num_classes = 80  # COCO classes
    try:
        model = build_model(config, num_classes)
        print(f"✅ Model built successfully")
        print(f"   - Backbone: {config.model.backbone.type}")
        print(f"   - Neck: {config.model.neck.type}")
        print(f"   - Head: {config.model.head.type}")
    except Exception as e:
        print(f"❌ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass
    try:
        # Create dummy input
        batch_size = 1
        input_size = 640
        x = jt.randn(batch_size, 3, input_size, input_size)
        print(f"✅ Input tensor created: {x.shape}")

        # Test backbone only first
        print("🔧 Testing backbone...")
        backbone_out = model.backbone(x)
        print(f"✅ Backbone forward pass successful!")
        print(f"   - Backbone outputs: {len(backbone_out)} levels")
        total_channels = 0
        for i, feat in enumerate(backbone_out):
            channels = feat.shape[1]
            total_channels += channels
            print(f"     Level {i}: {feat.shape} (channels: {channels})")
        print(f"   - Total channels for fusion: {total_channels}")

        # Test neck
        print("🔧 Testing neck...")
        try:
            neck_out = model.neck(backbone_out)
            print(f"✅ Neck forward pass successful!")
            print(f"   - Neck outputs: {len(neck_out)} levels")
            for i, feat in enumerate(neck_out):
                print(f"     Level {i}: {feat.shape}")
        except Exception as e:
            print(f"❌ Neck forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Test head
        print("🔧 Testing head...")
        try:
            head_out = model.detect(neck_out)
            print(f"✅ Head forward pass successful!")
            if isinstance(head_out, (list, tuple)):
                print(f"   - Head outputs: {len(head_out)} items")
                for i, item in enumerate(head_out):
                    if hasattr(item, 'shape'):
                        print(f"     Item {i}: {item.shape}")
                    else:
                        print(f"     Item {i}: {type(item)}")
            else:
                print(f"   - Head output shape: {head_out.shape}")
        except Exception as e:
            print(f"❌ Head forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_components():
    """Test individual model components"""
    print("\n🔧 Testing individual components...")
    
    config = get_config()
    
    # Test backbone
    try:
        from models.backbone import EfficientRep
        backbone = EfficientRep(
            in_channels=3,
            channels_list=[32, 64, 128, 256, 512, 512],  # Simplified for test
            num_repeats=[1, 2, 3, 4, 2, 1],
            fuse_P2=True,
            cspsppf=True
        )
        
        x = jt.randn(1, 3, 640, 640)
        backbone_out = backbone(x)
        print(f"✅ Backbone test passed: {len(backbone_out)} outputs")
        for i, out in enumerate(backbone_out):
            print(f"   Output {i}: {out.shape}")
            
    except Exception as e:
        print(f"❌ Backbone test failed: {e}")
        return False
    
    return True


def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 Gold-YOLO Jittor Implementation Test")
    print("=" * 60)
    
    # Test components
    if not test_model_components():
        print("\n❌ Component tests failed!")
        return
    
    # Test full model
    if test_model_structure():
        print("\n🎉 All tests passed! Gold-YOLO-s Jittor implementation is working!")
    else:
        print("\n❌ Model tests failed!")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
