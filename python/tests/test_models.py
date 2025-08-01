try:
    import viennaps2d
except ImportError:
    print("ERROR: Python bindings for viennaps2d are not available")
    exit()

import viennaps2d as vps


def test_sf6o2_etching():
    """Test SF6O2 etching model"""
    print("Testing SF6O2Etching model...")
    try:
        model = vps.SF6O2Etching()
        print("SF6O2Etching model created successfully")
    except RuntimeError as e:
        print(f"Expected error for SF6O2Etching: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def test_other_models():
    """Test other available models"""
    print("Testing other models...")

    # Test if other models are available
    model_names = []
    for attr_name in dir(vps):
        if "Etching" in attr_name or "Deposition" in attr_name:
            model_names.append(attr_name)

    print(f"Available models: {model_names}")

    for model_name in model_names:
        try:
            model_class = getattr(vps, model_name)
            if callable(model_class):
                print(f"Testing {model_name}...")
                model = model_class()
                print(f"{model_name} created successfully")
        except Exception as e:
            print(f"Error with {model_name}: {e}")


if __name__ == "__main__":
    test_sf6o2_etching()
    test_other_models()
    print("Model tests completed!")
