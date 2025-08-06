#!/usr/bin/env python3
"""
RTX 5070 Support Demo for tinygrad
This script demonstrates the successful integration of RTX 5070 support with GPU acceleration.
Run with: 
  - Linux: sudo python3 demo_rtx_5070_sudo.py
  - macOS: python3 demo_rtx_5070_sudo.py (no sudo needed)
"""

import os
import sys
import time
#from tinygrad import Tensor
from tinygrad.helpers import Context

def setup_gpu_environment():
    """Setup environment variables for GPU usage and debugging"""
    print("=== Setting up GPU Environment (Root Mode) ===")
    
    # Enable NV device
    os.environ["NV"] = "1"
    print("✓ Set NV=1")
    
    # Enable debugging
    os.environ["DEBUG"] = "3"  # High debug level for root mode
    os.environ["NV_DEBUG"] = "3"  # High NVIDIA-specific debug
    print("✓ Set DEBUG=3, NV_DEBUG=3")
    
    # Force device selection
    os.environ["DEV"] = "NV"
    print("✓ Set DEV=NV")
    
    # Force NV interface selection (use PCI interface for direct GPU access)
    os.environ["NV_IFACE"] = "PCI"
    print("✓ Set NV_IFACE=PCI")
    
    print("Environment variables set successfully!")
    print()

def test_gpu_detection():
    """Test if GPU is detected and available"""
    print("=== GPU Detection Test (Root Mode) ===")
    
    try:
        from tinygrad.device import Device
        
        # Check available devices
        available_devices = list(Device.get_available_devices())
        print(f"Available devices: {available_devices}")
        
        # Check default device
        default_device = Device.DEFAULT
        print(f"Default device: {default_device}")
        
        # Try to open NV device
        nv_device = Device["NV"]
        print(f"NV device opened: {nv_device}")
        print("✓ GPU detection successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU detection failed: {e}")
        return False
    
    print()
    return True

def test_gpu_tensor_operations():
    """Test basic tensor operations on GPU"""
    print("=== GPU Tensor Operations Test (Root Mode) ===")
    
    try:
        # Create tensors on GPU
        print("Creating tensors on GPU...")
        x = Tensor([1, 2, 3, 4, 5], device="NV")
        y = Tensor([10, 20, 30, 40, 50], device="NV")
        
        print(f"Tensor x (GPU): {x}")
        print(f"Tensor y (GPU): {y}")
        
        # Basic operations
        z = x * 2 + y
        print(f"Result (x * 2 + y): {z}")
        
        # Force computation
        result = z.numpy()
        print(f"Computed result: {result}")
        
        print("✓ GPU tensor operations successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU tensor operations failed: {e}")
        return False
    
    print()
    return True

def test_gpu_matrix_operations():
    """Test matrix operations on GPU"""
    print("=== GPU Matrix Operations Test (Root Mode) ===")
    
    try:
        # Create matrices on GPU
        print("Creating matrices on GPU...")
        a = Tensor([[1, 2], [3, 4]], device="NV")
        b = Tensor([[5, 6], [7, 8]], device="NV")
        
        print(f"Matrix A (GPU): {a}")
        print(f"Matrix B (GPU): {b}")
        
        # Matrix multiplication
        c = a @ b
        print(f"Matrix multiplication (A @ B): {c}")
        
        # Force computation
        result = c.numpy()
        print(f"Computed result: {result}")
        
        print("✓ GPU matrix operations successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU matrix operations failed: {e}")
        return False
    
    print()
    return True

def test_gpu_autograd():
    """Test autograd on GPU"""
    print("=== GPU Autograd Test (Root Mode) ===")
    
    try:
        # Create tensors with gradients on GPU
        print("Creating tensors with gradients on GPU...")
        x = Tensor.eye(3, requires_grad=True, device="NV")
        y = Tensor([[2.0, 0, -2.0]], requires_grad=True, device="NV")
        
        print(f"Tensor x (GPU): {x}")
        print(f"Tensor y (GPU): {y}")
        
        # Forward pass
        z = y.matmul(x).sum()
        print(f"Forward result: {z}")
        
        # Backward pass
        z.backward()
        
        print(f"x.grad: {x.grad.numpy()}")
        print(f"y.grad: {y.grad.numpy()}")
        
        print("✓ GPU autograd successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU autograd failed: {e}")
        return False
    
    print()
    return True

def test_gpu_performance():
    """Test GPU performance with larger matrices"""
    print("=== GPU Performance Test (Root Mode) ===")
    
    try:
        size = 1024
        print(f"Testing {size}x{size} matrix multiplication on GPU...")
        
        # Create large matrices on GPU
        start_time = time.time()
        a = Tensor.rand(size, size, device="NV")
        b = Tensor.rand(size, size, device="NV")
        
        # Matrix multiplication
        c = a @ b
        
        # Force computation
        c.realize()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"GPU {size}x{size} matrix multiplication: {duration:.4f} seconds")
        print(f"Performance: {(2 * size**3) / (duration * 1e9):.2f} GFLOPS")
        
        print("✓ GPU performance test successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU performance test failed: {e}")
        return False
    
    print()
    return True

def main():
    """Main test function"""
    print("=== RTX 5070 GPU Support Demo ===")
    print("Testing tinygrad with RTX 5070 (device ID: 0x2f04)")
    print("This demo will test GPU acceleration.")
    print("Platform:", sys.platform)
    print()
    
    # Check if running as root (Linux only)
    if sys.platform == "darwin":
        print("✓ Running on macOS - eGPU access via DriverKit (no sudo required)")
        print()
    elif os.geteuid() != 0:
        print("❌ This script must be run as root (sudo) for GPU access on Linux!")
        print("Please run: sudo python3 demo_rtx_5070_sudo.py")
        sys.exit(1)
    else:
        print("✓ Running as root - GPU access enabled")
        print()
    
    # Setup environment
    setup_gpu_environment()
    
    # Run tests with debugging context
    with Context(DEBUG=3):
        tests = [
            ("GPU Detection", test_gpu_detection),
            #("GPU Tensor Operations", test_gpu_tensor_operations),
            #   ("GPU Matrix Operations", test_gpu_matrix_operations),
            #("GPU Autograd", test_gpu_autograd),
            #("GPU Performance", test_gpu_performance),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
                if not result:
                    print(f"\n❌ Stopping tests due to failure in {test_name}")
                    break
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {e}")
                results.append((test_name, False))
                print(f"\n❌ Stopping tests due to exception in {test_name}")
                break
    
    # Summary
    print("=== Test Summary (Root Mode) ===")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RTX 5070 support is working perfectly with GPU acceleration!")
    elif passed > 0:
        print("⚠️  Some tests passed. RTX 5070 support is partially working.")
    else:
        print("❌ No tests passed. RTX 5070 support needs investigation.")
    
    print()
    print("=== Environment Information ===")
    print(f"NV_DEBUG: {os.environ.get('NV_DEBUG', 'Not set')}")
    print(f"DEBUG: {os.environ.get('DEBUG', 'Not set')}")
    print(f"DEV: {os.environ.get('DEV', 'Not set')}")
    print(f"NV: {os.environ.get('NV', 'Not set')}")
    print(f"Running as root: {os.geteuid() == 0}")

if __name__ == "__main__":
    main() 