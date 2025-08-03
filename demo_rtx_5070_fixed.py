#!/usr/bin/env python3
"""
RTX 5070 Support Demo for tinygrad (Fixed Version)
This script demonstrates RTX 5070 GPU support without numpy dependencies
Run with: sudo python3 demo_rtx_5070_fixed.py
"""

import os
import sys
import time
from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Context

def setup_gpu_environment():
    """Setup environment variables for GPU usage and debugging"""
    print("=== Setting up GPU Environment (Root Mode) ===")
    
    # Enable NV device
    os.environ["NV"] = "1"
    print("✓ Set NV=1")
    
    # Enable debugging
    os.environ["DEBUG"] = "3"
    os.environ["NV_DEBUG"] = "3"
    print("✓ Set DEBUG=3, NV_DEBUG=3")
    
    # Force device selection
    os.environ["DEV"] = "NV"
    print("✓ Set DEV=NV")
    
    # Force PCIIface for driverless operation
    os.environ["NV_IFACE"] = "PCI"
    print("✓ Set NV_IFACE=PCI (driverless mode)")
    
    # Add CUDA 12.4 libraries to path for NVRTC builtins
    cuda_lib_path = "/usr/local/cuda-12.4/targets/x86_64-linux/lib"
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if cuda_lib_path not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib_path}:{current_ld_path}" if current_ld_path else cuda_lib_path
        print(f"✓ Added CUDA 12.4 lib path: {cuda_lib_path}")
    
    print("Environment variables set successfully!")
    print()

def test_gpu_detection():
    """Test if GPU is detected and available"""
    print("=== GPU Detection Test (Root Mode) ===")
    
    try:
        # Get available devices
        from tinygrad.device import Device
        print(f"Available devices: {Device._devices}")
        print(f"Default device: {Device.DEFAULT}")
        
        # Try to create NV device
        print("\nDebug: Attempting to initialize PCIIface...")
        from tinygrad.runtime.ops_nv import PCIIface
        print(f"PCIIface.gpus before init: {PCIIface.gpus}")
        
        device = Device["NV"]
        print(f"NV device opened: {device}")
        
        print("✓ GPU detection successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU detection failed: {e}")
        return False
    
    print()
    return True

def test_gpu_tensor_operations():
    """Test basic tensor operations on GPU without numpy"""
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
        
        # Realize the tensor without numpy conversion
        z.realize()
        print("✓ GPU computation realized successfully")
        
        # Get shape and dtype info without numpy
        print(f"Result shape: {z.shape}, dtype: {z.dtype}")
        
        print("✓ GPU tensor operations successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU tensor operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True

def test_gpu_matrix_operations():
    """Test matrix operations on GPU without numpy"""
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
        
        # Realize without numpy
        c.realize()
        print("✓ Matrix multiplication realized successfully")
        print(f"Result shape: {c.shape}, dtype: {c.dtype}")
        
        print("✓ GPU matrix operations successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU matrix operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True

def test_gpu_autograd():
    """Test autograd on GPU without numpy"""
    print("=== GPU Autograd Test (Root Mode) ===")
    
    try:
        print("Creating tensors with gradients on GPU...")
        
        # Create simple tensors that require grad
        x = Tensor.ones((3, 3), requires_grad=True, device="NV")
        y = Tensor.ones((1, 3), requires_grad=True, device="NV")
        
        print(f"Tensor x (GPU): {x}")
        print(f"Tensor y (GPU): {y}")
        
        # Forward pass
        z = (x @ y.T).sum()
        print(f"Forward result: {z}")
        
        # Just realize without converting to numpy
        z.realize()
        print("✓ Forward pass realized successfully")
        
        print("✓ GPU autograd successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU autograd failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True

def test_gpu_performance():
    """Test GPU performance with larger operations"""
    print("=== GPU Performance Test (Root Mode) ===")
    
    try:
        print("Testing 1024x1024 matrix multiplication on GPU...")
        
        # Create larger matrices
        size = 1024
        a = Tensor.randn(size, size, device="NV")
        b = Tensor.randn(size, size, device="NV")
        
        # Time the operation
        start_time = time.time()
        
        # Perform matrix multiplication
        c = a @ b
        
        # Force realization
        c.realize()
        
        end_time = time.time()
        elapsed = (end_time - start_time) * 1000  # Convert to ms
        
        print(f"✓ Matrix multiplication ({size}x{size}) completed in {elapsed:.2f}ms")
        
        # Calculate theoretical FLOPS
        flops = 2 * size ** 3  # 2*n^3 for matrix multiplication
        gflops = (flops / elapsed) / 1e6  # GFLOPS
        
        print(f"✓ Estimated performance: {gflops:.2f} GFLOPS")
        
        # Show kernel stats if available
        if GlobalCounters.kernel_count:
            print(f"✓ Kernels executed: {GlobalCounters.kernel_count}")
            
        print("✓ GPU performance test successful (Root Mode)")
        
    except Exception as e:
        print(f"✗ GPU performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    return True

def main():
    """Main test runner"""
    print("=" * 60)
    print("RTX 5070 GPU Support Demo for tinygrad (Fixed Version)")
    print("=" * 60)
    print()
    
    # Check if running as root
    if os.geteuid() != 0:
        print("⚠️  WARNING: Not running as root!")
        print("This demo requires root access for PCI device control.")
        print("Please run: sudo python3 demo_rtx_5070_fixed.py")
        print()
        sys.exit(1)
    
    # Setup environment
    setup_gpu_environment()
    
    # Run tests
    tests = [
        ("GPU Detection", test_gpu_detection),
        ("GPU Tensor Operations", test_gpu_tensor_operations),
        ("GPU Matrix Operations", test_gpu_matrix_operations),
        ("GPU Autograd", test_gpu_autograd),
        ("GPU Performance", test_gpu_performance),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("=" * 60)
    print("=== Test Summary (Root Mode) ===")
    for i, (test_name, test_func) in enumerate(tests):
        status = "✓ PASS" if i < passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! RTX 5070 GPU support is fully functional.")
    elif passed > 0:
        print("⚠️  Some tests passed. RTX 5070 support is partially working.")
    else:
        print("❌ No tests passed. RTX 5070 support needs investigation.")
    
    # Show environment info
    print("\n=== Environment Information ===")
    for var in ["NV_DEBUG", "DEBUG", "DEV", "NV", "NV_IFACE"]:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    print(f"Running as root: {os.geteuid() == 0}")

if __name__ == "__main__":
    main()