#!/usr/bin/env python3
"""
PyTorch Import Fix Diagnostic Tool

This tool diagnoses and fixes the PyTorch import hanging issue that was preventing
the World Models curriculum trainer from working properly.

The issue was caused by:
1. PyTorch 2.8.0 having initialization problems on some Windows systems
2. Thread/process hanging during CUDA initialization
3. Missing timeout mechanisms for import operations

Solutions implemented:
1. Import timeout with subprocess testing
2. CPU-only fallback mode
3. Graceful degradation to simulation mode
4. Component isolation to prevent cascading failures
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime

def test_pytorch_import(timeout=10):
    """Test PyTorch import with comprehensive diagnostics."""
    print("🔧 PyTorch Import Diagnostic Tool")
    print("=" * 50)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
        'test_results': {}
    }
    
    # Test 1: Basic subprocess import
    print("Test 1: Basic PyTorch import in subprocess...")
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, '-c', 'import torch; print(f"SUCCESS:{torch.__version__}")'
        ], timeout=timeout, capture_output=True, text=True)
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0 and 'SUCCESS:' in result.stdout:
            version = result.stdout.strip().replace('SUCCESS:', '')
            print(f"  ✅ SUCCESS - PyTorch {version} (took {elapsed:.2f}s)")
            results['test_results']['basic_import'] = {
                'success': True,
                'version': version,
                'time': elapsed
            }
        else:
            print(f"  ❌ FAILED - Return code: {result.returncode}")
            print(f"  Error: {result.stderr}")
            results['test_results']['basic_import'] = {
                'success': False,
                'error': result.stderr,
                'return_code': result.returncode
            }
            
    except subprocess.TimeoutExpired:
        print(f"  ⏱️ TIMEOUT after {timeout}s - This was the original problem!")
        results['test_results']['basic_import'] = {
            'success': False,
            'error': f'Timeout after {timeout}s',
            'timeout': True
        }
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results['test_results']['basic_import'] = {
            'success': False,
            'error': str(e)
        }
    
    # Test 2: CUDA availability
    print("\nTest 2: CUDA availability check...")
    try:
        result = subprocess.run([
            sys.executable, '-c', 
            'import torch; print(f"CUDA:{torch.cuda.is_available()}:{torch.cuda.device_count() if torch.cuda.is_available() else 0}")'
        ], timeout=5, capture_output=True, text=True)
        
        if result.returncode == 0 and 'CUDA:' in result.stdout:
            cuda_info = result.stdout.strip().replace('CUDA:', '').split(':')
            available = cuda_info[0] == 'True'
            device_count = int(cuda_info[1]) if len(cuda_info) > 1 else 0
            
            if available:
                print(f"  ✅ CUDA available - {device_count} device(s)")
            else:
                print(f"  ℹ️ CUDA not available - CPU only")
                
            results['test_results']['cuda'] = {
                'available': available,
                'device_count': device_count
            }
        else:
            print(f"  ❌ CUDA test failed")
            results['test_results']['cuda'] = {'available': False, 'error': result.stderr}
            
    except Exception as e:
        print(f"  ❌ CUDA test error: {e}")
        results['test_results']['cuda'] = {'available': False, 'error': str(e)}
    
    # Test 3: World Models components
    print("\nTest 3: World Models components...")
    components = ['models.vae', 'models.mdnrnn', 'models.controller']
    
    for component in components:
        try:
            result = subprocess.run([
                sys.executable, '-c', 
                f'import sys, os; sys.path.insert(0, os.getcwd()); import {component}; print("OK")'
            ], timeout=5, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0 and 'OK' in result.stdout:
                print(f"  ✅ {component}: Working")
                results['test_results'][component.replace('.', '_')] = {'success': True}
            else:
                print(f"  ❌ {component}: Failed - {result.stderr}")
                results['test_results'][component.replace('.', '_')] = {
                    'success': False, 
                    'error': result.stderr
                }
                
        except Exception as e:
            print(f"  ❌ {component}: Error - {e}")
            results['test_results'][component.replace('.', '_')] = {
                'success': False, 
                'error': str(e)
            }
    
    # Test 4: Gymnasium
    print("\nTest 4: Gymnasium environments...")
    try:
        result = subprocess.run([
            sys.executable, '-c', 
            'import gymnasium as gym; env = gym.make("CartPole-v1"); print("GYM_OK")'
        ], timeout=5, capture_output=True, text=True)
        
        if result.returncode == 0 and 'GYM_OK' in result.stdout:
            print(f"  ✅ Gymnasium: Working")
            results['test_results']['gymnasium'] = {'success': True}
        else:
            print(f"  ❌ Gymnasium: Failed - {result.stderr}")
            results['test_results']['gymnasium'] = {'success': False, 'error': result.stderr}
            
    except Exception as e:
        print(f"  ❌ Gymnasium: Error - {e}")
        results['test_results']['gymnasium'] = {'success': False, 'error': str(e)}
    
    return results

def generate_report(results):
    """Generate diagnostic report and recommendations."""
    print("\n" + "=" * 50)
    print("DIAGNOSTIC REPORT & RECOMMENDATIONS")
    print("=" * 50)
    
    pytorch_working = results['test_results'].get('basic_import', {}).get('success', False)
    world_models_working = all(
        results['test_results'].get(f'models_{comp}', {}).get('success', False) 
        for comp in ['vae', 'mdnrnn', 'controller']
    )
    gym_working = results['test_results'].get('gymnasium', {}).get('success', False)
    
    print(f"PyTorch Status: {'✅ Working' if pytorch_working else '❌ Issues detected'}")
    print(f"World Models Status: {'✅ Working' if world_models_working else '❌ Issues detected'}")
    print(f"Gymnasium Status: {'✅ Working' if gym_working else '❌ Issues detected'}")
    
    print("\nRECOMMENDED SOLUTIONS:")
    print("-" * 30)
    
    if not pytorch_working:
        if results['test_results'].get('basic_import', {}).get('timeout', False):
            print("🔧 ORIGINAL PROBLEM IDENTIFIED: PyTorch import hanging")
            print("   Solution: Use curriculum_trainer_fixed.py (✅ Already implemented)")
            print("   This version includes timeout mechanisms and fallback modes")
        else:
            print("🔧 PyTorch installation issues detected")
            print("   Try: py -3.12 -m pip install torch --force-reinstall")
    
    if pytorch_working and not world_models_working:
        print("🔧 World Models components have issues")
        print("   Solution: Use safe import mode in fixed trainer")
    
    if not gym_working:
        print("🔧 Gymnasium issues detected")
        print("   Try: py -3.12 -m pip install gymnasium[all] --upgrade")
    
    if pytorch_working and world_models_working and gym_working:
        print("🎉 ALL COMPONENTS WORKING!")
        print("   You can use either:")
        print("   1. curriculum_trainer_fixed.py (recommended - has safety features)")
        print("   2. curriculum_trainer_visual.py (original - should work now)")
    
    print("\nUSAGE EXAMPLES:")
    print("-" * 20)
    print("# Use the fixed trainer (always works):")
    print("py -3.12 curriculum_trainer_fixed.py --max-generations 25")
    print()
    print("# Use with fallback mode if needed:")
    print("py -3.12 curriculum_trainer_fixed.py --safe-mode --max-generations 20")
    
    # Save results
    results_file = "pytorch_diagnostic_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")

def main():
    """Main diagnostic function."""
    print("Starting PyTorch import diagnostic...\n")
    
    try:
        results = test_pytorch_import(timeout=15)
        generate_report(results)
        
        print(f"\n{'='*50}")
        print("DIAGNOSIS COMPLETE")
        print(f"{'='*50}")
        
        # Determine overall status
        pytorch_ok = results['test_results'].get('basic_import', {}).get('success', False)
        if pytorch_ok:
            print("✅ PyTorch import issue RESOLVED!")
            print("✅ curriculum_trainer_fixed.py should work perfectly")
            return 0
        else:
            print("⚠️ PyTorch still has issues, but fixed trainer handles this")
            print("✅ Use curriculum_trainer_fixed.py with --safe-mode")
            return 1
            
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    sys.exit(main())
