#!/usr/bin/env python3
"""
World Models Repository Vetting and Smoke Testing Script

This script exhaustively verifies that the World Models repository is importable and
runnable under Python 3.12, performing fast smoke tests for each module to identify
and help fix any remaining runtime errors.

Usage on Windows: py -3.12 copilot_vet_and_fix.py --device cpu
Usage on Unix: python3.12 copilot_vet_and_fix.py --device cpu
"""

import os
import sys
import json
import logging
import argparse
import subprocess
import traceback
import platform
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def setup_logging():
    """Set up logging with both console and file output."""
    # Create runs directory
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(runs_dir / "vet.log")
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class WorldModelsVetter:
    """Simple but comprehensive World Models vetting suite."""
    
    def __init__(self, device='cpu', install_deps=False):
        self.device = device
        self.install_deps = install_deps
        self.results = {
            'success': False,
            'python_version': sys.version,
            'tests': {},
            'suggestions': [],
            'artifacts': {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.info("[VET] World Models Vetting Suite Starting")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Platform: {platform.system()} {platform.release()}")
        logger.info(f"Device: {device}")
    
    def check_python_version(self) -> bool:
        """Check if Python version is 3.12+."""
        logger.info("\n" + "="*50)
        logger.info("[FAIL] PYTHON VERSION CHECK")
        logger.info("="*50)
        
        version_info = sys.version_info
        logger.info(f"Current Python: {version_info.major}.{version_info.minor}.{version_info.micro}")
        
        if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 12):
            error_msg = (
                f"[FAIL] Python {version_info.major}.{version_info.minor} is too old. "
                f"World Models requires Python 3.12+\n"
                f"On Windows: py -3.12 copilot_vet_and_fix.py --device cpu\n"
                f"On Unix: python3.12 copilot_vet_and_fix.py --device cpu"
            )
            logger.error(error_msg)
            self.results['suggestions'].append("Upgrade to Python 3.12+")
            return False
        
        logger.info("[OK] Python version OK")
        return True
    
    def install_dependencies(self) -> bool:
        """Install dependencies if requested."""
        if not self.install_deps:
            logger.info("[FAIL]  Dependency installation skipped (use --install-deps to enable)")
            return True
            
        logger.info("\n" + "="*50)
        logger.info("[FAIL] INSTALLING DEPENDENCIES")
        logger.info("="*50)
        
        if not Path("requirements.txt").exists():
            logger.warning("[FAIL]  requirements.txt not found")
            return True
        
        try:
            if platform.system() == "Windows":
                cmd = ["py", "-3.12", "-m", "pip", "install", "-r", "requirements.txt"]
            else:
                cmd = ["python3.12", "-m", "pip", "install", "-r", "requirements.txt"]
                
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("[OK] Dependencies installed")
                return True
            else:
                logger.error(f"[FAIL] Install failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[FAIL] Install error: {e}")
            return False
    
    def test_pytorch(self) -> bool:
        """Test PyTorch availability."""
        logger.info("\n" + "="*50)
        logger.info("[FAIL] PYTORCH TEST")
        logger.info("="*50)
        
        try:
            import torch
            logger.info(f"[OK] PyTorch {torch.__version__}")
            
            # Basic tensor test
            x = torch.randn(2, 3)
            y = torch.sum(x)
            logger.info(f"[OK] Basic operations work")
            
            # Device test
            if self.device == 'cuda' and torch.cuda.is_available():
                logger.info(f"[OK] CUDA available: {torch.cuda.get_device_name()}")
            else:
                logger.info(f"[FAIL]  Using CPU device")
                
            return True
            
        except ImportError as e:
            logger.error(f"[FAIL] PyTorch not found: {e}")
            self.results['suggestions'].append("Install PyTorch: pip install torch torchvision")
            return False
        except Exception as e:
            logger.error(f"[FAIL] PyTorch test failed: {e}")
            return False
    
    def test_world_models_imports(self) -> Dict[str, bool]:
        """Test importing World Models components."""
        logger.info("\n" + "="*50)
        logger.info("[FAIL] WORLD MODELS IMPORTS")
        logger.info("="*50)
        
        # Add current directory to path for imports
        sys.path.insert(0, os.getcwd())
        
        import_tests = {}
        
        # Test VAE
        try:
            from models.vae import ConvVAE
            logger.info("[OK] VAE import success")
            import_tests['vae'] = True
        except Exception as e:
            logger.error(f"[FAIL] VAE import failed: {e}")
            import_tests['vae'] = False
            self.results['suggestions'].append("Check models/vae.py exists with ConvVAE class")
        
        # Test MDN-RNN
        try:
            from models.mdnrnn import MDNRNN
            logger.info("[OK] MDN-RNN import success")
            import_tests['mdnrnn'] = True
        except Exception as e:
            logger.error(f"[FAIL] MDN-RNN import failed: {e}")
            import_tests['mdnrnn'] = False
            self.results['suggestions'].append("Check models/mdnrnn.py exists with MDNRNN class")
        
        # Test Controller
        try:
            from models.controller import Controller
            logger.info("[OK] Controller import success")
            import_tests['controller'] = True
        except Exception as e:
            logger.error(f"[FAIL] Controller import failed: {e}")
            import_tests['controller'] = False
            self.results['suggestions'].append("Check models/controller.py exists with Controller class")
        
        # Test Dataset Utils
        try:
            from tools.dataset_utils import FramesToLatentConverter
            logger.info("[OK] Dataset utils import success")
            import_tests['dataset_utils'] = True
        except Exception as e:
            logger.error(f"[FAIL] Dataset utils import failed: {e}")
            import_tests['dataset_utils'] = False
            self.results['suggestions'].append("Check tools/dataset_utils.py exists")
        
        return import_tests
    
    def test_model_instantiation(self, imports: Dict[str, bool]) -> Dict[str, bool]:
        """Test basic model instantiation."""
        logger.info("\n" + "="*50)
        logger.info("[FAIL]  MODEL INSTANTIATION")
        logger.info("="*50)
        
        instantiation_tests = {}
        
        # Test VAE instantiation
        if imports.get('vae', False):
            try:
                from models.vae import ConvVAE
                vae = ConvVAE(latent_size=32)
                logger.info("[OK] VAE instantiated successfully")
                instantiation_tests['vae'] = True
            except Exception as e:
                logger.error(f"[FAIL] VAE instantiation failed: {e}")
                instantiation_tests['vae'] = False
        else:
            instantiation_tests['vae'] = False
        
        # Test MDN-RNN instantiation
        if imports.get('mdnrnn', False):
            try:
                from models.mdnrnn import MDNRNN
                mdnrnn = MDNRNN(z_dim=32, action_dim=4, rnn_size=64, num_mixtures=3)
                logger.info("[OK] MDN-RNN instantiated successfully")
                instantiation_tests['mdnrnn'] = True
            except Exception as e:
                logger.error(f"[FAIL] MDN-RNN instantiation failed: {e}")
                instantiation_tests['mdnrnn'] = False
        else:
            instantiation_tests['mdnrnn'] = False
        
        # Test Controller instantiation
        if imports.get('controller', False):
            try:
                from models.controller import Controller
                controller = Controller(input_size=64, action_size=4)
                logger.info("[OK] Controller instantiated successfully")
                instantiation_tests['controller'] = True
            except Exception as e:
                logger.error(f"[FAIL] Controller instantiation failed: {e}")
                instantiation_tests['controller'] = False
        else:
            instantiation_tests['controller'] = False
        
        return instantiation_tests
    
    def test_forward_passes(self, instantiation: Dict[str, bool]) -> Dict[str, bool]:
        """Test basic forward passes."""
        logger.info("\n" + "="*50)
        logger.info("[FAIL] FORWARD PASS TESTS")
        logger.info("="*50)
        
        forward_tests = {}
        
        try:
            import torch
            torch.manual_seed(42)  # Deterministic results
            
            # Test VAE forward pass
            if instantiation.get('vae', False):
                try:
                    from models.vae import ConvVAE
                    vae = ConvVAE(latent_size=32)
                    vae.eval()
                    
                    # Test input (batch=1, channels=3, height=64, width=64)
                    test_input = torch.randn(1, 3, 64, 64)
                    
                    with torch.no_grad():
                        output = vae(test_input)
                        if isinstance(output, tuple):
                            recon, mu, logvar = output[:3]
                            logger.info(f"[OK] VAE forward: recon {recon.shape}, mu {mu.shape}, logvar {logvar.shape}")
                        else:
                            logger.info(f"[OK] VAE forward: output {output.shape}")
                    
                    forward_tests['vae'] = True
                except Exception as e:
                    logger.error(f"[FAIL] VAE forward failed: {e}")
                    forward_tests['vae'] = False
            else:
                forward_tests['vae'] = False
            
            # Test MDN-RNN forward pass
            if instantiation.get('mdnrnn', False):
                try:
                    from models.mdnrnn import MDNRNN
                    mdnrnn = MDNRNN(z_dim=32, action_dim=4, rnn_size=64, num_mixtures=3)
                    mdnrnn.eval()
                    
                    # Test inputs - MDN-RNN expects sequences
                    z_t = torch.randn(1, 1, 32)  # (batch, seq_len, z_dim)
                    a_t = torch.randn(1, 1, 4)   # (batch, seq_len, action_dim)
                    
                    with torch.no_grad():
                        output = mdnrnn(z_t, a_t)
                        if isinstance(output, tuple):
                            logger.info(f"[OK] MDN-RNN forward: {len(output)} outputs")
                            for i, o in enumerate(output):
                                if hasattr(o, 'shape'):
                                    logger.info(f"   Output {i}: {o.shape}")
                        else:
                            logger.info(f"[OK] MDN-RNN forward: output {output.shape}")
                    
                    forward_tests['mdnrnn'] = True
                except Exception as e:
                    logger.error(f"[FAIL] MDN-RNN forward failed: {e}")
                    forward_tests['mdnrnn'] = False
            else:
                forward_tests['mdnrnn'] = False
            
            # Test Controller forward pass
            if instantiation.get('controller', False):
                try:
                    from models.controller import Controller
                    controller = Controller(input_size=64, action_size=4)
                    controller.eval()
                    
                    # Test input (z_t + h_t concatenated)
                    test_input = torch.randn(1, 64)
                    
                    with torch.no_grad():
                        if hasattr(controller, 'get_action'):
                            action = controller.get_action(test_input[:, :32], test_input[:, 32:], deterministic=True)
                            if isinstance(action, tuple):
                                action_tensor = action[0]
                                logger.info(f"[OK] Controller get_action: action {action_tensor.shape}, log_prob {action[1].shape}")
                            else:
                                logger.info(f"[OK] Controller get_action: {action.shape}")
                        else:
                            action = controller(test_input)
                            logger.info(f"[OK] Controller forward: {action.shape}")
                    
                    forward_tests['controller'] = True
                except Exception as e:
                    logger.error(f"[FAIL] Controller forward failed: {e}")
                    forward_tests['controller'] = False
            else:
                forward_tests['controller'] = False
                
        except ImportError:
            logger.error("[FAIL] PyTorch not available for forward tests")
            forward_tests = {k: False for k in ['vae', 'mdnrnn', 'controller']}
        
        return forward_tests
    
    def test_pipeline_script(self) -> bool:
        """Test if pipeline script exists and runs."""
        logger.info("\n" + "="*50)
        logger.info("[FAIL] PIPELINE SCRIPT TEST")
        logger.info("="*50)
        
        pipeline_path = Path("copilot_run_all.py")
        if not pipeline_path.exists():
            logger.warning("[FAIL]  copilot_run_all.py not found")
            return False
        
        try:
            # Test help command
            if platform.system() == "Windows":
                cmd = ["py", "-3.12", "copilot_run_all.py", "--help"]
            else:
                cmd = ["python3.12", "copilot_run_all.py", "--help"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("[OK] Pipeline script help works")
                return True
            else:
                logger.error(f"[FAIL] Pipeline script failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"[FAIL] Pipeline script test error: {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all vetting tests."""
        logger.info("[FAIL] Starting comprehensive World Models vetting...")
        
        overall_success = True
        
        # 1. Python version check
        if not self.check_python_version():
            return False
        
        # 2. Install dependencies
        if not self.install_dependencies():
            overall_success = False
        
        # 3. PyTorch test
        if not self.test_pytorch():
            overall_success = False
        
        # 4. Import tests
        imports = self.test_world_models_imports()
        self.results['tests']['imports'] = imports
        
        # 5. Instantiation tests
        instantiation = self.test_model_instantiation(imports)
        self.results['tests']['instantiation'] = instantiation
        
        # 6. Forward pass tests
        forward_passes = self.test_forward_passes(instantiation)
        self.results['tests']['forward_passes'] = forward_passes
        
        # 7. Pipeline script test
        pipeline_ok = self.test_pipeline_script()
        self.results['tests']['pipeline'] = pipeline_ok
        
        # Determine overall success
        critical_tests = ['vae', 'mdnrnn', 'controller']
        for test in critical_tests:
            if not (imports.get(test, False) and 
                   instantiation.get(test, False) and 
                   forward_passes.get(test, False)):
                overall_success = False
        
        self.results['success'] = overall_success
        return overall_success
    
    def generate_report(self):
        """Generate final report."""
        logger.info("\n" + "="*50)
        logger.info("[FAIL] FINAL REPORT")
        logger.info("="*50)
        
        # Console summary
        if self.results['success']:
            logger.info("[OK] VET SUCCESSFUL! All critical tests passed.")
        else:
            logger.info("[FAIL] VET FAILED! Some critical tests failed.")
        
        if self.results['suggestions']:
            logger.info(f"[FAIL] {len(self.results['suggestions'])} suggestions:")
            for i, suggestion in enumerate(self.results['suggestions'], 1):
                logger.info(f"   {i}. {suggestion}")
        
        # Write JSON report
        report_path = Path("runs/vet_results.json")
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"[FAIL] Full results: {report_path}")
        
        # Print JSON to console for script output
        print(json.dumps(self.results, indent=2))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="World Models Repository Vetting Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Windows: py -3.12 copilot_vet_and_fix.py --device cpu
  Unix:    python3.12 copilot_vet_and_fix.py --device cpu
  
  Install dependencies first:
  py -3.12 copilot_vet_and_fix.py --install-deps --device cpu
        """
    )
    
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device for testing (default: cpu)')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install dependencies from requirements.txt')
    
    return parser.parse_args()

def main():
    """Main function."""
    try:
        args = parse_args()
        
        vetter = WorldModelsVetter(
            device=args.device,
            install_deps=args.install_deps
        )
        
        success = vetter.run_all_tests()
        vetter.generate_report()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Vetting interrupted by user")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Vetting failed: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()
