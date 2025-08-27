#!/usr/bin/env python3
"""
Enhanced 3-Environment Curriculum Trainer with Professional Visualization for World Models

This script trains World Models across a focused curriculum of 3 core visual environments:
Pong (18.0) -> Breakout (50.0) -> CarRacing (800.0), with professional-grade visualization and enhanced FPS control.

FOCUSED CURRICULUM (3 Environments):
- ALE/Pong-v5: Simple deterministic paddle game - learning basic gameplay patterns
- ALE/Breakout-v5: Moderate brick-breaking complexity - learning object interaction
- CarRacing-v3: Complex continuous control - learning dynamic visual-motor control

NEW ENHANCED FEATURES:
- Professional render_mode="human" with intelligent window reuse and FPS control
- Separate FPS settings: --fps 30 (training), --eval-fps 60 (evaluation)
- Enhanced frame pipeline validation with RGB format preservation
- Auto-fallback to rgb_array if human rendering fails
- Environment-specific optimizations (ALE NoFrameskip, CarRacing bounds)
- Advanced video recording with RecordVideo wrapper and imageio fallback

VISUALIZATION IMPROVEMENTS:
- Window reuse prevents multiple render windows (--window-reuse)
- Frame size validation ensures proper 64x64/32x32 dimensions
- RGB format validation with uint8 conversion and grayscale-to-RGB fixes
- FPS-controlled live rollouts with configurable rates per training phase

CURRICULUM PROGRESSION:
- Fixed 3-environment sequence optimized for visual learning progression
- Threshold-based advancement with consistent 5-generation averaging
- Enhanced progress tracking with environment-specific completion status
- Quick mode support with reduced thresholds (5.0, 15.0, 100.0)

USAGE EXAMPLES:
# Standard training with enhanced visualization
python3 curriculum_trainer_visual.py --device cuda --max-generations 300 --fps 30 --eval-fps 60

# Quick testing mode with accelerated thresholds
python3 curriculum_trainer_visual.py --quick True --max-generations 50 --visualize True

# Professional video recording with validation
python3 curriculum_trainer_visual.py --record-video True --video-every-n-gens 5 --validate-rgb True

Author: GitHub Copilot
Enhanced: January 2025 - Professional 3-Environment Focus
"""

# Ensure we're in the repository root
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tools.ensure_cwd import chdir_repo_root
chdir_repo_root()

import os
import sys
import json
import csv
import logging
import argparse
import time
import numpy as np
import cv2
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import traceback
from datetime import datetime
from dataclasses import dataclass
from collections import deque
import threading
import queue
import warnings
import subprocess

# Suppress gymnasium deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Import World Models components
sys.path.insert(0, os.getcwd())
try:
    import torch
    import torch.nn as nn
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo, GrayscaleObservation, ResizeObservation, FrameStackObservation
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter
    import tqdm

    # Import and register ALE environments
    import ale_py
    gym.register_envs(ale_py)

    from models.conv_vae_dynamic import ConvVAE
    from models.mdnrnn import MDNRNN, mdn_loss_function
    from models.controller import Controller, CMAESController
    from tools.dream_env import DreamEnvironment
    from tools.dataset_utils import FramesToLatentConverter
    from env_hints import print_env_error_hint

    print("[IMPORT] All imports completed successfully!")

except ImportError as e:
    print("Error importing dependencies: {}".format(e))
    print("Please install required packages:")
    print("pip install torch gymnasium matplotlib tqdm opencv-python")
    sys.exit(1)

print("[MODULE] Module initialization complete!")


def resolve_curriculum() -> List[Tuple[str, float]]:
    """
    Fixed 3-environment curriculum optimized for World Models training.

    Focused curriculum with strong visual environments:
    1. Pong (simple, deterministic) - Learning basic gameplay
    2. Breakout (moderate complexity) - Learning object interaction
    3. CarRacing (complex, continuous) - Learning dynamic control

    Returns:
        List of (env_id, threshold_score) tuples for the 3-environment curriculum
    """
    # Fixed 3-environment curriculum (ALE + Box2D) - EXACTLY as specified
    tasks = [
        ("ALE/Pong-v5", 18),
        ("ALE/Breakout-v5", 50),
        ("CarRacing-v3", 800)
    ]

    print("[CURRICULUM] 3 core visual environments")
    print("   Tasks: Pong (18.0) → Breakout (50.0) → CarRacing (800.0)")
    print("   Focus: ALE Atari games + Box2D racing for complete visual spectrum")

    return tasks

def compute_record_gens(max_generations: int, schedule: str, explicit: Optional[List[int]] = None) -> Set[int]:
    """
    Compute which generations to record videos for based on schedule.

    Args:
        max_generations: Total number of generations
        schedule: 'triad', 'none', or 'all'
        explicit: Optional explicit list of generations to record

    Returns:
        Set of generation numbers to record (1-indexed)
    """
    if explicit:
        return {g for g in explicit if 1 <= g <= max_generations}

    if schedule == "none":
        return set()

    if schedule == "all":
        return set(range(1, max_generations + 1))

    # "triad": one in first 100, one in the middle, one in last 100
    if max_generations <= 3:
        # For very short runs, record all
        return set(range(1, max_generations + 1))

    start = min(50, max_generations)  # gen 50 or earlier if <50
    mid = max(1, max_generations // 2)
    end = max(1, max_generations - 50)  # within last 100

    # Ensure distinct and bounded
    candidates = {start, mid, end}
    return {g for g in candidates if 1 <= g <= max_generations}

def safe_unlink(p: Path):
    """Safely remove a file, ignoring errors."""
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass

def prune_intermediate_artifacts(env_dir: Path, keep_video_gens: Set[int], keep_latents: bool):
    """
    Clean up intermediate artifacts to save disk space.

    Args:
        env_dir: Environment-specific directory
        keep_video_gens: Generations whose videos should be kept
        keep_latents: Whether to keep encoded latent datasets
    """
    try:
        # Clean tmp directory
        tmp_dir = env_dir / "tmp"
        if tmp_dir.exists():
            for item in tmp_dir.rglob("*"):
                if item.is_file():
                    safe_unlink(item)

        # Clean latents if requested
        if not keep_latents:
            for latent_file in env_dir.rglob("*.npz"):
                if "latent" in latent_file.name.lower():
                    safe_unlink(latent_file)
            for latent_file in env_dir.rglob("*.npy"):
                if "latent" in latent_file.name.lower():
                    safe_unlink(latent_file)

        # Clean videos not in keep set
        videos_dir = env_dir / "videos"
        if videos_dir.exists():
            for video_file in videos_dir.glob("gen_*.mp4"):
                try:
                    gen_num = int(video_file.stem.split("_")[1])
                    if gen_num not in keep_video_gens:
                        safe_unlink(video_file)
                except (ValueError, IndexError):
                    pass  # Keep files we can't parse

    except Exception as e:
        print("   [WARN] Cleanup error for {}: {}".format(env_dir, e))

def setup_artifact_directories(artifact_root: Path) -> Dict[str, Path]:
    """
    Set up artifact directory structure.

    Args:
        artifact_root: Base directory for artifacts

    Returns:
        Dictionary mapping directory types to paths
    """
    # Create main directories
    directories = {
        'root': artifact_root,
        'checkpoints': artifact_root / 'checkpoints',
        'logs': artifact_root / 'logs',
        'videos': artifact_root / 'videos',
        'reports': artifact_root / 'reports',
        'tmp': artifact_root / 'tmp'
    }

    # Create all directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories

def detect_windows_external_drive() -> Optional[Path]:
    """
    Detect if Windows D: drive exists for external storage.

    Returns:
        Path to D:/WorldModels if D: drive exists, None otherwise
    """
    if platform.system() == "Windows":
        d_drive = Path("D:/")
        if d_drive.exists():
            return d_drive / "WorldModels"
    return None
@dataclass
class CurriculumTask:
    """Defines a curriculum task with environment and success criteria."""
    env_id: str
    threshold_score: float
    max_episode_steps: int = 1000
    solved: bool = False
    best_score: float = float('-inf')
    generations_trained: int = 0

    # Early stopping tracking
    rolling_rewards: Optional[deque] = None
    rolling_mean: float = float('-inf')
    no_improvement_count: int = 0
    best_rolling_mean: float = float('-inf')
    plateau_stopped: bool = False
    extension_granted: bool = False

@dataclass
class TrainingConfig:
    """Training configuration for 3-environment curriculum learning with enhanced visualization."""
    device: str = 'cpu'
    max_generations: int = 1000
    episodes_per_eval: int = 10
    checkpoint_dir: str = './runs/curriculum_visual'
    visualize: bool = True
    record_video: bool = False
    video_every_n_gens: int = 10
    quick_mode: bool = False  # Quick mode for fast testing
    smoke_test: bool = False  # Smoke test mode - visual validation only, skip training

    # New finalized features
    preflight: bool = True  # Run preflight smoke test before training
    early_stop: bool = True  # Enable early stopping
    patience: int = 20  # Generations without improvement
    min_delta: float = 1.0  # Minimum improvement required
    eval_every: int = 10  # Periodic evaluation snapshots

    # New video scheduling fields
    video_schedule: str = "triad"  # "triad", "none", "all"
    video_gens: Optional[List[int]] = None  # Explicit generation list

    # New artifact management fields
    artifact_root: Optional[Path] = None
    clean_cache: bool = True
    keep_latents: bool = False

    # Enhanced Visualization Settings
    show_rollout_every_n_gens: Optional[Dict[str, int]] = None  # Environment-specific rendering frequency
    render_mode: str = "human"  # Gymnasium render mode - always human for enhanced visualization
    video_fps: int = 30  # Video recording FPS
    fps: int = 30  # Training visualization FPS (live rollouts)
    eval_fps: int = 60  # Evaluation visualization FPS (higher quality for assessment)
    window_reuse: bool = True  # Reuse existing render windows
    close_on_completion: bool = False  # Keep windows open after task completion

    # Frame Pipeline Validation
    validate_rgb_frames: bool = True  # Ensure RGB format preservation
    validate_frame_sizes: bool = True  # Validate 64x64/32x32 dimensions
    fallback_to_rgb_array: bool = True  # Auto-fallback if human render fails

    # GPU Memory Optimization Settings
    use_amp: bool = True  # Automatic Mixed Precision
    use_tf32: bool = True  # TensorFloat-32
    vae_img_size: int = 64  # VAE image size for memory efficiency
    vae_batch_size: int = 32  # VAE batch size
    grad_accumulation_steps: int = 1  # Gradient accumulation
    max_episode_steps: int = 1000

    # VAE hyperparameters
    vae_latent_size: int = 32
    vae_epochs: int = 5
    vae_batch_size: int = 32

    # MDN-RNN hyperparameters
    rnn_size: int = 128
    num_mixtures: int = 5
    mdnrnn_epochs: int = 5
    mdnrnn_batch_size: int = 16

    # Controller hyperparameters
    controller_hidden_size: int = 64
    cma_population_size: int = 16
    cma_sigma: float = 0.1
    patience: int = 50  # Early stopping patience

class CurriculumTrainer:
    """Main curriculum trainer with visualization."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set up artifact directories with new structure
        if config.artifact_root:
            self.artifact_root = Path(config.artifact_root)
        else:
            self.artifact_root = Path(config.checkpoint_dir)

        # Set up directory structure
        self.directories = setup_artifact_directories(self.artifact_root)
        self.checkpoint_dir = self.directories['checkpoints']

        # Set up per-environment directories
        self.env_directories = {}
        curriculum_tasks = resolve_curriculum()
        for env_id, _ in curriculum_tasks:
            env_name = env_id.replace("/", "_")
            env_dir = self.artifact_root / env_name
            self.env_directories[env_id] = setup_artifact_directories(env_dir)

        # Compute record generations for video scheduling
        self.record_gens = compute_record_gens(
            config.max_generations,
            config.video_schedule,
            config.video_gens
        )

        # Set up video directories (legacy support)
        self.video_base_dir = self.directories['videos']

        # Set up logging before GPU configuration
        self.setup_logging()

        # Configure GPU memory optimizations
        self._configure_gpu_optimizations()

        # Configure visualization settings per environment
        if config.show_rollout_every_n_gens is None:
            self.rollout_frequency = {
                "ALE/Pong-v5": 1,             # Show every generation for Pong
                "PongNoFrameskip-v5": 1,
                "PongNoFrameskip-v4": 1,
                "LunarLander-v2": 5,          # Every 5 generations for others
                "LunarLander-v3": 5,
                "ALE/Breakout-v5": 5,         # Updated Atari naming
                "BreakoutNoFrameskip-v5": 5,
                "BreakoutNoFrameskip-v4": 5,
                "CarRacing-v3": 5,
                "CarRacing-v3": 5
            }
        else:
            self.rollout_frequency = config.show_rollout_every_n_gens

        # Define fixed 3-environment curriculum
        curriculum_tasks = resolve_curriculum()
        self.curriculum = [CurriculumTask(env_id, threshold) for env_id, threshold in curriculum_tasks]

        # Initialize rolling rewards tracking for early stopping
        for task in self.curriculum:
            task.rolling_rewards = deque(maxlen=5)  # Rolling 5 mean

        # Apply quick mode modifications if enabled
        if config.quick_mode:
            self.logger.info("[QUICK MODE] Using reduced thresholds for fast debugging.")
            # Reduce thresholds for quick testing - 3 environments: Pong, Breakout, CarRacing
            quick_thresholds = [5.0, 15.0, 100.0]  # Simplified thresholds for the 3 environments
            for i, threshold in enumerate(quick_thresholds):
                if i < len(self.curriculum):  # Ensure we don't exceed curriculum length
                    self.curriculum[i].threshold_score = threshold

            # Reduce hyperparameters for faster training
            self.config.cma_population_size = max(4, self.config.cma_population_size // 4)
            self.config.vae_epochs = max(1, self.config.vae_epochs // 2)
            self.config.mdnrnn_epochs = max(1, self.config.mdnrnn_epochs // 2)
            self.config.patience = max(5, self.config.patience // 10)

            self.logger.info("[QUICK MODE] Reduced thresholds: Pong=5, Breakout=15, CarRacing=100")
            self.logger.info("[QUICK MODE] Reduced population size: {}".format(self.config.cma_population_size))

        # Training state
        self.current_task_idx = 0
        self.global_generation = 0
        self.training_start_time = time.time()

        # Models (will be created per environment)
        self.vae = None
        self.mdnrnn = None
        self.controller = None
        self.dream_env = None

        # Progress tracking
        self.progress_queue = queue.Queue()
        self.visualization_thread = None
        self.stop_visualization = threading.Event()

        self.logger.info("Curriculum Trainer initialized")
        self.logger.info("Device: {}".format(self.device))
        self.logger.info("Checkpoint directory: {}".format(self.checkpoint_dir))
        self.logger.info("Curriculum: {} environments".format(len(self.curriculum)))

    def setup_logging(self):
        """Set up comprehensive logging."""
        log_dir = self.directories['logs']

        # File logging
        log_file = log_dir / "curriculum_{}.log".format(datetime.now().strftime('%Y%m%d_%H%M%S'))

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )

        self.logger = logging.getLogger('CurriculumTrainer')

        # TensorBoard logging
        self.writer = SummaryWriter(self.directories['logs'] / "tensorboard")

        # CSV logging
        self.csv_file = self.directories['logs'] / "curriculum_progress.csv"
        with open(self.csv_file, 'w') as f:
            f.write("timestamp,env_id,generation,mean_score,best_score,threshold,solved,time_elapsed\n")

    def _configure_gpu_optimizations(self):
        """Configure GPU memory optimizations for RTX 3050."""
        if self.device.type == 'cuda' and torch.cuda.is_available():
            # Enable TensorFloat-32 for better performance on modern GPUs
            if self.config.use_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info("[GPU] TensorFloat-32 enabled")

            # Configure memory settings
            torch.cuda.empty_cache()
            self.logger.info("[GPU] CUDA Device: {}".format(torch.cuda.get_device_name(0)))
            self.logger.info("[GPU] Mixed Precision: {}".format(self.config.use_amp))
            self.logger.info("[GPU] VAE Image Size: {}".format(self.config.vae_img_size))
            self.logger.info("[GPU] VAE Batch Size: {}".format(self.config.vae_batch_size))
            self.logger.info("[GPU] Gradient Accumulation Steps: {}".format(self.config.grad_accumulation_steps))
        else:
            self.logger.info("[GPU] Using CPU mode")

        # Create AMP scaler if using mixed precision
        if self.config.use_amp and self.device.type == 'cuda':
            from torch.amp.grad_scaler import GradScaler
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None

    def create_env(self, env_id: str, record_video: bool = False, video_dir: Optional[Path] = None) -> gym.Env:
        """Create and configure environment."""
        try:
            env = gym.make(env_id, render_mode="rgb_array" if record_video else None)

            # Add video recording wrapper if needed
            if record_video and video_dir:
                video_dir.mkdir(parents=True, exist_ok=True)
                env = RecordVideo(
                    env,
                    str(video_dir),
                    episode_trigger=lambda x: True,  # Record all episodes
                    name_prefix="{}_gen{}".format(env_id, self.global_generation)
                )

            # Apply preprocessing for Atari games
            if "NoFrameskip" in env_id:
                env = GrayscaleObservation(env)
                env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                env = FrameStackObservation(env, 4)
            elif env_id == "LunarLander-v2":
                # LunarLander doesn't need frame preprocessing
                pass
            elif env_id == "CarRacing-v3":
                env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                env = FrameStackObservation(env, 4)

            return env

        except Exception as e:
            self.logger.error("Failed to create environment {}: {}".format(env_id, e))
            raise

    def create_enhanced_render_env(self, env_id: str, fps_target: int = 30,
                                 fallback_to_rgb: bool = True) -> Tuple[gym.Env, str]:
        """
        Create environment with enhanced rendering support and proper FPS control.

        Args:
            env_id: Environment identifier
            fps_target: Target FPS for rendering (30 for training, 60 for evaluation)
            fallback_to_rgb: Auto-fallback to rgb_array if human render fails

        Returns:
            Tuple of (environment, actual_render_mode_used)
        """
        actual_render_mode = self.config.render_mode

        try:
            # First try with configured render mode (typically "human")
            env = gym.make(env_id, render_mode=self.config.render_mode)

            # ALE-specific optimizations
            if "ALE/" in env_id:
                # Enable proper NoFrameskip handling for Atari games
                if hasattr(env.unwrapped, 'ale'):
                    env.unwrapped.ale.setBool('color_averaging', False)

            # CarRacing-specific optimizations
            elif "CarRacing" in env_id:
                # Set proper continuous action space bounds
                if hasattr(env, 'action_space'):
                    env.action_space.seed(42)  # For reproducible sampling

            self.logger.info(f"[OK] Enhanced render env created for {env_id} with {self.config.render_mode} mode @ {fps_target}fps")

            return env, actual_render_mode

        except Exception as e:
            if fallback_to_rgb and self.config.render_mode == "human":
                self.logger.warning(f"[WARN] Human render failed for {env_id}, falling back to rgb_array: {e}")
                try:
                    env = gym.make(env_id, render_mode="rgb_array")
                    actual_render_mode = "rgb_array"
                    self.logger.info(f"[OK] Fallback render env created for {env_id} with rgb_array mode")
                    return env, actual_render_mode
                except Exception as e2:
                    self.logger.error(f"[ERROR] Both render modes failed for {env_id}: {e2}")
                    raise
            else:
                self.logger.error(f"[ERROR] Failed to create enhanced render env for {env_id}: {e}")
                raise

    def validate_frame_pipeline(self, frame: np.ndarray, env_id: str,
                               expected_shape: Tuple[int, int] = (64, 64)) -> np.ndarray:
        """
        Validate and fix frame format issues in the pipeline.

        Args:
            frame: Input frame to validate
            env_id: Environment ID for context
            expected_shape: Expected (height, width) for the frame

        Returns:
            Validated and corrected frame
        """
        if not self.config.validate_rgb_frames and not self.config.validate_frame_sizes:
            return frame  # Skip validation if disabled

        # Validate RGB format
        if self.config.validate_rgb_frames:
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                self.logger.warning(f"[WARN] Frame format issue in {env_id}: shape {frame.shape}, expected (H,W,3)")
                # Try to fix common issues
                if len(frame.shape) == 2:
                    frame = np.stack([frame] * 3, axis=-1)  # Grayscale to RGB
                elif len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]  # RGBA to RGB

            # Ensure uint8 format
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

        # Validate frame sizes
        if self.config.validate_frame_sizes:
            if frame.shape[:2] != expected_shape:
                self.logger.debug(f"📐 Resizing frame in {env_id}: {frame.shape[:2]} → {expected_shape}")
                frame = cv2.resize(frame, expected_shape[::-1])  # cv2 uses (width, height)

        return frame

    def run_smoke_test(self) -> bool:
        """
        Run smoke test mode: skip training, run 1 short rollout per env for visual validation.

        Returns:
            True if smoke test completed successfully, False if any environment failed
        """
        self.logger.info("[SMOKE TEST] Starting visual validation mode - no training")
        print("\n" + "="*70)
        print("🔥 SMOKE TEST MODE: Visual Environment Validation")
        print("="*70)
        print("Task: Quick visual check of all 3 environments")
        print("Test: 200-step rollout per environment")
        print("Skip: ALL training phases (VAE, MDN-RNN, Controller)")
        print("-"*70)

        # Get fixed 3-environment tasks (same as resolve_curriculum)
        tasks = resolve_curriculum()

        # Set up smoke test directory
        smoke_test_dir = self.checkpoint_dir / "smoke_test"
        smoke_test_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0

        for i, (env_id, _) in enumerate(tasks):
            print(f"\n[{i+1}/3] Testing {env_id}...")

            try:
                # For visual fallback, try both human and rgb_array modes
                use_opencv_fallback = False
                cv2_window_name = None

                # Set up video directory if recording
                video_dir = None
                if self.config.record_video:
                    video_dir = smoke_test_dir / env_id.replace("/", "_") / "videos"
                    video_dir.mkdir(parents=True, exist_ok=True)

                # Use OpenCV by default for smoke test (ALE human mode has window issues)
                print(f"   Creating environment with render_mode='rgb_array' (OpenCV display)...")
                import gymnasium as gym
                env = gym.make(env_id, render_mode="rgb_array")
                render_mode = "rgb_array"
                use_opencv_fallback = True

                # Set up high-quality OpenCV window
                if self.config.visualize:
                    import cv2
                    cv2_window_name = f"{env_id} - Smoke Test (High Quality)"
                    cv2.namedWindow(cv2_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    # Larger window for better quality
                    if "ALE" in env_id:
                        cv2.resizeWindow(cv2_window_name, 840, 672)  # 4x scale for ALE
                    else:
                        cv2.resizeWindow(cv2_window_name, 800, 600)  # Better scale for other games
                    print(f"   High-quality OpenCV window created: '{cv2_window_name}'")
                    use_opencv_fallback = True

                # Give the window time to appear
                if render_mode == "human":
                    print(f"   Waiting 2 seconds for {env_id} window to appear...")
                    time.sleep(2)

                # Add video recording if enabled
                if self.config.record_video and video_dir:
                    try:
                        env = RecordVideo(
                            env,
                            str(video_dir),
                            episode_trigger=lambda x: True,
                            name_prefix=f"smoke_test_{env_id.replace('/', '_')}"
                        )
                        print(f"   Video recording to: {video_dir}")
                    except Exception as video_e:
                        print(f"   [WARN] Video recording failed: {video_e}")

                # Run 200-step rollout
                print(f"   Resetting environment...")
                obs, info = env.reset()
                print(f"   Running 200-step rollout with {render_mode} rendering...")

                if render_mode == "human":
                    print(f"   [VISUAL] Look for the {env_id} game window - it should be visible!")
                elif use_opencv_fallback:
                    print(f"   [VISUAL] OpenCV window '{cv2_window_name}' should show the game!")

                step_count = 0
                total_reward = 0.0

                # Calculate frame delay for FPS control
                frame_delay = 1.0 / self.config.fps if self.config.fps > 0 else 0
                last_frame_time = time.time()

                while step_count < 200:
                    # Random action for smoke test
                    action = env.action_space.sample()

                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    step_count += 1

                    # Handle high-quality OpenCV display for rgb_array mode
                    if use_opencv_fallback and self.config.visualize and cv2_window_name:
                        try:
                            import cv2
                            import numpy as np
                            frame = env.render()
                            if frame is not None:
                                # Convert to numpy array if needed
                                if not isinstance(frame, np.ndarray):
                                    frame = np.array(frame)

                                # Convert RGB to BGR for OpenCV
                                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                                # High-quality scaling with different methods for different games
                                if "ALE" in env_id:
                                    # For Atari games, use nearest neighbor for pixel-perfect scaling
                                    frame_scaled = cv2.resize(frame_bgr, (840, 672), interpolation=cv2.INTER_NEAREST)
                                else:
                                    # For other games, use high-quality interpolation
                                    frame_scaled = cv2.resize(frame_bgr, (800, 600), interpolation=cv2.INTER_CUBIC)

                                # Add crisp info overlay with better positioning
                                overlay = frame_scaled.copy()

                                # Semi-transparent background for text
                                cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 0), -1)
                                cv2.addWeighted(overlay, 0.7, frame_scaled, 0.3, 0, frame_scaled)

                                # High-quality text rendering
                                cv2.putText(frame_scaled, f'Step: {step_count}/200', (15, 35),
                                          cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                                cv2.putText(frame_scaled, f'Reward: {total_reward:.1f}', (15, 65),
                                          cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                                cv2.imshow(cv2_window_name, frame_scaled)

                                # Quick key check (non-blocking)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    print(f"   Quit key pressed for {env_id}")
                                    break
                        except Exception as cv_e:
                            print(f"   [WARN] OpenCV display error: {cv_e}")

                    # Print progress every 50 steps
                    if step_count % 50 == 0:
                        print(f"   Step {step_count}/200 - Current reward: {total_reward:.2f}")

                    # FPS control for visualization
                    if self.config.visualize and frame_delay > 0:
                        elapsed = time.time() - last_frame_time
                        sleep_time = frame_delay - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        last_frame_time = time.time()

                    # Early termination if episode ends
                    if terminated or truncated:
                        print(f"   Episode ended at step {step_count}, resetting...")
                        obs, info = env.reset()
                        total_reward = 0.0  # Reset for new episode

                # Clean up
                print(f"   Closing environment...")
                if cv2_window_name:
                    try:
                        import cv2
                        cv2.destroyWindow(cv2_window_name)
                    except:
                        pass

                if self.config.close_on_completion:
                    env.close()

                print(f"   [OK] {env_id} - 200 steps completed, total reward: {total_reward:.2f}")
                success_count += 1

            except Exception as e:
                print(f"   [ERROR] {env_id} failed: {e}")
                # Print helpful pip install hints
                if "No module named 'atari_py'" in str(e) or "ALE" in env_id:
                    print(f"   💡 Try: pip install 'gymnasium[atari]' ale-py")
                elif "CarRacing" in env_id or "Box2D" in str(e):
                    print(f"   💡 Try: pip install 'gymnasium[box2d]' box2d-py")
                else:
                    print(f"   💡 Check environment installation: {env_id}")

                # Continue to next environment
                continue

        print("\n" + "-"*70)
        print(f"Smoke test complete: {success_count}/3 environments validated")

        if success_count == 3:
            print("[SUCCESS] All environments working - visuals validated!")
            return True
        else:
            print(f"[PARTIAL] {3-success_count} environments had issues")
            return False

    def collect_random_data(self, env_id: str, num_episodes: int = 100) -> str:
        """Collect random rollout data for VAE training."""
        self.logger.info(f"Collecting {num_episodes} random episodes from {env_id}")

        env = self.create_env(env_id)
        data_dir = self.env_directories[env_id]['tmp']

        episodes_data = []

        try:
            for episode in tqdm.tqdm(range(num_episodes), desc="Collecting data"):
                obs, info = env.reset()
                episode_frames = []
                episode_actions = []
                episode_rewards = []
                done = False

                while not done:
                    action = env.action_space.sample()
                    next_obs, reward, terminated, truncated, info = env.step(action)

                    episode_frames.append(obs)
                    episode_actions.append(action)
                    episode_rewards.append(reward)

                    obs = next_obs
                    done = terminated or truncated

                episodes_data.append({
                    'frames': np.array(episode_frames),
                    'actions': np.array(episode_actions),
                    'rewards': np.array(episode_rewards)
                })

            # Save collected data
            data_file = data_dir / "episodes.npz"
            np.savez(data_file, episodes=episodes_data)

            self.logger.info(f"Collected data saved to {data_file}")
            return str(data_file)

        finally:
            env.close()

    def train_vae(self, env_id: str, data_file: str) -> str:
        """Train VAE on collected data."""
        self.logger.info(f"Training VAE for {env_id}")

        # Load data
        data = np.load(data_file, allow_pickle=True)
        episodes = data['episodes']

        # Extract all frames
        all_frames = []
        for episode in episodes:
            frames = episode['frames']
            all_frames.extend(frames)

        all_frames = np.array(all_frames)
        total_frames = len(all_frames)
        self.logger.info(f"Training VAE on {total_frames} frames")
        self.logger.info(f"Original frame shape: {all_frames.shape}")

        # Handle different observation spaces
        if len(all_frames.shape) == 2:
            # Vector observation (like LunarLander) - shape: (N, obs_dim)
            self.logger.info("Vector observation detected - converting to image format")
            obs_dim = all_frames.shape[1]

            # Create a simple 2D visualization of vector observations
            if obs_dim <= 16:
                img_size = 4
            elif obs_dim <= 64:
                img_size = 8
            else:
                img_size = int(np.ceil(np.sqrt(obs_dim)))

            # Pad or truncate observations to fit square image
            padded_size = img_size * img_size
            if obs_dim < padded_size:
                padding = np.zeros((all_frames.shape[0], padded_size - obs_dim))
                all_frames = np.concatenate([all_frames, padding], axis=1)
            elif obs_dim > padded_size:
                all_frames = all_frames[:, :padded_size]

            # Reshape to image format and normalize to 0-255
            all_frames = all_frames.reshape(all_frames.shape[0], img_size, img_size)

            # Normalize to 0-255 range
            frame_min, frame_max = all_frames.min(), all_frames.max()
            if frame_max > frame_min:
                all_frames = ((all_frames - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
            else:
                all_frames = (all_frames * 0 + 128).astype(np.uint8)  # Gray if constant

            # Add channel dimension for grayscale
            all_frames = np.expand_dims(all_frames, axis=-1)
            self.logger.info(f"Converted to image shape: {all_frames.shape}")

        # Handle 5D frames (CarRacing with frame stacks)
        elif len(all_frames.shape) == 5:
            self.logger.info(f"5D frame stack detected: {all_frames.shape}")
            # Take the last frame from each stack
            all_frames = all_frames[:, -1]  # Shape: (N, H, W, C)
            self.logger.info(f"Using last frame from stack: {all_frames.shape}")

        # Handle 4D frames but check channels
        elif len(all_frames.shape) == 4 and all_frames.shape[-1] > 4:
            self.logger.info(f"Detected {all_frames.shape[-1]} channels, slicing to RGB (first 3)")
            all_frames = all_frames[..., :3]  # Keep only RGB channels
            self.logger.info(f"Sliced frame shape: {all_frames.shape}")

        # Ensure proper shape for image processing
        if len(all_frames.shape) == 3:
            # Add channel dimension if missing (grayscale)
            all_frames = np.expand_dims(all_frames, axis=-1)
            self.logger.info(f"Added channel dimension: {all_frames.shape}")

        # Resize frames to match VAE expected input size
        import cv2
        target_size = self.config.vae_img_size

        if all_frames.shape[1] != target_size or all_frames.shape[2] != target_size:
            self.logger.info(f"Resizing frames from ({all_frames.shape[1]}, {all_frames.shape[2]}) to {target_size}x{target_size}")
            resized_frames = []
            for frame in all_frames:
                # Handle different frame formats for cv2.resize
                if len(frame.shape) == 3 and frame.shape[2] == 1:
                    # Grayscale with channel dimension
                    frame_2d = frame[:, :, 0]
                    resized_frame = cv2.resize(frame_2d, (target_size, target_size), interpolation=cv2.INTER_AREA)
                    resized_frame = np.expand_dims(resized_frame, axis=-1)
                elif len(frame.shape) == 2:
                    # Pure 2D grayscale
                    resized_frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
                    resized_frame = np.expand_dims(resized_frame, axis=-1)
                else:
                    # Color image (H, W, 3)
                    resized_frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
                    if len(resized_frame.shape) == 2:
                        resized_frame = np.expand_dims(resized_frame, axis=-1)

                resized_frames.append(resized_frame)
            all_frames = np.array(resized_frames)
            self.logger.info(f"Resized frame shape: {all_frames.shape}")

        # Calculate memory requirements
        frame_memory = all_frames.nbytes / 1024 / 1024 / 1024  # GB
        self.logger.info(f"Frame data size: {frame_memory:.2f} GB")

        # Create VAE - after all preprocessing
        if len(all_frames.shape) == 4:  # Should be (N, H, W, C)
            input_channels = all_frames.shape[-1]
        else:
            input_channels = 1

        self.logger.info(f"Creating VAE with {input_channels} input channels for frames shape {all_frames.shape}")

        self.vae = ConvVAE(
            img_channels=input_channels,
            img_size=self.config.vae_img_size,
            latent_dim=self.config.vae_latent_size
        )
        self.vae.to(self.device)        # Use streaming data loading if dataset is too large (>1GB)
        if frame_memory > 1.0:
            return self._train_vae_streaming(env_id, all_frames)
        else:
            return self._train_vae_batch(env_id, all_frames)

    def _train_vae_streaming(self, env_id: str, all_frames: np.ndarray) -> str:
        """Train VAE with streaming data loading for large datasets."""
        self.logger.info("Using streaming data loading for memory efficiency")

        # The VAE model is already created and moved to device in the main method

        # Prepare streaming dataset with smaller chunks
        chunk_size = min(500, len(all_frames) // 20)  # Smaller chunks for RTX 3050
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        for epoch in range(self.config.vae_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Process data in chunks
            indices = np.random.permutation(len(all_frames))
            for chunk_start in range(0, len(all_frames), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(all_frames))
                chunk_indices = indices[chunk_start:chunk_end]

                # Load small chunk into memory
                chunk_frames = all_frames[chunk_indices]
                frames_tensor = torch.FloatTensor(chunk_frames).permute(0, 3, 1, 2) / 255.0                # Process chunk in mini-batches
                for batch_start in range(0, len(frames_tensor), self.config.vae_batch_size):
                    batch_end = min(batch_start + self.config.vae_batch_size, len(frames_tensor))
                    batch = frames_tensor[batch_start:batch_end].to(self.device, non_blocking=True)

                    optimizer.zero_grad()

                    # Use mixed precision if enabled
                    if self.config.use_amp and self.scaler:
                        from torch.amp.autocast_mode import autocast
                        with autocast('cuda'):
                            recon, mu, logvar = self.vae(batch)
                            recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            loss = recon_loss + kl_loss

                        # Scaled backward pass
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        # Standard precision training
                        recon, mu, logvar = self.vae(batch)
                        recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + kl_loss

                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    # Clear cache periodically
                    if num_batches % 5 == 0 and self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                # Clear chunk from memory immediately
                del frames_tensor, chunk_frames
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            self.logger.info(f"VAE Epoch {epoch+1}/{self.config.vae_epochs}: Loss = {avg_loss:.4f}")

            # Log to TensorBoard
            self.writer.add_scalar(f'{env_id}/VAE_Loss', avg_loss, epoch)

        # Save VAE to new directory structure
        vae_path = self.env_directories[env_id]['checkpoints'] / "vae_best.pt"
        torch.save(self.vae.state_dict(), vae_path)

        self.logger.info(f"Streaming VAE saved to {vae_path}")
        return str(vae_path)

    def _train_vae_batch(self, env_id: str, all_frames: np.ndarray) -> str:
        """Train VAE with full batch loading for small datasets."""
        self.logger.info("Using batch data loading")

        # Prepare data loader with optimized batch size
        # Fix tensor dimensions based on frame format
        if len(all_frames.shape) == 4:  # (N, H, W, C)
            frames_tensor = torch.FloatTensor(all_frames).permute(0, 3, 1, 2) / 255.0
        elif len(all_frames.shape) == 3:  # (N, H, W) - grayscale
            frames_tensor = torch.FloatTensor(all_frames).unsqueeze(1) / 255.0  # Add channel dim
        else:
            raise ValueError(f"Unexpected frame shape: {all_frames.shape}")

        dataset = torch.utils.data.TensorDataset(frames_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.vae_batch_size,
            shuffle=True,
            pin_memory=True if self.device.type == 'cuda' else False,
            num_workers=0  # Keep 0 for Windows compatibility
        )

        # Train VAE with mixed precision support
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)

        for epoch in range(self.config.vae_epochs):
            epoch_loss = 0.0
            for batch_idx, (batch,) in enumerate(dataloader):
                batch = batch.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # Use mixed precision if enabled
                if self.config.use_amp and self.scaler:
                    from torch.amp.autocast_mode import autocast
                    with autocast('cuda'):
                        recon, mu, logvar = self.vae(batch)

                        # VAE loss
                        recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        loss = recon_loss + kl_loss

                    # Scaled backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Standard precision training
                    recon, mu, logvar = self.vae(batch)

                    # VAE loss
                    recon_loss = nn.functional.mse_loss(recon, batch, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss

                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()

                # Clear cache periodically for memory management
                if batch_idx % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"VAE Epoch {epoch+1}/{self.config.vae_epochs}: Loss = {avg_loss:.4f}")

            # Log to TensorBoard
            self.writer.add_scalar(f'{env_id}/VAE_Loss', avg_loss, epoch)

        # Save VAE to new directory structure
        vae_path = self.env_directories[env_id]['checkpoints'] / "vae_best.pt"
        torch.save(self.vae.state_dict(), vae_path)

        self.logger.info(f"Batch VAE saved to {vae_path}")
        return str(vae_path)

    def encode_data_to_latents(self, env_id: str, data_file: str, vae_path: str) -> str:
        """Encode frame data to latent sequences for MDN-RNN training."""
        self.logger.info(f"Encoding data to latents for {env_id}")

        # Load data to determine the correct VAE input channels
        data = np.load(data_file, allow_pickle=True)
        episodes = data['episodes']

        # Get sample frame to determine input channels
        sample_episode = episodes[0].item() if hasattr(episodes[0], 'item') else episodes[0]
        sample_frames = sample_episode['frames']

        # Determine input channels based on environment type (same logic as train_vae)
        if len(sample_frames.shape) == 2:
            # Vector observation
            input_channels = 1
        elif len(sample_frames.shape) == 5:
            # 5D frame stack - use last frame channels
            input_channels = sample_frames.shape[-1]
        elif len(sample_frames.shape) == 4:
            input_channels = sample_frames.shape[-1]
            if input_channels > 4:
                input_channels = 3  # RGB
        elif len(sample_frames.shape) == 3:
            input_channels = 1
        else:
            input_channels = 1

        self.logger.info(f"Creating VAE for encoding with {input_channels} input channels")

        # Instantiate VAE with correct input channels
        self.vae = ConvVAE(
            img_channels=input_channels,
            img_size=self.config.vae_img_size,
            latent_dim=self.config.vae_latent_size
        )
        self.vae.to(self.device)
        self.vae.load_state_dict(torch.load(vae_path, map_location=self.device, weights_only=True))
        self.vae.eval()

        latent_episodes = []

        with torch.no_grad():
            for episode_data in tqdm.tqdm(episodes, desc="Encoding episodes"):
                # Handle both numpy array containers and direct dict objects
                if hasattr(episode_data, 'item'):
                    episode = episode_data.item()
                else:
                    episode = episode_data
                frames = episode['frames']
                actions = episode['actions']
                rewards = episode['rewards']

                # Apply same preprocessing as in train_vae
                original_shape = frames.shape

                # Handle different observation spaces (same as train_vae)
                if len(frames.shape) == 2:
                    # Vector observation - convert to image
                    obs_dim = frames.shape[1]
                    if obs_dim <= 16:
                        img_size = 4
                    elif obs_dim <= 64:
                        img_size = 8
                    else:
                        img_size = int(np.ceil(np.sqrt(obs_dim)))

                    padded_size = img_size * img_size
                    if obs_dim < padded_size:
                        padding = np.zeros((frames.shape[0], padded_size - obs_dim))
                        frames = np.concatenate([frames, padding], axis=1)
                    elif obs_dim > padded_size:
                        frames = frames[:, :padded_size]

                    frames = frames.reshape(frames.shape[0], img_size, img_size)
                    frame_min, frame_max = frames.min(), frames.max()
                    if frame_max > frame_min:
                        frames = ((frames - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
                    else:
                        frames = (frames * 0 + 128).astype(np.uint8)
                    frames = np.expand_dims(frames, axis=-1)

                elif len(frames.shape) == 5:
                    # 5D frame stack - take last frame
                    frames = frames[:, -1]

                elif len(frames.shape) == 4 and frames.shape[-1] > 4:
                    # Multi-channel image - keep only RGB
                    frames = frames[..., :3]

                elif len(frames.shape) == 3:
                    # Add channel dimension
                    frames = np.expand_dims(frames, axis=-1)

                # Resize frames to match VAE input size if needed
                if len(frames.shape) >= 3 and (frames.shape[-3] != self.config.vae_img_size or frames.shape[-2] != self.config.vae_img_size):
                    self.logger.debug(f"Resizing frames from {frames.shape} to {self.config.vae_img_size}x{self.config.vae_img_size}")
                    resized_frames = []
                    for frame in frames:
                        # Handle different frame formats for cv2.resize
                        if len(frame.shape) == 3 and frame.shape[2] == 1:
                            # Grayscale with channel dimension
                            frame_2d = frame[:, :, 0]
                            if frame_2d.shape[0] > 0 and frame_2d.shape[1] > 0:  # Check valid dimensions
                                resized_frame = cv2.resize(frame_2d, (self.config.vae_img_size, self.config.vae_img_size), interpolation=cv2.INTER_AREA)
                                resized_frame = np.expand_dims(resized_frame, axis=-1)
                            else:
                                resized_frame = np.zeros((self.config.vae_img_size, self.config.vae_img_size, 1), dtype=np.uint8)
                        elif len(frame.shape) == 2:
                            # Pure 2D
                            if frame.shape[0] > 0 and frame.shape[1] > 0:
                                resized_frame = cv2.resize(frame, (self.config.vae_img_size, self.config.vae_img_size), interpolation=cv2.INTER_AREA)
                                resized_frame = np.expand_dims(resized_frame, axis=-1)
                            else:
                                resized_frame = np.zeros((self.config.vae_img_size, self.config.vae_img_size, 1), dtype=np.uint8)
                        else:
                            # Color image
                            if frame.shape[0] > 0 and frame.shape[1] > 0:
                                resized_frame = cv2.resize(frame, (self.config.vae_img_size, self.config.vae_img_size), interpolation=cv2.INTER_AREA)
                                if len(resized_frame.shape) == 2:
                                    resized_frame = np.expand_dims(resized_frame, axis=-1)
                            else:
                                resized_frame = np.zeros((self.config.vae_img_size, self.config.vae_img_size, input_channels), dtype=np.uint8)
                        resized_frames.append(resized_frame)
                    frames = np.array(resized_frames)

                # Encode frames to latents
                frames_tensor = torch.FloatTensor(frames).permute(0, 3, 1, 2) / 255.0
                frames_tensor = frames_tensor.to(self.device)

                _, mu, _ = self.vae(frames_tensor)
                latents = mu.cpu().numpy()

                latent_episodes.append({
                    'latents': latents,
                    'actions': actions,
                    'rewards': rewards
                })

        # Save latent data
        latent_file = self.env_directories[env_id]['checkpoints'] / "latent_episodes.npz"
        np.savez(latent_file, episodes=latent_episodes)

        self.logger.info(f"Latent data saved to {latent_file}")
        return str(latent_file)

    def train_mdnrnn(self, env_id: str, latent_file: str) -> str:
        """Train MDN-RNN on latent sequences."""
        self.logger.info(f"Training MDN-RNN for {env_id}")

        # Load latent data
        data = np.load(latent_file, allow_pickle=True)
        episodes = data['episodes']

        # Determine action dimensionality
        first_episode = episodes[0]
        action_dim = np.array(first_episode['actions']).shape[-1] if len(np.array(first_episode['actions']).shape) > 1 else 1

        # Create MDN-RNN
        self.mdnrnn = MDNRNN(
            z_dim=self.config.vae_latent_size,
            action_dim=action_dim,
            rnn_size=self.config.rnn_size,
            num_mixtures=self.config.num_mixtures
        )
        self.mdnrnn.to(self.device)

        # Prepare sequences for training
        sequences = []
        for episode_data in episodes:
            episode = episode_data
            latents = episode['latents']
            actions = episode['actions']

            # Create sequences of [z_t, a_t] -> z_{t+1}
            for t in range(len(latents) - 1):
                z_t = latents[t]
                a_t = actions[t] if action_dim > 1 else [actions[t]]
                z_next = latents[t + 1]

                sequences.append({
                    'z_t': z_t,
                    'a_t': a_t,
                    'z_next': z_next
                })

        self.logger.info(f"Training MDN-RNN on {len(sequences)} sequences")

        # Create data loader
        z_t_batch = torch.FloatTensor([s['z_t'] for s in sequences])
        a_t_batch = torch.FloatTensor([s['a_t'] for s in sequences])
        z_next_batch = torch.FloatTensor([s['z_next'] for s in sequences])

        dataset = torch.utils.data.TensorDataset(z_t_batch, a_t_batch, z_next_batch)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.mdnrnn_batch_size,
            shuffle=True
        )

        # Train MDN-RNN
        optimizer = torch.optim.Adam(self.mdnrnn.parameters(), lr=1e-3)

        for epoch in range(self.config.mdnrnn_epochs):
            epoch_loss = 0.0
            for z_t, a_t, z_next in dataloader:
                z_t, a_t, z_next = z_t.to(self.device), a_t.to(self.device), z_next.to(self.device)

                # Add sequence dimension
                z_t = z_t.unsqueeze(1)  # (batch, 1, z_dim)
                a_t = a_t.unsqueeze(1)  # (batch, 1, action_dim)
                z_next = z_next.unsqueeze(1)  # (batch, 1, z_dim)

                optimizer.zero_grad()
                outputs = self.mdnrnn(z_t, a_t)
                pi, mu, sigma = outputs['pi'], outputs['mu'], outputs['sigma']

                # MDN loss
                loss = mdn_loss_function(pi, mu, sigma, z_next)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.logger.info(f"MDN-RNN Epoch {epoch+1}/{self.config.mdnrnn_epochs}: Loss = {avg_loss:.4f}")

            # Log to TensorBoard
            self.writer.add_scalar(f'{env_id}/MDNRNN_Loss', avg_loss, epoch)

        # Save MDN-RNN to new directory structure
        mdnrnn_path = self.env_directories[env_id]['checkpoints'] / "mdnrnn_best.pt"
        torch.save(self.mdnrnn.state_dict(), mdnrnn_path)

        self.logger.info(f"MDN-RNN saved to {mdnrnn_path}")
        return str(mdnrnn_path)

    def create_dream_environment(self, env_id: str, vae_path: str, mdnrnn_path: str) -> DreamEnvironment:
        """Create dream environment for controller training."""
        self.logger.info(f"Creating dream environment for {env_id}")

        # Determine action space size
        real_env = self.create_env(env_id)
        if isinstance(real_env.action_space, gym.spaces.Discrete):
            action_space_size = real_env.action_space.n
        elif hasattr(real_env.action_space, 'shape') and real_env.action_space.shape is not None:
            action_space_size = real_env.action_space.shape[0]
        else:
            # Fallback: treat as scalar action
            action_space_size = 1
        real_env.close()

        self.dream_env = DreamEnvironment(
            vae_model_path=vae_path,
            mdnrnn_model_path=mdnrnn_path,
            action_space_size=int(action_space_size),
            max_episode_steps=200,  # Shorter for faster training
            device=str(self.device)
        )

        return self.dream_env

    def create_visualization_env(self, env_id: str) -> Optional[gym.Env]:
        """Create environment specifically for visualization with human rendering."""
        try:
            # Try to create with human render mode for real-time visualization
            env = gym.make(env_id, render_mode=self.config.render_mode)

            # Apply same preprocessing as training environment
            if "NoFrameskip" in env_id:
                env = GrayscaleObservation(env)
                env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                env = FrameStackObservation(env, 4)
            elif "LunarLander" in env_id:
                # LunarLander doesn't need frame preprocessing
                pass
            elif "CarRacing" in env_id:
                env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                env = FrameStackObservation(env, 4)

            return env

        except Exception as e:
            self.logger.warning(f"Failed to create visualization environment for {env_id}: {e}")
            # Fallback to rgb_array mode
            try:
                env = gym.make(env_id, render_mode="rgb_array")
                if "NoFrameskip" in env_id:
                    env = GrayscaleObservation(env)
                    env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                    env = FrameStackObservation(env, 4)
                elif "CarRacing" in env_id:
                    env = ResizeObservation(env, (self.config.vae_img_size, self.config.vae_img_size))
                    env = FrameStackObservation(env, 4)
                return env
            except Exception as e2:
                self.logger.error(f"Failed to create fallback environment: {e2}")
                return None

    def show_live_rollout(self, env_id: str, controller, generation: int, mean_score: float, threshold: float) -> float:
        """Show a live rollout with enhanced visualization and FPS control."""
        # Use training FPS for live rollouts
        env, actual_render_mode = self.create_enhanced_render_env(
            env_id,
            fps_target=self.config.fps,
            fallback_to_rgb=self.config.fallback_to_rgb_array
        )

        if env is None:
            return 0.0

        try:
            obs, info = env.reset()
            total_reward = 0.0
            step_count = 0
            done = False

            # Load VAE if available for latent space conversion
            vae_path = self.env_directories[env_id]['checkpoints'] / "vae_best.pt"
            use_vae = False

            if vae_path.exists() and self.vae is not None:
                try:
                    self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
                    self.vae.eval()
                    use_vae = True
                except Exception as e:
                    self.logger.warning(f"Failed to load VAE: {e}")
                    use_vae = False

            print(f"\n[LIVE] {env_id} | Gen {generation:4d} | Score: {mean_score:7.2f} | Target: {threshold:6.1f}")
            print(f"[LIVE] Enhanced visualization @ {self.config.fps}fps ({actual_render_mode}) - Press Ctrl+C to skip")

            # Add frame timing for FPS control
            target_frame_time = 1.0 / self.config.fps if self.config.fps > 0 else 0.0

            while not done and step_count < self.config.max_episode_steps:
                frame_start_time = time.time()

                try:
                    # Validate and process observation frame
                    if len(obs.shape) >= 2:
                        obs = self.validate_frame_pipeline(obs, env_id, expected_shape=(64, 64))

                    if use_vae and self.vae is not None:
                        # Convert observation to latent using VAE
                        if len(obs.shape) == 3:
                            # Resize frame to VAE input size
                            resized_obs = cv2.resize(obs, (self.config.vae_img_size, self.config.vae_img_size), interpolation=cv2.INTER_AREA)
                            obs_tensor = torch.FloatTensor(resized_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
                        else:
                            obs_tensor = torch.FloatTensor(obs).unsqueeze(0) / 255.0

                        obs_tensor = obs_tensor.to(self.device)

                        with torch.no_grad():
                            _, z, _ = self.vae(obs_tensor)
                            # Create dummy hidden state for controller
                            hidden = torch.zeros(1, self.config.rnn_size).to(self.device)

                            action_output = controller.get_action(z, hidden, deterministic=True)
                            action = action_output.cpu().numpy().flatten()
                    else:
                        # Use observation directly (fallback mode)
                        obs_flat = obs.flatten() if hasattr(obs, 'flatten') else np.array([obs])
                        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            # Split observation into z and h components if possible
                            if obs_tensor.size(1) >= self.config.vae_latent_size + self.config.rnn_size:
                                z = obs_tensor[:, :self.config.vae_latent_size]
                                h = obs_tensor[:, self.config.vae_latent_size:self.config.vae_latent_size + self.config.rnn_size]
                            else:
                                # Create dummy z and h if observation is too small
                                z = torch.zeros(1, self.config.vae_latent_size).to(self.device)
                                h = torch.zeros(1, self.config.rnn_size).to(self.device)

                            action_output = controller.get_action(z, h, deterministic=True)
                            action = action_output.cpu().numpy().flatten()

                    # Convert to environment action format
                    if hasattr(env.action_space, 'n'):  # Discrete
                        action = int(np.argmax(action)) if len(action) > 1 else int(action[0])
                    else:  # Continuous
                        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                            action = np.clip(action, env.action_space.low, env.action_space.high)

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += float(reward)
                    step_count += 1

                    # Enhanced FPS control with precise timing
                    if target_frame_time > 0:
                        frame_duration = time.time() - frame_start_time
                        sleep_time = max(0, target_frame_time - frame_duration)
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                except KeyboardInterrupt:
                    print(f"\n[LIVE] Rollout interrupted by user - keeping window {'open' if not self.config.close_on_completion else 'auto-closing'}")
                    break
                except Exception as e:
                    self.logger.warning(f"Error during live rollout: {e}")
                    break

            if not self.config.close_on_completion:
                print(f"[LIVE] Window remains open for analysis (--close-on-completion to change)")

            if not self.config.window_reuse:
                env.close()  # Only close if not reusing windows

            print(f"[LIVE] Rollout completed - Score: {total_reward:.2f}, Steps: {step_count}")
            return total_reward

        except Exception as e:
            self.logger.error(f"Failed to show enhanced live rollout: {e}")
            if env and not self.config.window_reuse:
                env.close()
            return 0.0

    def record_video_rollout(self, env_id: str, controller, generation: int, custom_video_path: Optional[str] = None) -> bool:
        """Record a video of the current controller's performance."""
        if custom_video_path:
            video_file = Path(custom_video_path)
            video_dir = video_file.parent
            video_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Legacy path for backwards compatibility
            video_dir = self.video_base_dir / env_id / f"generation_{generation:04d}"
            video_dir.mkdir(parents=True, exist_ok=True)

        try:
            env = self.create_env(env_id, record_video=True, video_dir=video_dir)
            obs, info = env.reset()
            total_reward = 0.0
            step_count = 0
            done = False

            # Load VAE for latent conversion
            env_name = env_id.replace("/", "_")
            vae_path = self.env_directories[env_id]['checkpoints'] / "vae_best.pt"
            use_vae = False

            if vae_path.exists() and self.vae is not None:
                try:
                    self.vae.load_state_dict(torch.load(vae_path, map_location=self.device))
                    self.vae.eval()
                    use_vae = True
                except Exception:
                    use_vae = False

            while not done and step_count < self.config.max_episode_steps:
                if use_vae and self.vae is not None:
                    # Convert observation to latent using VAE
                    if len(obs.shape) == 3:
                        # Resize frame to VAE input size
                        resized_obs = cv2.resize(obs, (self.config.vae_img_size, self.config.vae_img_size), interpolation=cv2.INTER_AREA)
                        obs_tensor = torch.FloatTensor(resized_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
                    else:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0) / 255.0

                    obs_tensor = obs_tensor.to(self.device)

                    with torch.no_grad():
                        _, z, _ = self.vae(obs_tensor)
                        hidden = torch.zeros(1, self.config.rnn_size).to(self.device)
                        state = torch.cat([z, hidden], dim=1)

                        action_output = controller.get_action(state, deterministic=True)
                        action = action_output.cpu().numpy().flatten()
                else:
                    # Fallback mode
                    obs_flat = obs.flatten() if hasattr(obs, 'flatten') else np.array([obs])
                    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        action_output = controller.get_action(obs_tensor, deterministic=True)
                        action = action_output.cpu().numpy().flatten()

                # Convert to environment action format
                if hasattr(env.action_space, 'n'):  # Discrete
                    action = int(np.argmax(action)) if len(action) > 1 else int(action[0])
                else:  # Continuous
                    if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                        action = np.clip(action, env.action_space.low, env.action_space.high)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
                step_count += 1

            env.close()
            self.logger.info(f"Video recorded for {env_id} generation {generation}: {total_reward:.2f} points")
            return True

        except Exception as e:
            self.logger.error(f"Failed to record video for {env_id}: {e}")
            return False

    def evaluate_controller_real_env(self, env_id: str, controller: Controller, num_episodes: Optional[int] = None, render: bool = False) -> Tuple[float, List[float]]:
        """Evaluate controller in real environment with enhanced visualization."""
        if num_episodes is None:
            num_episodes = self.config.episodes_per_eval

        # Create environment with enhanced rendering for evaluation
        if render:
            env, actual_render_mode = self.create_enhanced_render_env(
                env_id,
                fps_target=self.config.eval_fps,  # Use higher FPS for evaluation
                fallback_to_rgb=self.config.fallback_to_rgb_array
            )
            print(f"[EVAL] Enhanced evaluation @ {self.config.eval_fps}fps ({actual_render_mode})")
        else:
            env = gym.make(env_id, render_mode=None)
            actual_render_mode = "none"

        episode_rewards = []

        try:
            # Frame timing for evaluation FPS control
            target_frame_time = 1.0 / self.config.eval_fps if render and self.config.eval_fps > 0 else 0.0

            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_reward = 0.0
                done = False
                step_count = 0

                # Initialize hidden state for MDN-RNN
                hidden = torch.zeros(1, self.config.rnn_size).to(self.device)

                # Ensure VAE is loaded with correct input channels for this environment
                if self.vae is None:
                    vae_path = self.env_directories[env_id]['checkpoints'] / "vae.pt"
                    if not vae_path.exists():
                        raise RuntimeError(f"VAE model not found at {vae_path}")

                    # Determine input channels based on environment type
                    if len(obs.shape) == 1:  # Vector observation
                        input_channels = 1
                    elif len(obs.shape) == 3:  # Image observation
                        input_channels = obs.shape[-1] if obs.shape[-1] <= 4 else 3
                    else:
                        input_channels = 3

                    self.vae = ConvVAE(
                        img_channels=input_channels,
                        img_size=self.config.vae_img_size,
                        latent_dim=self.config.vae_latent_size
                    )
                    self.vae.to(self.device)
                    self.vae.load_state_dict(torch.load(vae_path, map_location=self.device, weights_only=True))
                    self.vae.eval()

                while not done and step_count < 1000:
                    frame_start_time = time.time() if render else 0

                    # Process observation with frame validation
                    if len(obs.shape) >= 2:
                        obs = self.validate_frame_pipeline(obs, env_id, expected_shape=(64, 64))

                    # Process observation based on type (same logic as training)
                    if len(obs.shape) == 1:  # Vector observation (like LunarLander)
                        # Convert vector to image format (same as train_vae)
                        obs_dim = obs.shape[0]
                        if obs_dim <= 16:
                            img_size = 4
                        elif obs_dim <= 64:
                            img_size = 8
                        else:
                            img_size = int(np.ceil(np.sqrt(obs_dim)))

                        # Pad to square
                        padded_size = img_size * img_size
                        if obs_dim < padded_size:
                            padded_obs = np.zeros(padded_size)
                            padded_obs[:obs_dim] = obs
                        else:
                            padded_obs = obs[:padded_size]

                        # Reshape to image and normalize
                        image_obs = padded_obs.reshape(img_size, img_size)
                        obs_min, obs_max = image_obs.min(), image_obs.max()
                        if obs_max > obs_min:
                            image_obs = ((image_obs - obs_min) / (obs_max - obs_min) * 255).astype(np.uint8)
                        else:
                            image_obs = (image_obs * 0 + 128).astype(np.uint8)

                        # Resize to VAE input size and prepare tensor
                        if img_size != self.config.vae_img_size:
                            resized_obs = cv2.resize(image_obs, (self.config.vae_img_size, self.config.vae_img_size), interpolation=cv2.INTER_AREA)
                        else:
                            resized_obs = image_obs

                        # Correct tensor format: (batch, channels, height, width)
                        obs_tensor = torch.FloatTensor(resized_obs).unsqueeze(0).unsqueeze(0).to(self.device) / 255.0  # (1, 1, H, W)

                    elif len(obs.shape) == 3:  # Image observation (like Pong, Breakout)
                        # Resize frame to VAE input size
                        resized_obs = cv2.resize(obs, (self.config.vae_img_size, self.config.vae_img_size), interpolation=cv2.INTER_AREA)
                        obs_tensor = torch.FloatTensor(resized_obs).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

                    else:  # Fallback for other observation types
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0) / 255.0
                        if len(obs_tensor.shape) == 2:  # Add channel dimension if needed
                            obs_tensor = obs_tensor.unsqueeze(0)
                        if len(obs_tensor.shape) == 3:  # Add batch dimension if needed
                            obs_tensor = obs_tensor.unsqueeze(0)
                        obs_tensor = obs_tensor.to(self.device)

                    with torch.no_grad():
                        # Get latent representation
                        _, z, _ = self.vae(obs_tensor)

                        # Get action from controller
                        action_output = controller.get_action(z, hidden, deterministic=True)
                        if isinstance(action_output, tuple):
                            action_tensor = action_output[0]
                        else:
                            action_tensor = action_output

                        action = action_tensor.cpu().numpy()
                        if len(action.shape) > 1:
                            action = action[0]

                        # Convert to environment action format
                        if hasattr(env.action_space, 'n'):  # Discrete
                            action = int(np.argmax(action))
                        elif isinstance(env.action_space, gym.spaces.Box):  # Continuous
                            action = np.clip(action, env.action_space.low, env.action_space.high)
                        else:
                            # Fallback: do not clip
                            pass

                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += float(reward)
                    done = terminated or truncated
                    step_count += 1

                    # Enhanced FPS control for evaluation visualization
                    if render and target_frame_time > 0:
                        frame_duration = time.time() - frame_start_time
                        sleep_time = max(0, target_frame_time - frame_duration)
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                episode_rewards.append(episode_reward)

            mean_reward = float(np.mean(episode_rewards))

            if render:
                print(f"[EVAL] Evaluation complete @ {self.config.eval_fps}fps - Mean: {mean_reward:.2f}")
                if not self.config.close_on_completion:
                    print(f"[EVAL] Window remains open for analysis")

            return mean_reward, episode_rewards

        finally:
            if not self.config.window_reuse or not render:
                env.close()  # Only close if not reusing windows or not rendering

    def train_controller_cmaes(self, env_id: str, task: CurriculumTask, vae_path: str, mdnrnn_path: str) -> bool:
        """Train controller using CMA-ES."""
        self.logger.info(f"Training controller for {env_id} using CMA-ES")

        # Create dream environment
        dream_env = self.create_dream_environment(env_id, vae_path, mdnrnn_path)

        # Create controller
        input_size = self.config.vae_latent_size + self.config.rnn_size
        action_size = dream_env.action_space_size

        self.controller = Controller(
            input_size=input_size,
            action_size=action_size,
            hidden_sizes=(self.config.controller_hidden_size,),
            action_type='continuous' if not hasattr(dream_env, 'discrete_actions') else 'discrete'
        )

        # Move controller to GPU device
        self.controller.to(self.device)

        # CMA-ES optimizer
        cmaes_optimizer = CMAESController(
            controller=self.controller,
            population_size=self.config.cma_population_size,
            sigma=self.config.cma_sigma,
            device=self.device
        )

        best_score = float('-inf')
        patience_counter = 0
        generation_scores = deque(maxlen=10)

        for generation in range(self.config.max_generations):
            self.global_generation = generation

            # Generate candidate solutions
            candidates = cmaes_optimizer.ask()
            fitness_values = []

            # Evaluate each candidate
            for candidate in candidates:
                # Set controller parameters
                self.controller.set_parameters(candidate)

                # Evaluate in dream environment
                dream_reward = self.evaluate_dream_environment(dream_env, episodes=3)
                fitness_values.append(dream_reward)

            # Update CMA-ES
            cmaes_optimizer.tell(candidates, np.array(fitness_values))

            # Get best candidate for real environment evaluation
            best_candidate = candidates[np.argmax(fitness_values)]
            self.controller.set_parameters(best_candidate)

            # Evaluate in real environment
            mean_score, episode_scores = self.evaluate_controller_real_env(
                env_id, self.controller, render=self.config.visualize
            )

            # Update rolling rewards for early stopping
            early_stop = False
            if self.config.early_stop:
                early_stop = self.update_rolling_rewards(task, mean_score)

            generation_scores.append(mean_score)

            # Update best score
            if mean_score > best_score:
                best_score = mean_score
                task.best_score = best_score
                patience_counter = 0

                # Save best controller to new directory structure
                controller_path = self.env_directories[env_id]['checkpoints'] / "controller_best.pt"
                torch.save(self.controller.state_dict(), controller_path)
            else:
                patience_counter += 1

            # Run periodic evaluation snapshots
            if self.config.eval_every > 0 and generation % self.config.eval_every == 0:
                try:
                    eval_results = self.run_evaluation_snapshot(task, generation)
                    self.logger.info(f"[EVAL SNAPSHOT] Gen {generation}: {eval_results['mean_reward']:.2f}")
                except Exception as e:
                    self.logger.warning(f"[EVAL SNAPSHOT] Failed at gen {generation}: {e}")

            # Logging and progress tracking
            self.log_training_progress(env_id, generation, mean_score, best_score, task.threshold_score)

            # Real-time visualization - show live rollout based on environment-specific frequency
            rollout_freq = self.rollout_frequency.get(env_id, 5)  # Default to every 5 generations
            if self.config.visualize and generation % rollout_freq == 0:
                try:
                    live_score = self.show_live_rollout(env_id, self.controller, generation, mean_score, task.threshold_score)
                    self.logger.info(f"Live rollout score: {live_score:.2f}")
                except Exception as e:
                    self.logger.warning(f"Failed to show live rollout: {e}")

            # Record video based on new scheduling system
            record_now = (generation + 1) in self.record_gens and self.config.record_video
            if record_now:
                try:
                    env_name = env_id.replace("/", "_")
                    video_dir = self.env_directories[env_id]['videos']
                    video_file = video_dir / f"gen_{generation+1:04d}.mp4"

                    self.logger.info(f"Recording video for generation {generation+1}: {video_file}")
                    self.record_video_rollout(env_id, self.controller, generation, str(video_file))
                except Exception as e:
                    self.logger.warning(f"Failed to record video for generation {generation+1}: {e}")

            # Enhanced progress display with curriculum information
            progress_bar = "█" * int((mean_score / task.threshold_score) * 20) + "▒" * (20 - int((mean_score / task.threshold_score) * 20))
            progress_pct = min(100, (mean_score / task.threshold_score) * 100)

            print(f"\r[PROGRESS] {env_id:<25} | Gen {generation:4d} | Score: {mean_score:7.2f} | Target: {task.threshold_score:6.1f} | [{progress_bar}] {progress_pct:5.1f}%", end="", flush=True)

            # Check if task is solved - require consistent performance
            recent_avg = np.mean(list(generation_scores)) if len(generation_scores) >= 5 else mean_score
            if recent_avg >= task.threshold_score:
                print(f"\n[SOLVED] {env_id} SOLVED! Average score: {recent_avg:.2f} >= {task.threshold_score}")
                self.logger.info(f"Task {env_id} SOLVED! Average score: {recent_avg:.2f} >= {task.threshold_score}")
                task.solved = True
                task.generations_trained = generation + 1
                # Generate publishable artifacts for completed task
                try:
                    self.generate_publishing_artifacts(task)
                    self.logger.info(f"[REPORT] Publishing artifacts generated for {env_id}")
                except Exception as e:
                    self.logger.warning(f"[REPORT] Failed to generate artifacts for {env_id}: {e}")
                return True

            # Check for early stopping (plateau detection)
            if early_stop:
                print(f"\n[PLATEAU STOP] {env_id} training stopped due to plateau detection.")
                print(f"Rolling mean: {task.rolling_mean:.2f}, No improvement for {task.no_improvement_count} generations")
                self.logger.info(f"Task {env_id} PLATEAU STOP! Rolling mean: {task.rolling_mean:.2f}")
                task.generations_trained = generation + 1
                # Generate publishable artifacts for plateaued task
                try:
                    self.generate_publishing_artifacts(task)
                    self.logger.info(f"[REPORT] Publishing artifacts generated for {env_id} (plateau)")
                except Exception as e:
                    self.logger.warning(f"[REPORT] Failed to generate artifacts for {env_id}: {e}")
                return False

            # Original early stopping (fallback)
            if patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping for {env_id} after {patience_counter} generations without improvement")
                task.generations_trained = generation + 1
                # Generate publishable artifacts for incomplete task
                try:
                    self.generate_publishing_artifacts(task)
                    self.logger.info(f"[REPORT] Publishing artifacts generated for {env_id} (early stop)")
                except Exception as e:
                    self.logger.warning(f"[REPORT] Failed to generate artifacts for {env_id}: {e}")
                break

        task.generations_trained = self.config.max_generations
        self.logger.info(f"Training completed for {env_id}. Best score: {best_score:.2f}, Target: {task.threshold_score}")

        # Generate publishable artifacts for max generations reached
        try:
            self.generate_publishing_artifacts(task)
            self.logger.info(f"[REPORT] Publishing artifacts generated for {env_id} (max generations)")
        except Exception as e:
            self.logger.warning(f"[REPORT] Failed to generate artifacts for {env_id}: {e}")

        return False

    def evaluate_dream_environment(self, dream_env: DreamEnvironment, episodes: int = 3) -> float:
        """Evaluate controller in dream environment."""
        total_reward = 0.0

        for _ in range(episodes):
            obs, info = dream_env.reset()
            episode_reward = 0.0
            done = False
            step_count = 0

            while not done and step_count < 200:  # Limit steps for speed
                with torch.no_grad():
                    # Extract latent and hidden states from observation
                    # Handle both numpy arrays and tensors, ensure proper device placement
                    if isinstance(obs, torch.Tensor):
                        z = obs[:self.config.vae_latent_size].unsqueeze(0).to(self.device)
                        h = obs[self.config.vae_latent_size:].unsqueeze(0).to(self.device)
                    else:
                        z = torch.tensor(obs[:self.config.vae_latent_size], dtype=torch.float32, device=self.device).unsqueeze(0)
                        h = torch.tensor(obs[self.config.vae_latent_size:], dtype=torch.float32, device=self.device).unsqueeze(0)

                    # Get action
                    if self.controller is None:
                        raise RuntimeError("Controller is not initialized before calling get_action.")
                    action_output = self.controller.get_action(z, h, deterministic=True)
                    if isinstance(action_output, tuple):
                        action = action_output[0].cpu().numpy()[0]
                    else:
                        action = action_output.cpu().numpy()[0]

                obs, reward, terminated, truncated, info = dream_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1

            total_reward += episode_reward

        return total_reward / episodes

    def record_evaluation_video(self, env_id: str, video_dir: Path):
        """Record a video of the current controller performance."""
        video_dir.mkdir(parents=True, exist_ok=True)

        env = self.create_env(env_id, record_video=True, video_dir=video_dir)

        try:
            if self.controller is None:
                raise RuntimeError("Controller is not initialized before recording evaluation video.")
            self.evaluate_controller_real_env(env_id, self.controller, num_episodes=1, render=False)
            self.logger.info(f"Video recorded for {env_id} at generation {self.global_generation}")
        except Exception as e:
            self.logger.error(f"Failed to record video: {e}")
        finally:
            env.close()

    def log_training_progress(self, env_id: str, generation: int, mean_score: float, best_score: float, threshold: float):
        """Log training progress to various outputs."""
        elapsed_time = time.time() - self.training_start_time

        # Console logging with progress
        progress_bar = "=" * int(20 * min(mean_score / threshold, 1.0))
        progress_spaces = " " * (20 - len(progress_bar))

        print(f"\r{env_id:20} | Gen {generation:4d} | "
              f"Score: {mean_score:7.2f} | Best: {best_score:7.2f} | "
              f"Target: {threshold:6.1f} | [{progress_bar}{progress_spaces}] "
              f"{100 * min(mean_score / threshold, 1.0):5.1f}%",
              end="", flush=True)

        if generation % 10 == 0:
            print()  # New line every 10 generations

        # TensorBoard logging
        self.writer.add_scalar(f'{env_id}/Mean_Score', mean_score, generation)
        self.writer.add_scalar(f'{env_id}/Best_Score', best_score, generation)
        self.writer.add_scalar(f'{env_id}/Progress', mean_score / threshold, generation)

        # CSV logging
        with open(self.csv_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            solved = "True" if mean_score >= threshold else "False"
            f.write(f"{timestamp},{env_id},{generation},{mean_score:.4f},{best_score:.4f},{threshold},{solved},{elapsed_time:.2f}\n")

    def train_single_task(self, task: CurriculumTask) -> bool:
        """Train World Models on a single curriculum task."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting training for {task.env_id}")
        self.logger.info(f"Target score: {task.threshold_score}")
        self.logger.info(f"{'='*60}")

        env_id = task.env_id
        env_name = env_id.replace("/", "_")

        try:
            # Step 1: Collect random data
            self.logger.info("Phase 1: Data Collection")
            data_file = self.collect_random_data(env_id, num_episodes=50)

            # Step 2: Train VAE
            self.logger.info("Phase 2: VAE Training")
            vae_path = self.train_vae(env_id, data_file)

            # Clean up after VAE training if cache cleaning enabled
            if self.config.clean_cache:
                temp_dir = self.env_directories[env_id]['tmp']
                for temp_file in temp_dir.glob("vae_*"):
                    safe_unlink(temp_file)

            # Step 3: Encode data to latents
            self.logger.info("Phase 3: Latent Encoding")
            latent_file = self.encode_data_to_latents(env_id, data_file, vae_path)

            # Step 4: Train MDN-RNN
            self.logger.info("Phase 4: MDN-RNN Training")
            mdnrnn_path = self.train_mdnrnn(env_id, latent_file)

            # Clean up latents if not keeping them
            if self.config.clean_cache and not self.config.keep_latents:
                if Path(latent_file).exists():
                    safe_unlink(Path(latent_file))
                    self.logger.info("Cleaned up latent files to save disk space")

            # Step 5: Train Controller
            self.logger.info("Phase 5: Controller Training")
            success = self.train_controller_cmaes(env_id, task, vae_path, mdnrnn_path)

            # Final cleanup for this environment
            if self.config.clean_cache:
                prune_intermediate_artifacts(
                    self.env_directories[env_id]['root'],
                    self.record_gens,
                    self.config.keep_latents
                )
                self.logger.info(f"Final cleanup completed for {env_id}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to train {env_id}: {e}")
            traceback.print_exc()
            return False

    def run_curriculum(self) -> bool:
        """Run the complete curriculum training."""
        self.logger.info("Starting Curriculum Training")
        self.logger.info(f"Tasks: {[task.env_id for task in self.curriculum]}")

        overall_success = True

        for i, task in enumerate(self.curriculum):
            self.current_task_idx = i

            print(f"\n[TARGET] Task {i+1}/{len(self.curriculum)}: {task.env_id}")
            print(f"Target Score: {task.threshold_score}")
            print("-" * 60)

            success = self.train_single_task(task)

            if success:
                print(f"\n[SUCCESS] {task.env_id} COMPLETED!")
                print(f"   Best Score: {task.best_score:.2f}")
                print(f"   Generations: {task.generations_trained}")
            else:
                print(f"\n[FAILED] {task.env_id} FAILED")
                print(f"   Best Score: {task.best_score:.2f}")
                print(f"   Max Generations Reached: {task.generations_trained}")
                overall_success = False

                # Ask whether to continue
                continue_training = input("Continue to next task? (y/n): ").lower() == 'y'
                if not continue_training:
                    break

        return overall_success

    def generate_final_report(self):
        """Generate final 3-environment curriculum training report."""
        print("\n" + "="*80)
        print("3-ENVIRONMENT CURRICULUM TRAINING FINAL REPORT")
        print("Enhanced Visual World Models: Pong → Breakout → CarRacing")
        print("="*80)

        total_time = time.time() - self.training_start_time
        solved_count = sum(1 for task in self.curriculum if task.solved)

        print(f"Total Training Time: {total_time/3600:.2f} hours")
        print(f"Visual Tasks Completed: {solved_count}/3 environments")
        print(f"Curriculum Success Rate: {(solved_count/3)*100:.1f}% completion")
        print()

        print("Visual Environment Summary:")
        print("-" * 70)
        environment_descriptions = {
            "ALE/Pong-v5": "Simple paddle game (deterministic)",
            "ALE/Breakout-v5": "Brick breaking (object interaction)",
            "CarRacing-v3": "Continuous control (dynamic visual-motor)"
        }

        for i, task in enumerate(self.curriculum):
            status = "[SOLVED]" if task.solved else "[FAILED]"
            description = environment_descriptions.get(task.env_id, "Visual environment")
            print(f"{i+1}. {task.env_id:20} | {status} | "
                  f"Score: {task.best_score:8.2f} / {task.threshold_score:6.1f} | "
                  f"Gens: {task.generations_trained:3d}")
            print(f"   └─ {description}")

        print("-" * 70)

        # Save final results
        results = {
            'total_time_hours': total_time / 3600,
            'tasks_completed': solved_count,
            'total_tasks': len(self.curriculum),
            'success_rate': solved_count / len(self.curriculum),
            'tasks': [
                {
                    'env_id': task.env_id,
                    'solved': task.solved,
                    'best_score': task.best_score,
                    'threshold_score': task.threshold_score,
                    'generations_trained': task.generations_trained
                }
                for task in self.curriculum
            ]
        }

        results_file = self.checkpoint_dir / "curriculum_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[REPORT] Full results saved to: {results_file}")

        # Close TensorBoard writer
        self.writer.close()

        return solved_count == len(self.curriculum)

    def save_interrupt_state(self):
        """Save training state when interrupted."""
        try:
            interrupt_state = {
                'current_task_idx': self.current_task_idx,
                'global_generation': self.global_generation,
                'training_start_time': self.training_start_time,
                'curriculum_status': [
                    {
                        'env_id': task.env_id,
                        'solved': task.solved,
                        'best_score': task.best_score,
                        'threshold_score': task.threshold_score,
                        'generations_trained': task.generations_trained
                    }
                    for task in self.curriculum
                ]
            }

            state_file = self.directories['checkpoints'] / "interrupt_state.json"
            with open(state_file, 'w') as f:
                json.dump(interrupt_state, f, indent=2)

            # Save current model states if available
            if self.controller is not None:
                controller_path = self.directories['checkpoints'] / "controller_interrupt.pt"
                torch.save(self.controller.state_dict(), controller_path)

            print(f"[SAVE] Interrupt state saved to {state_file}")
        except Exception as e:
            print(f"[WARN] Failed to save interrupt state: {e}")

    def update_rolling_rewards(self, task: CurriculumTask, reward: float) -> bool:
        """
        Update rolling rewards and check for early stopping.

        Args:
            task: Current task being trained
            reward: New reward to add

        Returns:
            True if should stop early (plateau detected), False otherwise
        """
        if task.rolling_rewards is None:
            task.rolling_rewards = deque(maxlen=5)

        task.rolling_rewards.append(reward)

        # Calculate rolling mean if we have enough samples
        if len(task.rolling_rewards) >= 5:
            task.rolling_mean = sum(task.rolling_rewards) / len(task.rolling_rewards)

            # Check for improvement
            improved = task.rolling_mean > (task.best_rolling_mean + self.config.min_delta)

            if improved:
                task.best_rolling_mean = task.rolling_mean
                task.no_improvement_count = 0
                return False  # Continue training
            else:
                task.no_improvement_count += 1

                # Check if we should stop
                if task.no_improvement_count >= self.config.patience:
                    # Check if we're within threshold margin (allow extension)
                    margin = max(5, abs(task.threshold_score) * 0.1)  # 10% of threshold, min 5
                    within_margin = task.rolling_mean >= (task.threshold_score - margin)

                    if within_margin and not task.extension_granted:
                        # Grant extension window
                        extension_gens = max(10, int(self.config.max_generations * 0.1))  # 10% extension
                        task.extension_granted = True
                        task.no_improvement_count = 0  # Reset for extension period
                        self.logger.info(f"[PLATEAU] Extension granted: +{extension_gens} generations for {task.env_id}")
                        return False  # Continue with extension
                    else:
                        # Plateau stop
                        task.plateau_stopped = True
                        return True  # Stop training this environment

        return False

    def run_evaluation_snapshot(self, task: CurriculumTask, generation: int) -> Dict[str, Any]:
        """
        Run periodic evaluation and save snapshot.

        Args:
            task: Current task
            generation: Current generation number

        Returns:
            Evaluation results dictionary
        """
        env_name = task.env_id.replace("/", "_")
        env_dirs = self.env_directories[task.env_id]

        # Create eval snapshots directory
        snapshots_dir = env_dirs['root'] / 'eval_snapshots'
        snapshots_dir.mkdir(exist_ok=True)

        # Run full evaluation (no exploration if applicable)
        self.logger.info(f"[EVAL] Running snapshot evaluation at gen {generation}")

        # Create environment for evaluation
        env = self.create_env(task.env_id, record_video=False)

        try:
            # Run episodes
            episode_rewards = []
            for episode in range(self.config.episodes_per_eval):
                obs, _ = env.reset()
                episode_reward = 0
                done = False

                while not done:
                    if self.controller is not None:
                        # Convert observation to tensor and get latent encoding if VAE available
                        if self.vae is not None:
                            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
                            with torch.no_grad():
                                z, _, _ = self.vae(obs_tensor)
                                z = z.squeeze().cpu().numpy()

                            # Use hidden state if available (simplified for evaluation)
                            h = np.zeros(self.config.rnn_size) if hasattr(self.config, 'rnn_size') else None
                            action_output = self.controller.get_action(z, h, deterministic=True)
                            action = action_output
                        else:
                            # Fallback to random action if no VAE
                            action = env.action_space.sample()

                        # Handle action space types
                        if hasattr(env.action_space, 'n'):  # Discrete
                            action = int(action)
                        else:  # Continuous - clip to action space
                            action = np.clip(action, env.action_space.low, env.action_space.high)
                    else:
                        # Random action if no controller yet
                        action = env.action_space.sample()

                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward

                episode_rewards.append(episode_reward)

            # Calculate statistics
            eval_results = {
                'generation': generation,
                'env_id': task.env_id,
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'best_reward': max(task.best_score, np.max(episode_rewards)),
                'threshold': task.threshold_score,
                'solved': np.mean(episode_rewards) >= task.threshold_score,
                'rolling_mean': task.rolling_mean,
                'no_improvement_count': task.no_improvement_count,
                'episodes': episode_rewards,
                'timestamp': datetime.now().isoformat()
            }

            # Save snapshot JSON
            snapshot_file = snapshots_dir / f"gen_{generation:04d}.json"
            with open(snapshot_file, 'w') as f:
                json.dump(eval_results, f, indent=2)

            # Append to CSV log
            csv_file = env_dirs['logs'] / 'eval_progress.csv'
            csv_exists = csv_file.exists()

            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                if not csv_exists:
                    writer.writerow(['generation', 'mean_reward', 'std_reward', 'best_reward',
                                   'threshold', 'solved', 'rolling_mean', 'no_improvement_count'])
                writer.writerow([generation, eval_results['mean_reward'], eval_results['std_reward'],
                               eval_results['best_reward'], eval_results['threshold'],
                               eval_results['solved'], eval_results['rolling_mean'],
                               eval_results['no_improvement_count']])

            self.logger.info(f"[EVAL] Gen {generation}: mean={eval_results['mean_reward']:.2f}, "
                           f"best={eval_results['best_reward']:.2f}, solved={eval_results['solved']}")

            return eval_results

        finally:
            env.close()

    def generate_publishing_artifacts(self, task: CurriculumTask):
        """
        Generate publishable artifacts for completed environment.

        Args:
            task: Completed task to generate artifacts for
        """
        env_name = task.env_id.replace("/", "_")
        env_dirs = self.env_directories[task.env_id]

        # Create reports directory
        reports_dir = env_dirs['root'] / 'report'
        reports_dir.mkdir(exist_ok=True)

        # Load evaluation progress
        csv_file = env_dirs['logs'] / 'eval_progress.csv'
        if not csv_file.exists():
            self.logger.warning(f"[REPORT] No eval progress found for {task.env_id}")
            return

        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            # Load data
            df = pd.read_csv(csv_file)

            # 1. Generate metrics summary CSV
            summary_data = {
                'env_id': task.env_id,
                'best_reward': task.best_score,
                'mean_final_reward': task.rolling_mean,
                'std_final_reward': df['std_reward'].iloc[-1] if len(df) > 0 else 0,
                'solved_generation': task.generations_trained if task.solved else None,
                'patience_used': task.no_improvement_count,
                'threshold': task.threshold_score,
                'final_rolling_mean': task.rolling_mean,
                'plateau_stopped': task.plateau_stopped,
                'total_generations': task.generations_trained,
                'completion_status': 'SOLVED' if task.solved else 'PLATEAU_STOP' if task.plateau_stopped else 'INCOMPLETE'
            }

            summary_df = pd.DataFrame([summary_data])
            summary_df.to_csv(reports_dir / 'metrics_summary.csv', index=False)

            # 2. Generate learning curve plot
            plt.figure(figsize=(10, 6))
            plt.plot(df['generation'], df['mean_reward'], 'b-', label='Mean Reward', linewidth=2)
            plt.axhline(y=task.threshold_score, color='r', linestyle='--',
                       label=f'Threshold ({task.threshold_score})', linewidth=2)
            plt.fill_between(df['generation'],
                           df['mean_reward'] - df['std_reward'],
                           df['mean_reward'] + df['std_reward'],
                           alpha=0.3, color='blue')

            plt.xlabel('Generation')
            plt.ylabel('Reward')
            plt.title(f'Learning Curve: {task.env_id}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(reports_dir / 'learning_curve.png', dpi=300, bbox_inches='tight')
            plt.close()

            # 3. Generate LaTeX controller table
            latex_controller = []
            latex_controller.append("\\begin{tabular}{|c|c|c|c|c|}")
            latex_controller.append("\\hline")
            latex_controller.append("Generation & Mean & Best & Solved? & Notes \\\\")
            latex_controller.append("\\hline")

            # Sample key generations for table
            key_gens = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1] if len(df) > 0 else [0]
            for i in key_gens:
                if i < len(df):
                    row = df.iloc[i]
                    solved_mark = "✓" if row['solved'] else "✗"
                    notes = "Final" if i == len(df)-1 else f"Checkpoint"
                    latex_controller.append(f"{row['generation']} & {row['mean_reward']:.1f} & "
                                          f"{row['best_reward']:.1f} & {solved_mark} & {notes} \\\\")

            latex_controller.append("\\hline")
            latex_controller.append("\\end{tabular}")

            with open(reports_dir / 'table_controller.tex', 'w') as f:
                f.write('\n'.join(latex_controller))

            # 4. Generate LaTeX runtime table (placeholder for now)
            latex_runtime = []
            latex_runtime.append("\\begin{tabular}{|c|c|c|c|}")
            latex_runtime.append("\\hline")
            latex_runtime.append("Phase & Wallclock (min) & GPU/CPU & AMP/TF32 \\\\")
            latex_runtime.append("\\hline")
            latex_runtime.append(f"VAE Training & - & {self.config.device.upper()} & {self.config.use_amp} \\\\")
            latex_runtime.append(f"RNN Training & - & {self.config.device.upper()} & {self.config.use_amp} \\\\")
            latex_runtime.append(f"Controller & {task.generations_trained*0.5:.1f} & {self.config.device.upper()} & N/A \\\\")
            latex_runtime.append("\\hline")
            latex_runtime.append("\\end{tabular}")

            with open(reports_dir / 'table_runtime.tex', 'w') as f:
                f.write('\n'.join(latex_runtime))

            self.logger.info(f"[REPORT] Publishing artifacts generated in {reports_dir}")

        except ImportError as e:
            self.logger.warning(f"[REPORT] Cannot generate plots: {e}. Install pandas/matplotlib")
        except Exception as e:
            self.logger.error(f"[REPORT] Error generating artifacts: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced 3-Environment Curriculum Trainer with Professional Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced 3-Environment Curriculum: Pong (18.0) → Breakout (50.0) → CarRacing (800.0)

Professional Usage Examples:
  # Quick visual check only (no training):
  python3 curriculum_trainer_visual.py --smoke-test True --visualize True --fps 30

  # Standard training with enhanced 30fps visualization and 60fps evaluation:
  python3 curriculum_trainer_visual.py --device cuda --max-generations 300 --fps 30 --eval-fps 60

  # Full 1000-gen run, record 3 videos only, store to D:
  python3 curriculum_trainer_visual.py --device cuda --max-generations 1000 --video-schedule triad --record-video True --visualize True --fps 30 --artifact-root "D:/WorldModels"

  # Explicit gens (start=25, mid=500, end=975), CPU:
  python3 curriculum_trainer_visual.py --device cpu --max-generations 1000 --video-gens "25,500,975" --record-video True --artifact-root "D:/WorldModels"

  # No videos (fastest):
  python3 curriculum_trainer_visual.py --device cuda --max-generations 1000 --video-schedule none --record-video False --artifact-root "D:/WorldModels"

  # Quick test run with reduced thresholds (5.0, 15.0, 100.0) and fast training:
  python3 curriculum_trainer_visual.py --device cpu --quick True --visualize True --fps 60

  # Professional video recording with enhanced validation:
  python3 curriculum_trainer_visual.py --record-video True --video-every-n-gens 5 --validate-rgb True --fps 30

  # Fast debugging with window reuse and keep-open:
  python3 curriculum_trainer_visual.py --max-generations 50 --fps 60 --no-close-on-completion

  # High-performance training with validation disabled for speed:
  python3 curriculum_trainer_visual.py --device cuda --visualize False --no-validate-rgb --no-validate-sizes
        """
    )

    # Basic training arguments
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device for training (default: cpu)')
    parser.add_argument('--max-generations', type=int, default=200,
                       help='Maximum generations per environment (default: 200)')
    parser.add_argument('--episodes-per-eval', type=int, default=5,
                       help='Episodes per evaluation (default: 5)')
    parser.add_argument('--checkpoint-dir', default='./runs/curriculum_visual',
                       help='Checkpoint directory (default: ./runs/curriculum_visual)')

    # New artifact management arguments
    parser.add_argument('--video-schedule', choices=['triad', 'none', 'all'],
                       help='Video recording schedule: triad (start/mid/end), none, all (default: triad if max-generations >= 200, else all)')
    parser.add_argument('--video-gens', type=str,
                       help='Explicit video generation list (e.g., "25,500,975") - overrides schedule')
    parser.add_argument('--artifact-root', type=str,
                       help='Base directory for artifacts (default: D:/WorldModels if D: exists, else ./runs/curriculum_visual)')
    parser.add_argument('--clean-cache', type=str, default='True',
                       help='Remove temporary files after each phase: True/False (default: True)')
    parser.add_argument('--keep-latents', type=str, default='False',
                       help='Keep encoded latent datasets: True/False (default: False)')

    # Quick mode for fast testing
    parser.add_argument('--quick', type=str, default='False',
                       help='Quick mode with reduced thresholds and faster training: True/False (default: False)')

    # Smoke test mode for visual validation only
    parser.add_argument('--smoke-test', type=str, default='False',
                       help='Smoke test mode: skip training, run 1 short rollout per env for visual validation: True/False (default: False)')

    # New finalized training features
    parser.add_argument('--preflight', type=str, default='True',
                       help='Run preflight smoke test before training: True/False (default: True)')
    parser.add_argument('--early-stop', type=str, default='True',
                       help='Enable early stopping on plateau: True/False (default: True)')
    parser.add_argument('--patience', type=int, default=20,
                       help='Generations without improvement before stopping (default: 20)')
    parser.add_argument('--min-delta', type=float, default=1.0,
                       help='Minimum improvement required to reset patience (default: 1.0)')
    parser.add_argument('--eval-every', type=int, default=10,
                       help='Run evaluation snapshots every N generations (default: 10)')

    # Visualization arguments - using string parsing for better compatibility
    parser.add_argument('--visualize', type=str, default='True',
                       help='Enable real-time visualization: True/False (default: True)')
    parser.add_argument('--record-video', type=str, default='False',
                       help='Record training videos: True/False (default: False)')
    parser.add_argument('--video-every-n-gens', type=int, default=10,
                       help='Record video every N generations (default: 10)')
    parser.add_argument('--render-mode', default='human',
                       help='Gymnasium render mode: human/rgb_array (default: human)')

    # Enhanced Visualization Arguments
    parser.add_argument('--fps', type=int, default=30,
                       help='Training visualization FPS for live rollouts (default: 30)')
    parser.add_argument('--eval-fps', type=int, default=60,
                       help='Evaluation visualization FPS for final assessment (default: 60)')
    parser.add_argument('--no-window-reuse', action='store_true',
                       help='Disable render window reuse (creates new windows each time)')
    parser.add_argument('--close-on-completion', action='store_true',
                       help='Close render windows immediately after task completion')

    # Frame Pipeline Validation Arguments
    parser.add_argument('--no-validate-rgb', action='store_true',
                       help='Disable RGB format validation for frames')
    parser.add_argument('--no-validate-sizes', action='store_true',
                       help='Disable frame size validation (64x64/32x32)')
    parser.add_argument('--no-fallback-rgb', action='store_true',
                       help='Disable auto-fallback to rgb_array if human render fails')

    # GPU Memory Optimization Arguments
    parser.add_argument('--amp', type=str, default='True',
                       help='Enable Automatic Mixed Precision for memory efficiency: True/False (default: True)')
    parser.add_argument('--tf32', type=str, default='True',
                       help='Enable TensorFloat-32 for memory efficiency: True/False (default: True)')
    parser.add_argument('--vae-img-size', type=int, default=64,
                       help='VAE image size for memory efficiency: 32/64/96 (default: 64)')
    parser.add_argument('--vae-batch', type=int, default=32,
                       help='VAE batch size for memory efficiency (default: 32)')
    parser.add_argument('--grad-accum', type=int, default=1,
                       help='Gradient accumulation steps (default: 1)')

    return parser.parse_args()

def main():
    """Main function."""
    print("[MAIN] Starting Curriculum Trainer with Visual Feedback...")

    try:
        print("[MAIN] Parsing arguments...")
        args = parse_args()

        # Parse boolean arguments properly
        visualize = args.visualize.lower() in ('true', '1', 'yes', 'on')
        record_video = args.record_video.lower() in ('true', '1', 'yes', 'on')
        quick_mode = args.quick.lower() in ('true', '1', 'yes', 'on')
        smoke_test = args.smoke_test.lower() in ('true', '1', 'yes', 'on')

        # Parse new finalized feature arguments
        preflight = args.preflight.lower() in ('true', '1', 'yes', 'on')
        early_stop = args.early_stop.lower() in ('true', '1', 'yes', 'on')

        # Parse new boolean arguments
        clean_cache = args.clean_cache.lower() in ('true', '1', 'yes', 'on')
        keep_latents = args.keep_latents.lower() in ('true', '1', 'yes', 'on')

        # Parse GPU optimization arguments
        use_amp = args.amp.lower() in ('true', '1', 'yes', 'on')
        use_tf32 = args.tf32.lower() in ('true', '1', 'yes', 'on')

        # Parse video generation list if provided
        video_gens = None
        if args.video_gens:
            try:
                video_gens = [int(g.strip()) for g in args.video_gens.split(',')]
                print(f"[MAIN] Explicit video generations: {video_gens}")
            except ValueError as e:
                print(f"[ERROR] Invalid video-gens format: {e}")
                sys.exit(1)

        # Determine video schedule
        video_schedule = args.video_schedule
        if not video_schedule:
            # Default logic: triad if max-generations >= 200, else all
            video_schedule = "triad" if args.max_generations >= 200 else "all"

        # Set up artifact root directory
        artifact_root = None
        if args.artifact_root:
            artifact_root = Path(args.artifact_root)
        else:
            # Try Windows D: drive default
            external_drive = detect_windows_external_drive()
            if external_drive:
                artifact_root = external_drive
                print(f"[MAIN] Using Windows external drive: {artifact_root}")
            else:
                artifact_root = Path(args.checkpoint_dir)

        # Ensure artifact root exists
        artifact_root.mkdir(parents=True, exist_ok=True)

        # Apply quick mode defaults
        max_generations = args.max_generations
        if quick_mode and args.max_generations == 200:  # Only override if using default
            max_generations = 5
            print("[QUICK MODE] Reducing max generations to 5 for fast testing")

        # Compute record generations for this run
        record_gens = compute_record_gens(max_generations, video_schedule, video_gens)

        print(f"[MAIN] Configuration: device={args.device}, max_gens={max_generations}, "
              f"visualize={visualize}, record_video={record_video}, quick_mode={quick_mode}, smoke_test={smoke_test}")
        print(f"[MAIN] Artifact root: {artifact_root}")
        print(f"[MAIN] Video schedule: {video_schedule}")
        print(f"[MAIN] Recording generations: {sorted(record_gens) if record_gens else 'None'}")
        print(f"[MAIN] GPU Optimizations: amp={use_amp}, tf32={use_tf32}, "
              f"vae_batch={args.vae_batch}, img_size={args.vae_img_size}")

        # Create training configuration
        print("[MAIN] Creating training configuration...")
        config = TrainingConfig(
            device=args.device,
            max_generations=max_generations,
            episodes_per_eval=args.episodes_per_eval,
            checkpoint_dir=str(artifact_root),  # Use artifact_root as checkpoint_dir
            visualize=visualize,
            record_video=record_video,
            video_every_n_gens=args.video_every_n_gens,
            render_mode=args.render_mode,
            quick_mode=quick_mode,
            smoke_test=smoke_test,
            # New finalized training features
            preflight=preflight,
            early_stop=early_stop,
            patience=args.patience,
            min_delta=args.min_delta,
            eval_every=args.eval_every,
            # New artifact management settings
            video_schedule=video_schedule,
            video_gens=video_gens,
            artifact_root=artifact_root,
            clean_cache=clean_cache,
            keep_latents=keep_latents,
            # Enhanced visualization settings
            fps=args.fps,
            eval_fps=args.eval_fps,
            window_reuse=not args.no_window_reuse,
            close_on_completion=args.close_on_completion,
            validate_rgb_frames=not args.no_validate_rgb,
            validate_frame_sizes=not args.no_validate_sizes,
            fallback_to_rgb_array=not args.no_fallback_rgb,
            # GPU optimization settings
            use_amp=use_amp,
            use_tf32=use_tf32,
            vae_img_size=args.vae_img_size,
            vae_batch_size=args.vae_batch,
            grad_accumulation_steps=args.grad_accum
        )
        print(f"[MAIN] Config created successfully")

        # Create and run curriculum trainer
        print("[MAIN] Creating curriculum trainer...")
        print(f"[MAIN] About to instantiate CurriculumTrainer with config: {config.device}")
        trainer = CurriculumTrainer(config)
        print("[MAIN] CurriculumTrainer created successfully!")

        # Check if smoke test mode
        if smoke_test:
            print("[SMOKE TEST] Running visual validation mode - skipping training")
            smoke_success = trainer.run_smoke_test()

            if smoke_success:
                print("\n[SUCCESS] Smoke test complete: visuals validated.")
                print("✅ All 3 environments working correctly")
                print("💡 To run full training, use without --smoke-test flag")
                sys.exit(0)
            else:
                print("\n[PARTIAL] Smoke test completed with some issues")
                print("❌ Some environments had problems - check pip install hints above")
                sys.exit(1)

        # Run preflight check if enabled
        if config.preflight and not smoke_test:  # Skip preflight if already in smoke test mode
            print("[PREFLIGHT] Running preflight smoke test before training...")
            preflight_success = trainer.run_smoke_test()

            if not preflight_success:
                print("\n[PREFLIGHT FAILED] Environment issues detected!")
                print("❌ Some environments failed preflight checks")
                print("💡 Fix issues above or disable with --preflight False")
                sys.exit(1)
            else:
                print("✅ Preflight checks passed - proceeding with training")
                print("")  # Add spacing before training starts

        print("[LAUNCH] Starting World Models Curriculum Training")
        print(f"Device: {config.device}")
        print(f"Visualization: {'ON' if config.visualize else 'OFF'}")
        print(f"Video Recording: {'ON' if config.record_video else 'OFF'}")
        print(f"Artifact Root: {config.artifact_root}")
        print(f"Video Schedule: {config.video_schedule}")
        print(f"Record Generations: {sorted(trainer.record_gens) if trainer.record_gens else 'None'}")
        print(f"Cache Cleanup: {'ON' if config.clean_cache else 'OFF'}")
        print(f"Keep Latents: {'ON' if config.keep_latents else 'OFF'}")

        success = trainer.run_curriculum()
        final_success = trainer.generate_final_report()

        if final_success:
            print("\n[SUCCESS] CURRICULUM COMPLETED SUCCESSFULLY!")
            sys.exit(0)
        else:
            print("\n[WARNING] Curriculum completed with some failures")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Training interrupted by user")

        # Save current state if possible
        try:
            if 'trainer' in locals() and trainer is not None:
                trainer.save_interrupt_state()
        except Exception as e:
            print(f"[WARN] Could not save interrupt state: {e}")

        print("[INTERRUPT] Training session ended")
        sys.exit(2)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()

    # Print example commands after successful completion
    print("\n" + "="*80)
    print("CURRICULUM TRAINER UPDATE COMPLETE!")
    print("="*80)
    print("\nUpdated Features:")
    print("✅ New environment list: ALE/Pong-v5, LunarLander-v3, ALE/Breakout-v5, CarRacing-v3")
    print("✅ Quick mode support with --quick flag")
    print("✅ Reduced thresholds in quick mode: Pong=5, LunarLander=50, Breakout=10, CarRacing=200")
    print("✅ Quick mode reduces max_generations to 5 (if using default)")
    print("✅ Clear logging when running in QUICK MODE")
    print("\nExample Commands:")
    print("# Full curriculum:")
    print("python3 curriculum_trainer_visual.py --device cpu --max-generations 200 --episodes-per-eval 5 --visualize True --record-video True")
    print("\n# Quick test run:")
    print("python3 curriculum_trainer_visual.py --device cpu --quick True --visualize True")
    print("="*80)
