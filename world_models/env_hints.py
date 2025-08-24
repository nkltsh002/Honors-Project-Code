"""
Environment installation hints for World Models project.
Provides helpful installation commands when environment creation fails.
"""

def install_hint(env_id: str) -> str:
    """
    Return installation hint for a given environment ID.

    Args:
        env_id: Environment identifier (e.g., "ALE/Pong-v5", "LunarLander-v3")

    Returns:
        str: Installation command hint
    """
    env_id = env_id.lower()

    # ALE/Atari environments
    if env_id.startswith("ale/") or "atari" in env_id:
        return 'pip install "gymnasium[atari,accept-roms]" ale-py autorom && AutoROM --accept-license'

    # Box2D environments (LunarLander, CarRacing, BipedalWalker)
    elif any(box2d_env in env_id for box2d_env in ["lunarlander", "carracing", "bipedal"]):
        return 'pip install swig && pip install "gymnasium[box2d]"'

    # MuJoCo environments
    elif "mujoco" in env_id or any(mj_env in env_id for mj_env in ["ant", "humanoid", "walker", "hopper"]):
        return 'pip install "gymnasium[mujoco]"'

    # Classic control (should always work with base gymnasium)
    elif any(classic_env in env_id for classic_env in ["cartpole", "acrobot", "mountaincar", "pendulum"]):
        return 'pip install gymnasium'

    # Generic fallback
    else:
        return f'pip install gymnasium  # For {env_id}, check gymnasium documentation for specific requirements'


def print_env_error_hint(env_id: str, error: Exception):
    """
    Print a helpful error message with installation hint when environment creation fails.

    Args:
        env_id: Environment identifier that failed
        error: The exception that was raised
    """
    print(f"❌ Failed to create environment '{env_id}'")
    print(f"   Error: {error}")
    print(f"   💡 Try: {install_hint(env_id)}")
    print()
