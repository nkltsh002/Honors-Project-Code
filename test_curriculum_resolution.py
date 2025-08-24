#!/usr/bin/env python3
"""
Test the curriculum resolution function
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'world_models'))

from curriculum_trainer_visual import resolve_curriculum

print("🔍 Testing curriculum resolution...")
curriculum = resolve_curriculum()
print("\n📋 Selected curriculum:")
for i, (env_id, threshold) in enumerate(curriculum, 1):
    print(f"  {i}. {env_id}: {threshold}")
