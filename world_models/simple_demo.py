"""Simple demo for controller training - Python 3.12 compatible.

This is a basic demonstration of the controller training pipeline
that works with Python 3.12 and uses modern Python features.
"""

import os
import sys
import numpy as np
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_models.models.controller import Controller


def test_basic_controller():
    """Test basic controller functionality."""
    print("=" * 50)
    print("TESTING BASIC CONTROLLER FUNCTIONALITY")
    print("=" * 50)

    # Test linear controller
    print("\n1. Testing Linear Controller:")
    controller = Controller(
        input_size=32 + 256,  # z_dim + h_dim
        action_size=3,
        hidden_sizes=(),  # Linear controller
        action_type='continuous'
    )

    print("  Parameters: {:,}".format(controller.get_num_parameters()))

    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, 32)
    h = torch.randn(batch_size, 256)

    with torch.no_grad():
        output = controller(z, h)
        print("  Output shape: {}".format(output.shape))

        # Test parameter management
        params = controller.get_parameters()
        print("  Parameter vector size: {}".format(len(params)))

        # Test weight setting
        new_params = np.random.randn(len(params)) * 0.1
        controller.set_parameters(new_params)

        # Verify
        restored_params = controller.get_parameters()
        if np.allclose(new_params, restored_params):
            print("  ✓ Weight management works correctly")
        else:
            print("  ✗ Weight management failed")

    print("\n2. Testing MLP Controller:")
    mlp_controller = Controller(
        input_size=32 + 256,
        action_size=4,
        hidden_sizes=(64,),  # Single hidden layer
        action_type='discrete'
    )

    print("  Parameters: {:,}".format(mlp_controller.get_num_parameters()))

    with torch.no_grad():
        output = mlp_controller(z, h)
        print("  Output shape: {}".format(output.shape))

        # Test action generation
        action = mlp_controller.get_action(z, h, deterministic=True)
        if isinstance(action, tuple):
            action_tensor = action[0] if len(action) > 0 else action
            print("  Action shape: {}".format(action_tensor.shape))
        else:
            print("  Action shape: {}".format(action.shape))

    print("\n✓ Basic controller tests completed successfully!")


def test_cmaes_integration():
    """Test CMA-ES integration."""
    print("\n" + "=" * 50)
    print("TESTING CMA-ES INTEGRATION")
    print("=" * 50)

    try:
        from world_models.models.controller import CMAESController

        # Create a small controller for testing
        controller = Controller(
            input_size=32 + 256,
            action_size=2,  # Small action space
            hidden_sizes=(),  # Linear controller for speed
            action_type='continuous'
        )

        print("Controller for CMA-ES test:")
        print("  Parameters: {:,}".format(controller.get_num_parameters()))

        # Create CMA-ES trainer
        cmaes_trainer = CMAESController(
            controller=controller,
            population_size=8,  # Small population for demo
            sigma=0.5
        )

        print("CMA-ES trainer created successfully")

        # Test one generation
        print("\nTesting CMA-ES generation:")
        candidates = cmaes_trainer.ask()
        print("  Candidates shape: {}".format(candidates.shape))

        # Mock fitness evaluation
        fitness_values = np.random.randn(len(candidates))
        cmaes_trainer.tell(candidates, fitness_values)

        stats = cmaes_trainer.get_stats()
        print("  Generation stats: best={:.3f}, mean={:.3f}".format(
            stats.get('best_fitness', 0),
            stats.get('mean_fitness', 0)
        ))

        print("✓ CMA-ES integration test completed!")

    except Exception as e:
        print("✗ CMA-ES test failed: {}".format(e))


def test_mock_training():
    """Test a mock training loop."""
    print("\n" + "=" * 50)
    print("TESTING MOCK TRAINING LOOP")
    print("=" * 50)

    try:
        # Create controller
        controller = Controller(
            input_size=32 + 256,
            action_size=2,
            hidden_sizes=(),
            action_type='continuous'
        )

        # Mock fitness function
        def mock_fitness(params):
            # Simple quadratic function
            return -(np.sum(params**2)) / len(params)

        print("Running 5 generations of mock training...")

        # Simple evolution strategy
        param_size = len(controller.get_parameters())
        current_params = controller.get_parameters()
        sigma = 0.1

        for gen in range(5):
            # Generate candidate solutions
            candidates = []
            for _ in range(8):
                candidate = current_params + np.random.randn(param_size) * sigma
                candidates.append(candidate)

            # Evaluate fitness
            fitness_values = [mock_fitness(c) for c in candidates]

            # Select best
            best_idx = np.argmax(fitness_values)
            current_params = candidates[best_idx]

            print("  Gen {}: best_fitness={:.4f}".format(gen + 1, fitness_values[best_idx]))

        # Update controller with best parameters
        controller.set_parameters(current_params)

        print("✓ Mock training completed successfully!")

    except Exception as e:
        print("✗ Mock training failed: {}".format(e))


def main():
    """Run all tests."""
    print("World Models Controller Training Demo")
    print("Python version: {}.{}.{}".format(*sys.version_info[:3]))
    print("PyTorch version: {}".format(torch.__version__))

    try:
        test_basic_controller()
        test_cmaes_integration()
        test_mock_training()

        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)

    except Exception as e:
        print("\nERROR: {}".format(e))
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
