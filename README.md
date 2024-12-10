# LlamaSwarm

A simulator for multi-agent reinforcement learning focused on collaborative and competitive agent behaviors.

## Features

- Modular architecture for easily implementing new environments and algorithms
- Built-in implementations of popular multi-agent RL algorithms (MAPPO, MADDPG, etc.)
- Customizable cooperative and competitive environments
- Tools for visualization and analysis of agent behaviors
- Utilities for experiment management and reproducibility

## Installation

```bash
# Clone the repository
git clone https://github.com/llamaswarm/llamaswarm.git
cd llamaswarm

# Install the package and its dependencies
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from llamaswarm.environments.cooperative import CooperativeNavigation
from llamaswarm.algorithms.policy_gradient import MAPPO
from llamaswarm.core import Trainer

# Create environment
env = CooperativeNavigation(num_agents=3, grid_size=10)

# Create algorithm
algorithm = MAPPO(
    state_dim=env.observation_space[0].shape[0],
    action_dim=env.action_space[0].n,
    num_agents=env.num_agents,
    hidden_dim=128,
    learning_rate=0.001
)

# Create trainer
trainer = Trainer(
    env=env,
    algorithm=algorithm,
    max_episodes=1000,
    max_steps=100,
    eval_interval=10
)

# Train agents
trainer.train()

# Evaluate trained agents
trainer.evaluate(num_episodes=10, render=True)
```

## Examples

See the `examples` directory for more detailed examples:

- `simple_navigation.py`: Basic cooperative navigation
- `predator_prey.py`: Competitive environment with predator and prey agents
- `grid_world.py`: Grid-based environment with obstacles and goals

## Documentation

For detailed documentation, visit [our documentation site](https://llamasearch.ai

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use LlamaSwarm in your research, please cite:

```
@software{llamaswarm2023,
  author = {LlamaSwarm Team},
  title = {LlamaSwarm: A Simulator for Multi-Agent Reinforcement Learning},
  year = {2023},
  url = {https://github.com/llamaswarm/llamaswarm}
}
``` 
# Updated in commit 1 - 2025-04-04 17:32:45

# Updated in commit 9 - 2025-04-04 17:32:46

# Updated in commit 17 - 2025-04-04 17:32:46

# Updated in commit 25 - 2025-04-04 17:32:47

# Updated in commit 1 - 2025-04-05 14:36:12

# Updated in commit 9 - 2025-04-05 14:36:13

# Updated in commit 17 - 2025-04-05 14:36:13

# Updated in commit 25 - 2025-04-05 14:36:13

# Updated in commit 1 - 2025-04-05 15:22:42
