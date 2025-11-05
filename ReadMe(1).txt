Autonomous Drone Landing with UR10 Arm – README

This project involves training a reinforcement learning (RL) agent to perform autonomous drone landings on a UR10 robotic arm using NVIDIA Isaac Sim and Isaac Lab.

== IMPORTANT == 

The following Isaac Lab files have been modified and must be replaced with the ones provided in the project ZIP.
Do ==NOT== use the official versions from the Isaac Lab GitHub repository for the following files.

# 1. scripts\reinforcement_learning\skrl\play.py
#    → Custom logging setup for TensorBoard

# 2. source\isaaclab_assets\isaaclab_assets\robots\quadcopter.py
#    → Updated quadcopter configuration for scale and tuning

# 3. Add these following folders found in my repo in your direct tasks folder "...\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct" 
#    → arm_drone_communication 
#    → dronemultiagent_marl
# Notes: You need to download them and add them to the Isaac Lab repo that you should have after following the official installation guide. ##These are the modified files needed to run the project.##

== SETUP INSTRUCTIONS ==

1. Install Isaac Sim & Isaac Lab by following the official installation instructions. (I used the Pip Installation)
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html <-- the link to the Guide

1.5 Update the required folders as specified above.

2. Activate your Conda environment:

    conda activate env_isaaclab

   (Replace with the actual environment name if different.)

3. Navigate to the Isaac Lab folder:

    cd "C:\Users\UMRobotics\Desktop\IsaacLab"
   
   (Update the path accordingly to where Isaac Lab is located on your machine.)

== TRAINING THE AGENT ==

To train the drone and robot agent in a single agent, run:

    python scripts\reinforcement_learning\skrl\train.py --task Isaac-Arm-Drone-Communication-Direct-v0 --num_env 10000 --headless

In a multi agent env, you want to use the following running command instead: 
    python scripts\reinforcement_learning\skrl\train.py --task Isaac-Dronemultiagent-Marl-Direct-v0 --algorithm MAPPO --num_env 10000 --headless

Notes:
- The --headless flag disables the GUI, drastically improving performance.
- Running 10,000 environments with GUI enabled is not feasible even on high-end machines.

== RUNNING THE TRAINED AGENT ==

Once training is complete, run the single agent with:

    python scripts\reinforcement_learning\skrl\play.py --task Isaac-Arm-Drone-Communication-Direct-v0 --num_env 10

And run the multi agent with the following:

    python scripts\reinforcement_learning\skrl\play.py --task Isaac-Dronemultiagent-Marl-Direct-v0 --algorithm MAPPO --num_env 10

Notes: "--num_env 10" is a simulation parameter where 10 is the number of parallel environments. Increasing this number will reduce your performance but increase the number of agents you have in your GUI

Add --headless if you want to run the experiment without GUI.

== VISUALIZATION WITH TENSORBOARD ==

If not already installed:

    pip install tensorboard

Then run:

    tensorboard --logdir "C:\Users\Stefan\Desktop\Thesis folder\IsaacNvidia\Drones\logs\skrl\Arm_Drone_Communication\play_logs"

	^^ This command will then show you how your program performed during the playing of the agent 

    tensorboard --logdir "C:\Users\UMRobotics\Desktop\IsaacLab\logs\skrl\Arm_Drone_Communication\2025-09-18_14-25-43_ppo_torch"
    	
	^^ This command will then show you how your agent trained and how the rewards changed as the training went on. 

Make sure to replace the path with your actual log directory (single agent/ multi agent.

Tip: Delete the `play_logs` folder before running a new experiment to avoid overlapping data in TensorBoard.

== WIND CONFIGURATION ==

Wind is configured in:

    source\isaaclab_tasks\isaaclab_tasks\direct\arm_drone_communication\arm_drone_communication_env_cfg.py

Choose one of the following wind configurations by uncommenting the appropriate lines:

    # No Wind
    # lower_wind_scale = 0.0
    # upper_wind_scale = 0.0

    # Light Wind
    # lower_wind_scale = 0.1
    # upper_wind_scale = 0.2

    # Medium Wind
    # lower_wind_scale = 0.3
    # upper_wind_scale = 0.45

    # High Wind
    # lower_wind_scale = 0.5
    # upper_wind_scale = 0.6

    # Varying Wind (recommended)
    lower_wind_scale = 0.1
    upper_wind_scale = 0.6

These settings allow testing performance under varying difficulty levels.

== NOTE == 

The wind scale values (lower_wind_scale and upper_wind_scale) define the base wind force intensity in the simulation.
These values are normalized within the Isaac Lab environment. When the drone is scaled to ~3× its real mass (~28g × 3), 
the following approximate real-world equivalents (in km/h) apply:

# - 0.0 – 0.0   → No wind (0 km/h)
# - 0.1 – 0.2   → Light breeze (~7–14 km/h)
# - 0.3 – 0.45  → Moderate wind (~20–32 km/h)
# - 0.5 – 0.6   → Strong wind (~35–45 km/h)
# - 0.1 – 0.6   → Variable wind (random between ~7–45 km/h)

These estimates represent the **base wind** only. In addition, stochastic wind gusts are applied with extra random force 
in the range of uniform_(0.1, 0.3), which temporarily increases the wind strength beyond the base level. This adds 
realism and simulates unpredictable offshore gust conditions.


== WIND IN OBSERVATIONS ==

Still in `arm_drone_communication_env_cfg.py`, configure the observation space:

    observation_space = 27  # (12 drone + 12 arm + 3 wind)

If you do not want wind as input to the policy:

    # observation_space = 24

However, you must also comment out this line in:

    source\isaaclab_tasks\isaaclab_tasks\direct\arm_drone_communication\arm_drone_communication_env.py

Specifically:

    # wind_forces, #(3,)

This is part of:

    obs = torch.cat(
        [
            self._DroneRobot.data.root_lin_vel_b,
            self._DroneRobot.data.root_ang_vel_b,
            self._DroneRobot.data.projected_gravity_b,
            desired_pos_b,
            joint_pos,
            joint_vel,
            wind_forces,  # Comment this line if removing wind
        ],
        dim=-1,
    )
    observations = {"policy": obs}

IMPORTANT: If you change the observation space, retrain the model from scratch.

== USEFUL COMMANDS SUMMARY ==

- conda activate env_isaaclab       → activate Isaac Lab environment
- cd <path>                         → navigate to your project folder
- train.py ... --headless           → trains the agent without GUI
- play.py ...                       → runs the trained agent
- tensorboard --logdir <path>       → launches TensorBoard for visualizing metrics
- pip install tensorboard           → installs TensorBoard if not already present

== FINAL NOTES ==

- Use --headless mode for performance.
- Always retrain the model if observation dimensions change.
- Clean your logs for fresh TensorBoard runs.
- Vary the wind configuration to test robustness.
- Go crazy and do drone stuff.

Good luck and enjoy experimenting with autonomous drone landings!
