如果debug文件，需要source isaac sim 环境，setup isaac-sim 环境变量
```bash
source /home/bjae/isaac-sim/setup_conda_env.sh
```

create ae_robot in isaaclab
```bash
./isaaclab.sh -p scripts/tutorials/01_assets/run_custom_robot.py
```

run the scene
```bash
./isaaclab.sh -p scripts/tutorials/02_scene/create_ae_scene.py --num_envs 16
```

run the rl training 
```bash
./isaaclab.sh -p scripts/tutorials/03_envs/run_ae_rl_env.py --num_envs 16
```


add the camera, 支持实时显示相机视角
```bash
./isaaclab.sh -p scripts/tutorials/04_sensors/run_ae_robot_with_camera.py --enable_cameras
```


controller
```bash
./isaaclab.sh -p scripts/tutorials/05_controllers/run_air_robot_diff_ik.py
./isaaclab.sh -p scripts/tutorials/05_controllers/run_air_robot_osc.py
```
