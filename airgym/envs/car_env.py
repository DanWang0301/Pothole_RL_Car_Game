import airsim
import numpy as np
import math
import time

from gym import spaces
from airgym.envs.airsim_env import AirSimEnv
from PIL import Image


class AirSimCarEnv(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.action_space = spaces.Discrete(6)
        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.Scene, False, False)
        self.car_controls = airsim.CarControls()
        self.car_state = None
        self.reward = 5
        self.hole_total = 0
        self.reward_time1 = time.time()
        self.num = 0
        self.car_step = 0
        self.total_step = 0

    def _setup_car(self):
        self.car.simRunConsoleCommand('open Newworld')
        time.sleep(2)
        self.car = airsim.CarClient()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)

    def __del__(self):
        self.car.simRunConsoleCommand('open Newworld')
        time.sleep(2)
        self.car = airsim.CarClient()

    def _do_action(self, action):
        self.car_state = self.car.getCarState()

        self.car_controls.throttle = 1
        self.car_controls.brake = 0

        if self.car_state.speed > 5:
            self.car_controls.throttle = 0
        else:
            self.car_controls.throttle = 1

        if action == 0:
            self.car_controls.throttle = 0
        elif action == 1:
            self.car_controls.steering = 0
        elif action == 2:
            self.car_controls.steering = 0.5
        elif action == 3:
            self.car_controls.steering = -0.5
        # elif action == 4:
        #     self.car_controls.steering = 0.25
        # elif action == 5:
        #     self.car_controls.steering = -0.25

        self.car.setCarControls(self.car_controls)
        self.car_step += 1
        self.total_step += 1
        print("Total step: ", self.car_step)
        time.sleep(0.1)

    def transform_obs(self, response):

        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (response.height, response.width, 3))
        # print(img2d.shape)

        try:

            image = Image.fromarray(img2d).convert("RGB")
            image_2 = image.resize((120, 160))
            im_final = np.array(image_2.convert("L"))
        except ValueError:
            print("PIL_ERROR!!!")

            error1 = self.num
            with open('car_error.csv', 'a+') as lst:
                lst.write(str(error1)+" Error!!! "+'\n')
            time.sleep(2)

            while(img2d.shape != (960, 1280, 3)):

                response_backup = self.car.simGetImages([self.image_request])
                img1d = np.fromstring(
                    response_backup[0].image_data_uint8, dtype=np.uint8)
                img2d = np.reshape(
                    img1d, (response_backup[0].height, response_backup[0].width, 3))
                # print("back_up:",img2d.shape)
                image = Image.fromarray(img2d).convert("RGB")
                image_2 = image.resize((120, 160))
                im_final = np.array(image_2.convert("L"))

        return im_final.reshape([120, 160, 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided

        return image

    def _compute_reward(self):

        reward = self.reward
        totalhole = self.hole_total
        reward_time1 = self.reward_time1
        reward_time2 = time.time()

        car_state_1 = self.car_state
        pd_1 = car_state_1.kinematics_estimated.position
        car_pt_1 = np.array([pd_1.x_val, pd_1.y_val, pd_1.z_val])
        # print('z_val_1: ',car_pt_1[2])

        time.sleep(0.1)

        car_state_2 = self.car.getCarState()
        pd_2 = car_state_2.kinematics_estimated.position
        car_pt_2 = np.array([pd_2.x_val, pd_2.y_val, pd_2.z_val])
        # print('z_val_2: ',car_pt_2[2])

        ans = round(float(car_pt_1[2] - car_pt_2[2]), 4)
        # print('ans: ',ans)

        if ans > 0.002:
            # print("Hole:",car_pt_1[2])
            totalhole += 1
            reward -= 1
        elif ans < -0.002:
            # print("Hole:",car_pt_2[2])
            totalhole += 1
            reward -= 1
        elif round(float(reward_time2 - reward_time1), 4) >= 5:
            reward += 1
            self.reward_time1 = reward_time1

        print("Total_hole:", totalhole)

        done = 0
        self.reward = reward
        self.hole_total = totalhole

        if reward < 1:
            done = 1
        if self.car_controls.brake == 0:
            if self.car_state.speed < 1:
                done = 1
                print('Too Slow!!')
        if self.state["collision"]:
            done = 1

        if done == 1:
            self.num += 1

            reward = reward - totalhole

            distance = round(float(car_pt_1[0]), 4)
            with open('car_distance.csv', 'a+') as lst:
                lst.write(str(distance)+'\n')
            print("Distance:", distance)

            with open('car_reward.csv', 'a+') as lst:
                lst.write(str(reward)+'\n')
            print("Reward:", reward)

            with open('car_step.csv', 'a+') as lst:
                lst.write(str(self.car_step)+'\n')
            print("Step:", self.car_step)

            with open('car_hole.csv', 'a+') as lst:
                lst.write(str(totalhole)+'\n')

        return done, reward, self.num

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        done, reward, num = self._compute_reward()
        print("Reward:", reward)
        print("Times:", num)
        print("---------------------------------------")

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()

        self.reward = 5
        self.hole_total = 0
        self.car_step = 0
        self._do_action(1)

        # time.sleep(1)

        return self._get_obs()
