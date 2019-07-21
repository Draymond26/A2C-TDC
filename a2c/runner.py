import numpy as np
from baselines.a2c.utils import discount_with_dones
from baselines.common.runners import AbstractEnvRunner
import cv2

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences

    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, classifier, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
        self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
        self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype
        self.classifier = classifier
        self.last_imgs = np.load('/home/liuguangze/Experiments/MR-tdc/generate-shuffled-dataset/last_imgs.npz')['last_imgs']
        self.tdc_pre = np.ones([32])*3
        self.current_states = np.zeros([32])
        self.reset_flag = False
        self.lives_pre = np.ones([32])*6

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [],[],[],[],[]
        mb_states = self.states

        for n in range(self.nsteps):
            # Given observations, take action and value (V(s))
            # We already have self.obs because Runner superclass run self.obs[:] = env.reset() on init
            actions, values, states, _ = self.model.step(self.obs, S=self.states, M=self.dones)

            # Append the experiences
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Take actions in env and look the results
            obs, rewards, dones, infos = self.env.step(actions)
            
            obs_shape = obs.shape
            ob = np.split(obs, obs_shape[-1], axis=-1)[-1]        #[batchsize, 84, 84, 1]
            ps = []
            for i in range(32):
                ps.append(self.last_imgs[int(self.current_states[i])])
            self.ps = np.array(ps)
            self.ps = np.reshape(self.ps, [32,84,84,1])
            #print(ob.shape)
            #print(self.ps.shape)
            reshaped_obs = np.concatenate([ob, self.ps], axis=1)
            
            tdc = self.classifier.predict(reshaped_obs)
            print("current state-tdc: ", end='')
            for i in range(32):
                print(str(int(self.current_states[i])) + '-' + str(tdc[i]) + ' ', end='')
            print('\n')
            for i in range(len(tdc)):
                if(self.tdc_pre[i]-tdc[i]==1):
                    rewards[i] += 10
                    if(self.tdc_pre[i]==1 and tdc[i]==0):
                        if self.current_states[i]==6:
                            rewards[i] += 20
                            self.reset_flag = True
                            print("----------Congratulations!!----------")
                        else:
                            self.current_states[i] += 1
                            rewards[i] += 10
                            tdc[i] = 3
                elif(self.tdc_pre[i]-tdc[i]==-1):
                    rewards[i] -= 10
                #elif(self.tdc_pre[i]==tdc[i]):
                #    rewards[i] -= 1
                if(infos[i]['ale.lives']<self.lives_pre[i]):
                    rewards[i] -= 10
                    self.reset_flag = True
                self.lives_pre[i] = infos[i]['ale.lives']
            self.tdc_pre = tdc
            print('rewards: ' + str(rewards))
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
                    self.reset_flag = True
            self.obs = obs
            mb_rewards.append(rewards)
            
            if self.reset_flag:
                print("----------reset envs----------")
                self.env.reset()
                self.tdc_pre = np.ones([32])*3
                self.current_states = np.zeros([32])
                self.lives_pre = np.ones([32])*6
                self.reset_flag = False
            
        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.ob_dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=self.model.train_model.action.dtype.name).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]


        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_actions = mb_actions.reshape(self.batch_action_shape)

        mb_rewards = mb_rewards.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values
