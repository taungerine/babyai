from multiprocessing import Process, Pipe
import gym
import numpy as np
import math

def get_global(env):
    # get global view
    grid = env.grid
    
    # position agent
    x, y = env.start_pos
    
    # rotate to match agent's orientation
    for i in range(env.agent_dir + 1):
        # rotate grid
        grid = grid.rotate_left()
        
        # rotate position of agent
        x_new = y
        y_new = grid.height - 1 - x
        x     = x_new
        y     = y_new
    
    # encode image for model
    image = grid.encode()

    # indicate position of agent
    image[x, y, 0] = 10
    
    return image

def get_local(obs):
    # get local view
    return obs['image'][3:4, 5:7, :]

def worker(conn, env0, env1):
    while True:
        cmd, scouting, action = conn.recv()
        if cmd == "step":
            if scouting:
                obs, reward, done, info = env0.step(action)
                obs   = env1.reset()
                globs = obs.copy()
                obs['image']   = get_local(obs)
                globs['image'] = get_global(env1)
                done = True
                reward *= 0
            else:
                obs, reward, done, info = env1.step(action)
                if done:
                    obs   = env0.reset()
                    globs = obs.copy()
                    obs['image']   = get_local(obs)
                    globs['image'] = get_global(env0)
                else:
                    globs = obs.copy()
                    obs['image']   = get_local(obs)
                    globs['image'] = get_global(env1)
            conn.send((globs, obs, reward, done, info))
        elif cmd == "reset":
            if scouting:
                env1.reset()
            obs   = env0.reset()
            globs = obs.copy()
            obs['image']   = get_local(obs)
            globs['image'] = get_global(env0)
            conn.send((globs, obs))
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs0, envs1):
        assert len(envs0) >= 1, "No environment given."

        self.envs0 = envs0
        self.envs1 = envs1
        self.observation_space = self.envs0[0].observation_space
        self.action_space = self.envs0[0].action_space

        self.locals = []
        for env0, env1 in zip(self.envs0[1:], self.envs1[1:]):
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env0, env1))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self, scouting):
        for local, scouting_ in zip(self.locals, scouting[1:]):
            local.send(("reset", scouting_, None))
        if scouting[0]:
            self.envs1[0].reset()
        obs   = self.envs0[0].reset()
        globs = obs.copy()
        obs['image']   = get_local(obs)
        globs['image'] = get_global(self.envs0[0])
        results = zip(*[(globs, obs)] + [local.recv() for local in self.locals])
        return results

    def step(self, actions, scouting):
        for local, action, scouting_ in zip(self.locals, actions[1:], scouting[1:]):
            local.send(("step", scouting_, action))
        
        if scouting[0]:
            obs, reward, done, info = self.envs0[0].step(actions[0])
            obs   = self.envs1[0].reset()
            globs = obs.copy()
            obs['image']   = get_local(obs)
            globs['image'] = get_global(self.envs1[0])
            done = True
            reward *= 0
        else:
            obs, reward, done, info = self.envs1[0].step(actions[0])
            if done:
                obs   = self.envs0[0].reset()
                globs = obs.copy()
                obs['image']   = get_local(obs)
                globs['image'] = get_global(self.envs0[0])
            else:
                globs = obs.copy()
                obs['image']   = get_local(obs)
                globs['image'] = get_global(self.envs1[0])

        results = zip(*[(globs, obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError
