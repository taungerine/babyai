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

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            globs = obs.copy()
            obs['image']   = get_local(obs)
            globs['image'] = get_global(env)
            step_count = env.step_count
            conn.send((globs, obs, reward, done, step_count, info))
        elif cmd == "reset":
            obs = env.reset()
            globs = obs.copy()
            obs['image']   = get_local(obs)
            globs['image'] = get_global(env)
            conn.send((globs, obs))
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        obs = self.envs[0].reset()
        globs = obs.copy()
        obs['image']   = get_local(obs)
        globs['image'] = get_global(self.envs[0])
        results = zip(*[(globs, obs)] + [local.recv() for local in self.locals])
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        globs = obs.copy()
        obs['image']   = get_local(obs)
        globs['image'] = get_global(self.envs[0])
        step_count = self.envs[0].step_count
        results = zip(*[(globs, obs, reward, done, step_count, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError
