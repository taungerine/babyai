from multiprocessing import Process, Pipe
import gym
import numpy as np
import math

def get_global(env, local_obs):
    # get global view
    grid = env.grid
    
    # position agent
    x, y = env.agent_pos
    
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

    # overlap global with local observation, i.e., include carried objects
    image[x, y, :] = local_obs['image'][3, 6, :]

    # indicate position of agent
    image[x, y, 0] += 10
    
    return image

def get_local(obs):
    # get local view
    return obs['image'][3:4, 5:7, :]

def adapt(results, scouting):
    results = results.copy()
    i = 0
    for scouting_ in scouting:
        if scouting_:
            results[i]    = list(results[i])
            results[i][2] = 0.0
            results[i][3] = True
            results[i]    = tuple(results[i])
        i += 1
    return results

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if 64 <= env.step_count:
                done = True
            if done:
                obs = env.reset()
            globs = obs.copy()
            globs['image'] = get_global(env, obs)
            obs['image']   = get_local(obs)
            step_count = env.step_count
            conn.send((globs, obs, reward, done, step_count))
        elif cmd == "reset":
            obs = env.reset()
            globs = obs.copy()
            globs['image'] = get_global(env, obs)
            obs['image']   = get_local(obs)
            conn.send((globs, obs, 0.0, False, 0))
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

    def reset(self, scouting):
        for local in self.locals:
            local.send(("reset", None))
        obs = self.envs[0].reset()
        globs = obs.copy()
        globs['image'] = get_global(self.envs[0], obs)
        obs['image']   = get_local(obs)
        self.results = [(globs, obs, 0.0, False, 0)] + [local.recv() for local in self.locals]
        return zip(*adapt(self.results, scouting))

    def step(self, actions, scouting):
        for local, action, scouting_ in zip(self.locals, actions[1:], scouting[1:]):
            if not scouting_:
                local.send(("step", action))
        if not scouting[0]:
            obs, reward, done, info = self.envs[0].step(actions[0])
            if 64 <= self.envs[0].step_count:
                done = True
            if done:
                obs = self.envs[0].reset()
            globs = obs.copy()
            globs['image'] = get_global(self.envs[0], obs)
            obs['image']   = get_local(obs)
            step_count = self.envs[0].step_count
            self.results[0] = (globs, obs, reward, done, step_count)
        for i in range(len(self.envs)-1):
            if not scouting[i+1]:
                self.results[i+1] = self.locals[i].recv()
        return zip(*adapt(self.results, scouting))

    def render(self):
        raise NotImplementedError
