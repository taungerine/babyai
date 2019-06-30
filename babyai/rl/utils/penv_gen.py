from multiprocessing import Process, Pipe
import gym
import numpy as np
import math

def get_global(env):
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

def get_agent_loc(env):
    # get global view
    grid = env.grid
    
    # position agent
    x, y = env.agent_pos
    
    # rotate to match agent's orientation
    for i in range(env.agent_dir + 1):
        # rotate position of agent
        x_new = y
        y_new = grid.height - 1 - x
        x     = x_new
        y     = y_new
    
    agent_loc_x = x
    agent_loc_y = y
    
    return agent_loc_x, agent_loc_y

def get_object_loc(env):
    # get global view
    grid = env.grid
    
    # object types
    object_loc = (4 < grid.encode()[:, :, 0]).nonzero()
    
    x = object_loc[0].item()
    y = object_loc[1].item()
    
    # rotate to match agent's orientation
    for i in range(env.agent_dir + 1):
        # rotate position of agent
        x_new = y
        y_new = grid.height - 1 - x
        x     = x_new
        y     = y_new
    
    object_loc_x = x
    object_loc_y = y

    return object_loc_x, object_loc_y

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9
}

def get_goal(env):
    goal_type  = OBJECT_TO_IDX[env.instrs.desc.type]
    goal_color = COLOR_TO_IDX[env.instrs.desc.color]
    
    return goal_type, goal_color

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
            agent_loc_x, agent_loc_y = get_agent_loc(env)
            goal_type, goal_color = get_goal(env)
            conn.send((globs, obs, reward, done, step_count, agent_loc_x, agent_loc_y, goal_type, goal_color))
        elif cmd == "reset":
            obs = env.reset()
            globs = obs.copy()
            obs['image']   = get_local(obs)
            globs['image'] = get_global(env)
            agent_loc_x, agent_loc_y = get_agent_loc(env)
            goal_type, goal_color = get_goal(env)
            conn.send((globs, obs, 0.0, False, 0, agent_loc_x, agent_loc_y, goal_type, goal_color))
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
        obs['image']   = get_local(obs)
        globs['image'] = get_global(self.envs[0])
        agent_loc_x, agent_loc_y = get_agent_loc(self.envs[0])
        goal_type, goal_color = get_goal(self.envs[0])
        self.results = [(globs, obs, 0.0, False, 0, agent_loc_x, agent_loc_y, goal_type, goal_color)] + [local.recv() for local in self.locals]
        return zip(*adapt(self.results, scouting))

    def step(self, actions, scouting):
        for local, action, scouting_ in zip(self.locals, actions[1:], scouting[1:]):
            if not scouting_:
                local.send(("step", action))
        if not scouting[0]:
            obs, reward, done, info = self.envs[0].step(actions[0])
            if done:
                obs = self.envs[0].reset()
            globs = obs.copy()
            obs['image']   = get_local(obs)
            globs['image'] = get_global(self.envs[0])
            step_count = self.envs[0].step_count
            agent_loc_x, agent_loc_y = get_agent_loc(self.envs[0])
            goal_type, goal_color = get_goal(self.envs[0])
            self.results[0] = (globs, obs, reward, done, step_count, agent_loc_x, agent_loc_y, goal_type, goal_color)
        for i in range(len(self.envs)-1):
            if not scouting[i+1]:
                self.results[i+1] = self.locals[i].recv()
        return zip(*adapt(self.results, scouting))

    def render(self):
        raise NotImplementedError
