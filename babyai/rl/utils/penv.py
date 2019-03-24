from multiprocessing import Process, Pipe
import gym

def worker(conn, env0, env1):
    while True:
        cmd, scouting, action = conn.recv()
        if cmd == "step":
            if scouting:
                obs, reward, done, info = env0.step(action)
                if done and env0.step_count >= env0.max_steps:
                    obs = env1.reset()
                else:
                    done = False
                reward *= 0
            else:
                obs, reward, done, info = env1.step(action)
                if done:
                    obs = env0.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            if scouting:
                env1.reset()
            obs = env0.reset()
            conn.send(obs)
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
        results = [self.envs0[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions, scouting):
        for local, action, scouting_ in zip(self.locals, actions[1:], scouting[1:]):
            local.send(("step", scouting_, action))
        
        if scouting[0]:
            obs, reward, done, info = self.envs0[0].step(actions[0])
            if done and self.envs0[0].step_count >= self.envs0[0].max_steps:
                obs = self.envs1[0].reset()
            else:
                done = False
            reward *= 0
        else:
            obs, reward, done, info = self.envs1[0].step(actions[0])
            if done:
                obs = self.envs0[0].reset()

        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError
