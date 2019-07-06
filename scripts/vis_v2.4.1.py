import collections
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import shutil
from sty import fg, bg, ef, rs, RgbFg
import sys



### SETTINGS ###################################################################

env = "GoToObjCustom"

n_frames = 2

num_symbols = 8
max_len_msg = 8

frames = 100000


### VISUALIZATION ###

sort_by = "frequency"

top = 2

show = 0

# show kth frame after message emission (-1 means average over all k)
kth = -1

# if fewer than this percentage of trajectories passed through a node,
# this node (as well as its following nodes) is not shown
threshold = 0.0


### COSMETICS ###

arrow_len = 3

lightness_scale = 1.0

def transform(v):
    return math.pow(v, 1/4)



### CHECK ######################################################################

print("Checking...", end="")
sys.stdout.flush()

assert sort_by in [  "message",
                   "frequency", "frequency_nothing", "frequency_wall", "frequency_object",
                     "action0",   "action0_nothing",   "action0_wall",   "action0_object",
                     "action1",   "action1_nothing",   "action1_wall",   "action1_object",
                     "action2",   "action2_nothing",   "action2_wall",   "action2_object",
                     "action3",   "action3_nothing",   "action3_wall",   "action3_object",
                     "action4",   "action4_nothing",   "action4_wall",   "action4_object",
                     "action5",   "action5_nothing",   "action5_wall",   "action5_object",
                     "action6",   "action6_nothing",   "action6_wall",   "action6_object"]

assert -1 <= kth and kth < n_frames

assert 0.0 <= threshold and threshold <= 1.0

assert 0 <= arrow_len

assert 0.0 <= lightness_scale and lightness_scale <= 1.0

print("             done.")
sys.stdout.flush()



### INITIALIZE #################################################################

print("Loading data...", end="")
sys.stdout.flush()

# load data
data = np.load("messages/%s/%d^%d/data_%s_%d^%d_n%d_f%d.npy" % (env, num_symbols, max_len_msg, env, num_symbols, max_len_msg, n_frames, frames))

frames_actual = len(data)

# dictionary message to action
d_m2a = collections.defaultdict(lambda: np.zeros((n_frames, 3, 7)))

# dictionary message to heatmap; relative (to agent) object positions
d_m2h_goal = collections.defaultdict(lambda: np.zeros((n_frames+1, 3, 13, 13)))
d_m2h      = collections.defaultdict(lambda: np.zeros((n_frames+1, 3, 13, 13, 3, 6)))

# dictionary message to heatmap; absolute object positions
d_m2h_goal_abs = collections.defaultdict(lambda: np.zeros((n_frames+1, 3, 8, 8)))
d_m2h_abs      = collections.defaultdict(lambda: np.zeros((n_frames+1, 3, 8, 8, 3, 6)))

# dictionary message to heatmap; absolute agent position
d_m2h_agt = collections.defaultdict(lambda: np.zeros((n_frames+1, 3, 8, 8)))

trajectories = collections.defaultdict(lambda: [0] + [[0] for i in range(21)])

print("         done.")
sys.stdout.flush()



### COLLECT ####################################################################

print("Combing through data...", end="")
sys.stdout.flush()

j = 0
for i in range(frames_actual):
    # reset if new episode
    new_episode = data[i, 3 + max_len_msg + 2]
    j *= 1 - new_episode
    
    agent_x = data[i, 0]
    agent_y = data[i, 1]

    objects_x     = []
    objects_y     = []
    objects_type  = []
    objects_color = []
    g = 3 + max_len_msg + 4
    while g < len(data[0]):
        object_x     = data[i, g  ]
        object_y     = data[i, g+1]
        object_type  = data[i, g+2]
        object_color = data[i, g+3]
        
        if object_x != 0:
            objects_x.append(object_x)
            objects_y.append(object_y)
            objects_type.append(object_type)
            objects_color.append(object_color)
            
        g += 4

    if agent_y == 1:
        # agent is facing a wall
        condition = 1
    else:
        condition = 0
        for b in range(len(objects_x)):
            if objects_x[b] == agent_x and objects_y[b] == agent_y - 1:
                # agent is facing object
                condition = 2
                break

    if j % n_frames == 0 and new_episode != 1:
        d_m2h_agt[message][n_frames][condition][agent_x][agent_y] += 1
        
        for b in range(len(objects_x)):
            x     = 6 + objects_x[b] - agent_x
            y     = 6 + objects_y[b] - agent_y
            type  = objects_type[b]
            color = objects_color[b]
            
            d_m2h[message][n_frames][condition][x][y][type-5][color] += 1
            
            d_m2h_abs[message][n_frames][condition][objects_x[b]][objects_y[b]][type-5][color] += 1
            
            if type == goal_type and color == goal_color:
                d_m2h_goal[message][n_frames][condition][x][y] += 1
                
                d_m2h_goal_abs[message][n_frames][condition][objects_x[b]][objects_y[b]] += 1

    message = tuple(data[i, 4:4+max_len_msg])
    action  = data[i, 3 + max_len_msg + 1]
    
    goal_type  = data[i, 2]
    goal_color = data[i, 3]
    
    d_m2h_agt[message][j%n_frames][condition][agent_x][agent_y] += 1
    
    for b in range(len(objects_x)):
        x     = 6 + objects_x[b] - agent_x
        y     = 6 + objects_y[b] - agent_y
        type  = objects_type[b]
        color = objects_color[b]
        
        d_m2a[message][j%n_frames][condition][action] += 1
        
        d_m2h[message][j%n_frames][condition][x][y][type-5][color] += 1
        
        d_m2h_abs[message][j%n_frames][condition][objects_x[b]][objects_y[b]][type-5][color] += 1
        
        if type == goal_type and color == goal_color:
            d_m2h_goal[message][j%n_frames][condition][x][y] += 1
            
            d_m2h_goal_abs[message][j%n_frames][condition][objects_x[b]][objects_y[b]] += 1

    # store trajectory
    if j % n_frames == 0:
        current     = trajectories[message]
        current[0] += 1
    
    if len(current) == 1:
        current += [[0] for i in range(21)]

    current = current[action+1 + condition*7]
    current[0] += 1

    j += 1

# sum total
total = np.zeros(n_frames)
for message in d_m2a:
    for i in range(n_frames):
        total[i] += d_m2a[message][i].sum()

print(" done.")
sys.stdout.flush()



### SORT #######################################################################

print("Sorting data...", end="")
sys.stdout.flush()

M = np.zeros((len(d_m2a), max_len_msg + 1))
for i, message in enumerate(d_m2a):
    M[i, :-1] = message

for i in reversed(range(max_len_msg)):
    M = M[M[:, i].argsort(kind="stable")]

def sort_by_action(action):
    if kth == -1:
        for i, message in enumerate(M[:, :-1]):
            if sort_by.endswith("_nothing"):
                s = d_m2a[tuple(message)][:, 0].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][:, 0][action] / s
            elif sort_by.endswith("_wall"):
                s = d_m2a[tuple(message)][:, 1].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][:, 1][action] / s
            elif sort_by.endswith("_object"):
                s = d_m2a[tuple(message)][:, 2].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][:, 2][action] / s
            else:
                M[i, -1] = d_m2a[tuple(message)][:, :, action].sum(0) / d_m2a[tuple(message)].sum()
    else:
        for i, message in enumerate(M[:, :-1]):
            if sort_by.endswith("_nothing"):
                s = d_m2a[tuple(message)][kth, 0].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, 0][action] / s
            elif sort_by.endswith("_wall"):
                s = d_m2a[tuple(message)][kth, 1].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, 1][action] / s
            elif sort_by.endswith("_object"):
                s = d_m2a[tuple(message)][kth, 2].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, 2][action] / s
            else:
                s = d_m2a[tuple(message)][kth].sum()
                if 0 < s:
                    M[i, -1] = d_m2a[tuple(message)][kth, :, action].sum(0) / s

if sort_by != "message":
    if sort_by.startswith("frequency"):
        if kth == -1:
            for i, message in enumerate(M[:, :-1]):
                if sort_by.endswith("_nothing"):
                    M[i, -1] = d_m2a[tuple(message)][:, 0].sum()
                elif sort_by.endswith("_wall"):
                    M[i, -1] = d_m2a[tuple(message)][:, 1].sum()
                else:
                    M[i, -1] = d_m2a[tuple(message)].sum()
        else:
            for i, message in enumerate(M[:, :-1]):
                if sort_by.endswith("_nothing"):
                    M[i, -1] = d_m2a[tuple(message)][kth, 0].sum()
                elif sort_by.endswith("_wall"):
                    M[i, -1] = d_m2a[tuple(message)][kth, 1].sum()
                else:
                    M[i, -1] = d_m2a[tuple(message)][kth].sum()
    elif sort_by.startswith("action"):
        sort_by_action(int(sort_by[6]))

    M = M[M[:, -1].argsort(kind="stable")]
    M = np.flip(M, axis=0)

M = M[:, :-1]

print("         done.")
sys.stdout.flush()



### PRINT ######################################################################

def get_d_max(trajectory, d=0):
    d_max = d
    for i in range(21):
        next_trajectory = trajectory[i+1]
        if threshold*frames_actual < next_trajectory[0]:
            if 1 < len(next_trajectory):
                new_d_max = get_d_max(next_trajectory, d=d+1)
                if d_max < new_d_max:
                    d_max = new_d_max
    return d_max

got_d_max = 0
for message in trajectories:
    new_got_d_max = get_d_max(trajectories[tuple(message)])
    if got_d_max < new_got_d_max:
        got_d_max = new_got_d_max

def sort_actions(trajectory):
    X = list(range(21))
    Y = [0]*21
    for i in range(21):
        Y[i] = trajectory[i+1][0]
    return [x for _,x in reversed(sorted(zip(Y,X)))]

def next_sum(next_trajectory):
    s = 0
    for i in range(21):
        s += next_trajectory[i+1][0]
    return s

def print_trajectory(trajectory, s=None, d=0, d_max=n_frames, j=0):
    if s is None:
        s = trajectory[0]
    actions_sorted = sort_actions(trajectory)
    for i in range(21):
        if actions_sorted[i] < 7:
            condition = " "
        elif actions_sorted[i] < 14:
            condition = "w"
        else:
            condition = "o"
        next_trajectory = trajectory[actions_sorted[i]+1]
        if threshold*frames_actual < next_trajectory[0]:
            q = math.floor((1 - transform(next_trajectory[0] / s)) * 24 * lightness_scale)
            if d == 0:
                if j == 0:
                    print((fg(q+232) + "   %s%d" + fg.rs) % (condition, actions_sorted[i] % 7), end="")
                else:
                    print((fg(q+232) + "\n   %s%d" + fg.rs) % (condition, actions_sorted[i] % 7), end="")
            else:
                if j == 0:
                    print((fg(q+232) + " " + arrow_len*"-" + "> %s%d" + fg.rs) % (condition, actions_sorted[i] % 7), end="")
                else:
                    print((fg(q+232) + "\n" + (5+(d-1)*(5+arrow_len))*" " + " " + arrow_len*"-" + "> %s%d" + fg.rs) % (condition, actions_sorted[i] % 7), end="")
            if 1 < len(next_trajectory):
                n_s = next_sum(next_trajectory)
                if next_trajectory[0] == n_s:
                    print_trajectory(next_trajectory, s, d=d+1, d_max=d_max)
                else:
                    print((fg(q+232) + (d_max-d)*(5+arrow_len)*" " + "   %7d / %-7d = %8.4f" + fg.rs) % (next_trajectory[0] - n_s, next_trajectory[0], (next_trajectory[0] - n_s)/next_trajectory[0]*100), end="")
                    print_trajectory(next_trajectory, s, d=d+1, d_max=d_max, j=1)
            else:
                print((fg(q+232) + (d_max-d)*(5+arrow_len)*" " + "   %7d / %-7d = %8.4f" + fg.rs) % (next_trajectory[0], next_trajectory[0], 100), end="")
            j += 1
    if d != 0 and j == 0:
        q = math.floor((1 - transform(trajectory[0] / s)) * 24 * lightness_scale)
        n_s = next_sum(trajectory)
        print((fg(q+232) + (d_max-d+1)*(5+arrow_len)*" " + "   %7d / %-7d = %8.4f" + fg.rs) % (trajectory[0] - n_s, trajectory[0], (trajectory[0] - n_s)/trajectory[0]*100), end="")

def action_probs(cond, m, k):
    print("%11s   " % cond, end="")
    if cond == "all":
        if k == -1:
            actions = d_m2a[tuple(m)][:, :].sum(0).sum(0)
        else:
            actions = d_m2a[tuple(m)][k].sum(0)
    else:
        if cond == "nothing":
            c = 0
        elif cond == "wall":
            c = 1
        elif cond == "object":
            c = 2
        if k == -1:
            actions = d_m2a[tuple(m)][:, c].sum(0)
        else:
            actions = d_m2a[tuple(m)][k, c]
    sum = actions.sum()
    if 0 < sum:
        actions /= sum
    for action in actions:
        print("%03.2f   " % action, end="")
    print("%7d" % sum, end="")
    if k == -1:
        print("   %7.4f" % (sum/total.sum()*100))
    else:
        print("   %7.4f" % (sum/total[k].sum()*100))

for i, m in enumerate(M[:top]):
    print("\n\n")
    print("%2d " % i, end="")
    for s in m:
        print(chr(97+int(s+0.5)), end="")
    print()
    print("                                   actions")
    print("                 0      1      2      3      4      5      6")
    print("        cdn    lft    rgt    fwd    pkp    drp    tgl    dne       frq     % tot")
    print("   -----------------------------------------------------------------------------")

    action_probs("all",     m, kth)
    action_probs("nothing", m, kth)
    action_probs("wall",    m, kth)
    action_probs("object",  m, kth)

    print("\n" + (5 + got_d_max*(5+arrow_len))*" " + "      stop / pass        % stop")
    print_trajectory(trajectories[tuple(m)], d_max=got_d_max)
    print()



### PLOT #######################################################################

# Used to map colors to integers
COLOR_TO_IDX = {
    "red"   : 0,
    "green" : 1,
    "blue"  : 2,
    "purple": 3,
    "yellow": 4,
    "grey"  : 5
}

# Map of object type to integers
OBJECT_TO_IDX = {
#    "unseen"        : 0,
#    "empty"         : 1,
#    "wall"          : 2,
#    "floor"         : 3,
#    "door"          : 4,
    "key"           : 0,
    "ball"          : 1,
    "box"           : 2,
#    "goal"          : 8,
#    "lava"          : 9
}

#for message in d_m2h:
    # object distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, :, :].sum(-1).sum(-1)
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, :, :].sum(-1).sum(-1)

    # ball distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, OBJECT_TO_IDX["ball"], :].sum(-1)
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, OBJECT_TO_IDX["ball"], :].sum(-1)

    # blue object distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, :, COLOR_TO_IDX["blue"]].sum(-1)
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, :, COLOR_TO_IDX["blue"]].sum(-1)

    # blue ball distribution
    #d_m2h[message]     = d_m2h[message][:, :, :, :, OBJECT_TO_IDX["ball"], COLOR_TO_IDX["blue"]]
    #d_m2h_abs[message] = d_m2h_abs[message][:, :, :, :, OBJECT_TO_IDX["ball"], COLOR_TO_IDX["blue"]]

# goal distribution
d_m2h     = d_m2h_goal
d_m2h_abs = d_m2h_goal_abs

def normalize(h):
    h_new = h.copy()
    s = h_new.sum()
    if 0 < s:
        h_new /= s
    return h_new.transpose()

def addagent(h, q=32):
    assert q == 32
    
    cell = h[6*q:7*q, 6*q:7*q, :]
    
    cell[ 2   , 15   , :] =         np.array([1, 0, 0, 1])
    cell[ 3: 4, 15:17, :] = np.tile(np.array([1, 0, 0, 1]), [1,  2, 1])
    cell[ 4: 5, 14:17, :] = np.tile(np.array([1, 0, 0, 1]), [1,  3, 1])
    cell[ 5: 7, 14:18, :] = np.tile(np.array([1, 0, 0, 1]), [1,  4, 1])
    cell[ 7: 9, 13:19, :] = np.tile(np.array([1, 0, 0, 1]), [1,  6, 1])
    cell[ 9:10, 12:19, :] = np.tile(np.array([1, 0, 0, 1]), [1,  7, 1])
    cell[10:12, 12:20, :] = np.tile(np.array([1, 0, 0, 1]), [1,  8, 1])
    cell[12:14, 11:21, :] = np.tile(np.array([1, 0, 0, 1]), [1, 10, 1])
    cell[14:15, 10:21, :] = np.tile(np.array([1, 0, 0, 1]), [1, 11, 1])
    cell[15:16, 10:22, :] = np.tile(np.array([1, 0, 0, 1]), [1, 12, 1])
    cell[16:17,  9:22, :] = np.tile(np.array([1, 0, 0, 1]), [1, 13, 1])
    cell[17:19,  9:23, :] = np.tile(np.array([1, 0, 0, 1]), [1, 14, 1])
    cell[19:21,  8:24, :] = np.tile(np.array([1, 0, 0, 1]), [1, 16, 1])
    cell[21:22,  7:24, :] = np.tile(np.array([1, 0, 0, 1]), [1, 17, 1])
    cell[22:24,  7:25, :] = np.tile(np.array([1, 0, 0, 1]), [1, 18, 1])
    cell[24:27,  6:26, :] = np.tile(np.array([1, 0, 0, 1]), [1, 20, 1])
    
    h[6*q:7*q, 6*q:7*q, :] = cell
    
    # red square
    #h[6*q:7*q, 6*q:7*q, :] = np.tile(np.array([1, 0, 0, 1]), [q, q, 1])
    
    return h

def expand(h, q=32):
    N = h.shape[0]
    M = h.shape[1]
    Z = h.shape[2]
    h_new = np.zeros((N*q, M*q, Z))
    for i in range(N):
        for j in range(M):
            h_new[i*q:(i+1)*q, j*q:(j+1)*q, :] = np.tile(h[i, j, :], [q, q, 1])
    return h_new

def imshow2(ax, h, t="", add_agent=False):
    ax.set_title(t)
    norm = plt.Normalize(0.0, h.max())
    rgba = expand(cmap(norm(h)))
    if add_agent:
        rgba = addagent(rgba)
    im = ax.imshow(h, cmap=cmap, vmin=0.0, vmax=h.max())
    ax.imshow(rgba)
    fig.colorbar(im, ax=ax, orientation='vertical')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

def imsave2(fname, h, cmap, add_agent=False):
    norm = plt.Normalize(0.0, h.max())
    rgba = expand(cmap(norm(h)))
    if add_agent:
        rgba = addagent(rgba)
    plt.imsave(fname, rgba, format="png")

def imssave2(d_m2, str):
    print("Saving %s files..." % str, end="")
    sys.stdout.flush()
    
    if str == "rel" or str == "abs":
        cmap = plt.cm.cividis
    elif str == "agt":
        cmap = plt.cm.inferno
    
    if str == "rel":
        add_agent = True
    else:
        add_agent = False

    for m in M[:top]:
        t = ""
        for s in m:
            t += chr(97+int(s))
        
        if not os.path.exists("renders/%s" % t):
            os.mkdir("renders/%s" % t)
        if not os.path.exists("renders/%s/%s" % (t, str)):
            os.mkdir("renders/%s/%s" % (t, str))
        if not os.path.exists("renders/%s/%s/all" % (t, str)):
            os.mkdir("renders/%s/%s/all" % (t, str))
        if not os.path.exists("renders/%s/%s/nothing" % (t, str)):
            os.mkdir("renders/%s/%s/nothing" % (t, str))
        if not os.path.exists("renders/%s/%s/wall" % (t, str)):
            os.mkdir("renders/%s/%s/wall" % (t, str))
        if not os.path.exists("renders/%s/%s/object" % (t, str)):
            os.mkdir("renders/%s/%s/object" % (t, str))

        h_a_agt = normalize(d_m2[tuple(m)].sum(0).sum(0))
        h_0_agt = normalize(d_m2[tuple(m)][:, 0].sum(0))
        h_1_agt = normalize(d_m2[tuple(m)][:, 1].sum(0))
        h_2_agt = normalize(d_m2[tuple(m)][:, 2].sum(0))
        
        imsave2("renders/%s/%s/all/k_avg.png"     % (t, str), h_a_agt, cmap, add_agent)
        imsave2("renders/%s/%s/nothing/k_avg.png" % (t, str), h_0_agt, cmap, add_agent)
        imsave2("renders/%s/%s/wall/k_avg.png"    % (t, str), h_1_agt, cmap, add_agent)
        imsave2("renders/%s/%s/object/k_avg.png"  % (t, str), h_2_agt, cmap, add_agent)
        
        for j in range(n_frames+1):
            h_a_agt = normalize(d_m2[tuple(m)][j].sum(0))
            h_0_agt = normalize(d_m2[tuple(m)][j, 0])
            h_1_agt = normalize(d_m2[tuple(m)][j, 1])
            h_2_agt = normalize(d_m2[tuple(m)][j, 2])
            
            imsave2("renders/%s/%s/all/k%d.png"     % (t, str, j), h_a_agt, cmap, add_agent)
            imsave2("renders/%s/%s/nothing/k%d.png" % (t, str, j), h_0_agt, cmap, add_agent)
            imsave2("renders/%s/%s/wall/k%d.png"    % (t, str, j), h_1_agt, cmap, add_agent)
            imsave2("renders/%s/%s/object/k%d.png"  % (t, str, j), h_2_agt, cmap, add_agent)
            
    print("        done.")
    sys.stdout.flush()

m = M[show]

t = ""
for s in m:
    t += chr(97+int(s))

### FIGURE 1 ###

cmap = plt.cm.cividis

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

if kth == -1:
    fig.suptitle(r"$m$ = %s, $k = k_{\mathrm{all}}$" % t)
    h_a = normalize(d_m2h[tuple(m)].sum(0).sum(0))
    h_0 = normalize(d_m2h[tuple(m)][:, 0].sum(0))
    h_1 = normalize(d_m2h[tuple(m)][:, 1].sum(0))
    h_2 = normalize(d_m2h[tuple(m)][:, 2].sum(0))
else:
    fig.suptitle(r"$m$ = %s, $k$ = %d" % (t, kth))
    h_a = normalize(d_m2h[tuple(m)][kth].sum(0))
    h_0 = normalize(d_m2h[tuple(m)][kth, 0])
    h_1 = normalize(d_m2h[tuple(m)][kth, 1])
    h_2 = normalize(d_m2h[tuple(m)][kth, 2])

imshow2(ax0, h_a, r"$P(g | m, k)$",          add_agent=True)
imshow2(ax1, h_0, r"$P(g | m, k, \mathrm{nothing})$", add_agent=True)
imshow2(ax2, h_1, r"$P(g | m, k, \mathrm{wall})$",    add_agent=True)
imshow2(ax3, h_2, r"$P(g | m, k, \mathrm{object})$",  add_agent=True)

### FIGURE 2 ###

cmap = plt.cm.cividis

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

if kth == -1:
    fig.suptitle(r"$m$ = %s, $k = k_{\mathrm{all}}$" % t)
    h_a_abs = normalize(d_m2h_abs[tuple(m)].sum(0).sum(0))
    h_0_abs = normalize(d_m2h_abs[tuple(m)][:, 0].sum(0))
    h_1_abs = normalize(d_m2h_abs[tuple(m)][:, 1].sum(0))
    h_2_abs = normalize(d_m2h_abs[tuple(m)][:, 2].sum(0))
else:
    fig.suptitle(r"$m$ = %s, $k$ = %d" % (t, kth))
    h_a_abs = normalize(d_m2h_abs[tuple(m)][kth].sum(0))
    h_0_abs = normalize(d_m2h_abs[tuple(m)][kth, 0])
    h_1_abs = normalize(d_m2h_abs[tuple(m)][kth, 1])
    h_2_abs = normalize(d_m2h_abs[tuple(m)][kth, 2])

imshow2(ax0, h_a_abs, r"$P(g | m, k)$")
imshow2(ax1, h_0_abs, r"$P(g | m, k, \mathrm{nothing})$")
imshow2(ax2, h_1_abs, r"$P(g | m, k, \mathrm{wall})$")
imshow2(ax3, h_2_abs, r"$P(g | m, k, \mathrm{object})$")

### FIGURE 3 ###

cmap = plt.cm.inferno

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)

if kth == -1:
    fig.suptitle(r"$m$ = %s, $k = k_{\mathrm{all}}$" % t)
    h_a_agt = normalize(d_m2h_agt[tuple(m)].sum(0).sum(0))
    h_0_agt = normalize(d_m2h_agt[tuple(m)][:, 0].sum(0))
    h_1_agt = normalize(d_m2h_agt[tuple(m)][:, 1].sum(0))
    h_2_agt = normalize(d_m2h_agt[tuple(m)][:, 2].sum(0))
else:
    fig.suptitle(r"$m$ = %s, $k$ = %d" % (t, kth))
    h_a_agt = normalize(d_m2h_agt[tuple(m)][kth].sum(0))
    h_0_agt = normalize(d_m2h_agt[tuple(m)][kth, 0])
    h_1_agt = normalize(d_m2h_agt[tuple(m)][kth, 1])
    h_2_agt = normalize(d_m2h_agt[tuple(m)][kth, 2])

imshow2(ax0, h_a_agt, r"$P(a | m, k)$")
imshow2(ax1, h_0_agt, r"$P(a | m, k, \mathrm{nothing})$")
imshow2(ax2, h_1_agt, r"$P(a | m, k, \mathrm{wall})$")
imshow2(ax3, h_2_agt, r"$P(a | m, k, \mathrm{object})$")

plt.pause(0.05)

### SAVE ###

print()

if not os.path.exists("renders"):
    os.mkdir("renders")
else:
    print("Deleting existing files...", end="")
    sys.stdout.flush()
    for file in os.listdir("renders"):
        file_path = os.path.join("renders", file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)
    print(" done.")
    sys.stdout.flush()

imssave2(d_m2h,     "rel")
imssave2(d_m2h_abs, "abs")
imssave2(d_m2h_agt, "agt")

plt.show()
