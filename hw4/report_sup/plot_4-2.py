import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

dqn_episodes = []
dqn_rewards = []
ddqn_episodes = []
ddqn_rewards = []

use_update_x = True
threshold = 28000 if not use_update_x else 1400000

plot_compare = len(sys.argv) > 2
with open(sys.argv[1], 'r') as f:
    for i,l in enumerate(f):
        if len(l.split(' ')) < 6:
            continue
        tmp = l.split(' ')
        e = int(tmp[1]) if not use_update_x else int(tmp[3].replace(':', ''))
        r = float(tmp[len(tmp) - 1].replace('\n', ''))
        dqn_episodes.append(e)
        dqn_rewards.append(r)
        if e >= threshold:
        	break
if plot_compare:
    with open(sys.argv[2], 'r') as f:
        for i,l in enumerate(f):
            if len(l.split(' ')) < 6:
                continue
            tmp = l.split(' ')
            e = int(tmp[1]) if not use_update_x else int(tmp[3].replace(':', ''))
            r = float(tmp[len(tmp) - 1].replace('\n', ''))
            ddqn_episodes.append(e)
            ddqn_rewards.append(r)
            if e >= threshold:
                break

fig, ax1 = plt.subplots()

ax1.set_xlabel('Episode' if not use_update_x else 'Update_times')
if not plot_compare:
	plt.title('DQN Training')
	ax1.set_ylabel('Mean 100 Ep Rewards')
	ax1.plot(dqn_episodes, dqn_rewards)
else:
	plt.title('DQN-Dueling_DQN Training')
	ax1.set_ylim(0, 130)
	ax1.set_ylabel('Mean 100 Ep Rewards (DQN)', color='red')
	ax1.plot(dqn_episodes, dqn_rewards, color='red', alpha=0.7)
	ax1.tick_params(axis='y', labelcolor='red')
	
	ax2 = ax1.twinx()
	ax2.set_ylim(0, 130)
	ax2.set_ylabel('Mean 100 Ep Rewards (Dueling_DQN)', color='blue')
	ax2.plot(ddqn_episodes, ddqn_rewards, color='blue', alpha=0.7)
	ax2.tick_params(axis='y', labelcolor='blue')
	

plt.savefig("Training.png")
