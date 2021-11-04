import argparse
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import *
from models.mlp_policy import Policy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from GazeboWorld_ppo import GazeboWorld

parser = argparse.ArgumentParser(description='PyTorch PPO example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-5, metavar='G',
                    help='learning rate (default: 3e-5)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = GazeboWorld()
img_depth_dim = env.observation_space_img_depth
# print "dimension:", env.observation_space_img_depth, "\n\n\n"
goal_dim = env.observation_space_goal
# state_dim = depth_dim + goal_dim
is_disc_action = len(env.action_space.shape) == 0
running_state = ZFilter(img_depth_dim, goal_dim, clip=30) # set clip to be 30 which is the maximum value for the depth value
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        policy_net = DiscretePolicy(0, env.action_space.n)
    else:
        policy_net = Policy(env.action_space.shape[0], log_std=args.log_std)
    value_net = Value()
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 10
optim_batch_size = 64

"""create agent"""
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads)


def update_params(batch, i_iter):
    imgs_depth = torch.from_numpy(np.stack(batch.img_depth)).to(dtype).to(device)
    goals = torch.from_numpy(np.stack(batch.goal)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(imgs_depth, goals)
        fixed_log_probs = policy_net.get_log_prob(imgs_depth, goals, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(imgs_depth.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(imgs_depth.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        imgs_depth, goals, actions, returns, advantages, fixed_log_probs = \
            imgs_depth[perm].clone(), goals[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, imgs_depth.shape[0]))
            imgs_depth_b, goals_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                imgs_depth[ind], goals[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, imgs_depth_b, goals_b,
                     actions_b, returns_b, advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()
        """evaluate with determinstic action (remove noise for exploration)"""
        if args.eval_batch_size > 0:
            _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        t2 = time.time()

        if i_iter % args.log_interval == 0:
            if args.eval_batch_size > 0:
                print('{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\teval_R_avg {:.2f}'.format(
                    i_iter, log['sample_time'], t1-t0, t2-t1, log['min_reward'], log['max_reward'], log['avg_reward'], log_eval['avg_reward']))
            else:
                print(
                '{}\tT_sample {:.4f}\tT_update {:.4f}\tT_eval {:.4f}\ttrain_R_min {:.2f}\ttrain_R_max {:.2f}\ttrain_R_avg {:.2f}\t'.format(
                    i_iter, log['sample_time'], t1 - t0, t2 - t1, log['min_reward'], log['max_reward'], log['avg_reward']))

        my_open = open(os.path.join(assets_dir(), 'learned_models/{}_ppo.txt'.format(args.env_name)), "a")
        data = [str(i_iter), " ", str(log['avg_reward']), " ", str(log['num_episodes']),
                " ", str(log['ratio_success']), " ", str(log['avg_steps_success']), " ", str(log['avg_last_reward']), "\n"]
        for element in data:
            my_open.write(element)
        my_open.close()

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            pickle.dump((policy_net, value_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
