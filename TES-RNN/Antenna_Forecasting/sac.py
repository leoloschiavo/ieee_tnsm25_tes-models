import torch
from torch.optim import Adam
from Utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from Utils.agents import SACAgent
from Utils.critics import SACCritic

MSELoss = torch.nn.MSELoss()


class SAC(object):
    """
    Wrapper class for SAC agent (extensible for multiple agents)
    """
    def __init__(self, agent_init_params, num_agents=1, gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01, reward_scale=10., pol_hidden_dim=128, critic_hidden_dim=128, **kwargs):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        self.nagents = num_agents
        self.num_in_pol = agent_init_params[0]['num_in_pol']
        self.num_out_pol = agent_init_params[0]['num_out_pol']
        self.agents = [SACAgent(lr=pi_lr, hidden_dim=pol_hidden_dim, **params) for params in agent_init_params]
        self.critic = SACCritic(self.num_in_pol, self.num_out_pol, nagents=self.nagents, hidden_dim=critic_hidden_dim)
        self.target_critic = SACCritic(self.num_in_pol, self.num_out_pol, nagents=self.nagents, hidden_dim=critic_hidden_dim)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0


    @property
    def policies(self):
        return [a.policy for a in self.agents]


    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]


    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents, observations)]


    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        """
        Update central critic
        """
        obs, acs, rews, next_obs = sample
        # Q loss
        next_acs = []
        next_log_pis = []
        for pi, ob in zip(self.target_policies, next_obs):
            curr_next_ac, curr_next_log_pi = pi(ob, return_log_pi=True)
            next_acs.append(curr_next_ac)
            next_log_pis.append(curr_next_log_pi)
        trgt_critic_in = list(zip(next_obs, next_acs))
        critic_in = list(zip(obs, acs))
        next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in)
        q_loss = 0
        for a_i, nq, log_pi in zip(range(self.nagents), next_qs, next_log_pis):
            target_q = (rews[a_i].view(-1, 1) + self.gamma * nq)
            if soft:
                target_q -= log_pi / self.reward_scale
            q_loss += MSELoss(critic_rets, target_q.detach())

        q_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar('losses/q_loss', q_loss, self.niter)
            logger.add_scalar('grad_norms/q', grad_norm, self.niter)

        self.niter += 1



    def update_policies(self, sample, soft=True, logger=None, **kwargs):

        obs, acs, rews, next_obs = sample
        samp_acs = []
        all_probs = []
        all_log_pis = []
        all_pol_regs = []

        for a_i, pi, ob in zip(range(self.nagents), self.policies, obs):
            curr_ac, probs, log_pi, pol_regs, ent = pi(ob, return_all_probs=True, return_log_pi=True, regularize=True, return_entropy=True)
            samp_acs.append(curr_ac)
            all_probs.append(probs)
            all_log_pis.append(log_pi)
            all_pol_regs.append(pol_regs)

        critic_in = list(zip(obs, samp_acs))
        critic_rets = self.critic(critic_in, return_all_q=True)
        for a_i, probs, log_pi, pol_regs in zip(range(self.nagents), all_probs, all_log_pis, all_pol_regs):
            q = critic_rets[0]
            all_q = critic_rets[1]
            curr_agent = self.agents[a_i]
            v = (all_q * probs).sum(dim=1, keepdim=True)
            pol_target = q - v
            if soft:
                pol_loss = (log_pi * (log_pi / self.reward_scale - pol_target).detach()).mean()
            else:
                pol_loss = (log_pi * (-pol_target).detach()).mean()
            for reg in pol_regs:
                pol_loss += 1e-3 * reg  # policy regularization
            # don't want critic to accumulate gradients from policy loss
            disable_gradients(self.critic)
            pol_loss.backward()
            enable_gradients(self.critic)

            grad_norm = torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar('agent%i/losses/pol_loss' % a_i, pol_loss, self.niter)
                logger.add_scalar('agent%i/grad_norms/pi' % a_i, grad_norm, self.niter)



    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been performed)
        """
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)



    def prep_training(self, device='gpu'):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device



    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device



    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict, 'agent_params': [a.get_params() for a in self.agents], 'critic_params': {'critic': self.critic.state_dict(), 'target_critic': self.target_critic.state_dict(), 'critic_optimizer': self.critic_optimizer.state_dict()}}
        torch.save(save_dict, filename)


    @classmethod
    def init_from_env(cls, num_states, num_actions, num_agents, gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01, reward_scale=100., pol_hidden_dim=128, critic_hidden_dim=128, **kwargs):
        """
        Instantiate instance of this class from single-agent environment
        """
        agent_init_params = []
        for i in range(num_agents):
            agent_init_params.append({'num_in_pol': num_states, 'num_out_pol': num_actions})
        init_dict = {'gamma': gamma, 'tau': tau, 'pi_lr': pi_lr, 'q_lr': q_lr, 'reward_scale': reward_scale, 'pol_hidden_dim': pol_hidden_dim, 'critic_hidden_dim': critic_hidden_dim, 'agent_init_params': agent_init_params}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance


    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)

        if load_critic:
            critic_params = save_dict['critic_params']
            instance.critic.load_state_dict(critic_params['critic'])
            instance.target_critic.load_state_dict(critic_params['target_critic'])
            instance.critic_optimizer.load_state_dict(critic_params['critic_optimizer'])
        return instance