import torch.nn as nn


class SACCritic(nn.Module):
    """
    Critic Network
    """
    def __init__(self, num_in_pol, num_out_pol, nagents=1, hidden_dim=32, norm_in=True):
        """
        Inputs:
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
        """

        super(SACCritic, self).__init__()
        self.nagents = nagents
        self.num_states = num_in_pol
        self.num_actions = num_out_pol
        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.state_encoders = nn.ModuleList()

        for i in range(self.nagents):
            idim = self.num_states + self.num_actions
            odim = self.num_actions
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module('enc_bn', nn.BatchNorm1d(idim, affine=False))
            encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            encoder.add_module('enc_nl', nn.LeakyReLU())
            self.critic_encoders.append(encoder)
            critic = nn.Sequential()
            critic.add_module('critic_fc1', nn.Linear(hidden_dim, hidden_dim))
            critic.add_module('critic_nl', nn.LeakyReLU())
            critic.add_module('critic_fc2', nn.Linear(hidden_dim, odim))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module('s_enc_bn', nn.BatchNorm1d(self.num_states, affine=False))
            state_encoder.add_module('s_enc_fc1', nn.Linear(self.num_states, hidden_dim))
            state_encoder.add_module('s_enc_nl', nn.LeakyReLU())
            self.state_encoders.append(state_encoder)



    def forward(self, inps, agents=None, return_q=True, return_all_q=False):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
        """
        if agents is None:
            agents = range(len(self.critic_encoders))

        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        s_encodings = [self.state_encoders[a_i](states[a_i]) for a_i in agents]

        # calculate Q
        all_rets = []
        for i, a_i in enumerate(agents):
            agent_rets = []
            all_q = self.critics[a_i](s_encodings[i])
            int_acs = actions[a_i].max(dim=1, keepdim=True)[1]
            q = all_q.gather(1, int_acs)

            if return_q:
                agent_rets.append(q)
            if return_all_q:
                agent_rets.append(all_q)
            if len(agent_rets) == 1:
                all_rets.append(agent_rets[0])
            else:
                all_rets.append(agent_rets)

        if len(all_rets) == 1:
            return all_rets[0]
        else:
            return all_rets
