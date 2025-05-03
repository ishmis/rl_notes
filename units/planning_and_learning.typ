= Planning and Learning 
*Planning: *$eq.def$ any process that uses a model of the environment to compute a plan of action (policy) to achieve a specified goal. \
// space!
*Types of Models:*
1. *Distribution Model: * Produce a description of all possibilities and their probabilities. *DP* methods use these: $p(s', r| s, a)$
2. *Simulation (Sample) Model: * Produces sample outcomes $(s', r) tilde hat(p) (s,a) s.t. Pr{hat(p)(s,a) eq (s', r)} eq p(s', r bar s, a)$
// space!
*Types of Planning:*
1. *State-Space (SSP): * involves searching through the state space for an optimal policy or an optimal path to the goal. (we focus on thiws)
2. *Plan-Space: * involves searching through a space of plans instead. 
*SSP* involves: 
1. Computing value functions as a key step to improving the policy
2. Computing value function by updates or backup operations (operations around update targets) applies to simulated experience 
*Offline Planning:*
- Uses MDP (provided) to find optimal policy before applying to real experience + uses as much time as needed to find optimal policy
- Policy is *complete* - it can give an optimal action for all possible states.
- Examples: *DP* and *Dyna-Q*
*Online Planning:* 
- Uses an MDP to find the best policy as the MDP is experienced 
- Limited compute budget at each state
- Policy usually incomplete: only possible to compute optimal action for current state
- Examples: *Rollout Planning* including *MCTS*. 
== Dyna-Q
*Dyna: * RL architecture that includes planning, learning and acting
// space!
#figure(image("../figures/dyna.png", width: 55%),
)
// space!
*Direct RL:* Involves methods like *MC* or *TD* that learn from experience\
// space!
*Search Control:* Refers to the process that selects starting states and actions for the simulated experiences. \
// space!
*Dyna-Q:* This is where *Q-Learning* is used for both learning and planning. It is possible for the planning, acting and model-learning steps of *Dyna* to be carried out in parallel but the provided algorithm is a sequential one:
// space!
#figure(image("../figures/tabular_dyna_q.png", width: 100%),
)
// space!
*Note*:
1. Direct RL, model-learning and planning are implemented by steps $(d), (e), "and" (f)$ respectively. 
2. The planning step involves _random-sample one-step Q-Learning_, where the estimate $Q$ is improved *only* through prior experienced $(s,a)$ pairs!
3. The environment is assumed too be deterministic. 
4. Updating the value function through planning is called *indirect RL*. 
// space!
#underline[*What happens when model is wrong?*]\ 
The learned model could be wrong when:
1. Stochastic environment but only limited number of samples observed 
2. Model learned using approximation that has generalized poorly
3. Environment has changed and new behavior has not been observed
In some cases, suboptimal policy computed by planning quickly leads to the discovery and correction of the modeling error. This tends to happen when the model is optimistic in the sense of predicting greater reward or better state transitions than are actually possible. The planned policy attempts to exploit these opportunities and in doing so discovers that they do not exist. \
// space
#underline[*Dyna-Q+*] \
In order to improve exploration in Dyna-Q, a heuristic is introduced to force the agent to try $(s,a)$ pairs that is has not for a long time by forcing it to assume that the model for them is incorrect. This is achieved simply by adding a _bonus reward_ to $(s,a)$, scaling with how long ago they were tried: 
$ R_t^' =  R_t + k sqrt(tau) $
where $k$ is small and $tau$ refers the time since last visiting $(s,a)$. 
== Rollout Algorithms 
1. _Dyna-Q_ uses a model to reuse past experiences. _Rollout_ algorithms instead use a model to simulate ("rollout") future trajectories. 
2. They apply *MC* control to simulated trajectories that all being at the current environment state.  
3. They estimate action values for a given policy by averaging the returns of many simulated trajectories that start with each possible action and then follow the given policy. When the action-value estimates are considered to be accurate enough, the action (or one of the actions) having the highest estimated value is executed, after which the process is carried out anew from the resulting next state
4. Goal is not to estimate a _complete optimal policy_ but instead to produce *MC* estimates of action-values *only* for each current state using a _rollout policy_, making immediate use of the estimates and then discarding them. 
5. The _policy improvement theorem_ holds with this approach through results similar to one step of policy iteration (closer to _asynchronous DP_ ). 
// space!
#figure(image("../figures/rollout_fwd.png", width: 100%),
)
// space!
#figure(image("../figures/rollout_bkwd.png", width: 100%),
)
// space!
$crossmark$ Several timing challenges with Rollout: number of actions that have to be evaluated for each decision, the number of time steps in the simulated trajectories needed to obtain useful sample returns, the time it takes the rollout policy to make decisions, and the number of simulated trajectories needed to obtain good Monte Carlo action-value estimates. $checkmark$ *MC* trials are independent can can be ran in parallel $checkmark$ simulated trajectories can be truncated $checkmark$ actions unlikely to be the best can be pruned away 
== MC Tree Search (MCTS)
1. General, efficient rollout planning with _backward updating_
2. Stores partial $Q$ as tree and asymmetrically expand tree based on most promising actions. That is, $ Q(s,a) eq EE [R_(t+1) + gamma max_a' Q(s_(t+1), a') bar S_t = a, A_t = a] $
3. Like any *MC* method, value of $(s,a)$ pairs are estimated as average simulated returns. However, estimates for only a subset of promising $(s,a)$ pairs are kept in a tree, rooted from the current state. 
4. *MCTS* incrementally extends the tree. Outside the tree and at leaf nodes, a _rollout policy_ is used. At states inside the tree, action are picked from the stored estimates, giving a _tree policy_.  
*Phases of MCTS:*
1. *Selection: * Starting at the root, the _tree policy_ is used to traverse the tree and select an action at one the leaf nodes.
2. *Expansion: * _On some iterations_, the tree is expanded from the selected leaf node by adding $>=1$ child nodes, reached via unexplored actions. 
3. *Simulation: * From the selected leaf node, or one of its newly-added children, episode simulate is ran using the _rollout policy_. The result is a *MC* trial with actions selected first by the _tree policy_ and beyond the tree by the _rollout policy_. 
4. *Backup:* The return generated is backed up to update, or to initialize, the action values attached to the edges of the tree traversed by the _tree policy_ in this iteration of *MCTS*. No values are saved for the states and actions visited by the _rollout policy_ beyond the tree.
// space!
#figure(image("../figures/mcts_visually.png", width: 100%),
)
// space!
*MCTS* continues executing these four steps, starting each time at the tree’s root node, until no more time is left, or some other computational resource is exhausted. An action is selected from the actions right after the root and *MCTS* is run again using the subtree rooted under the selected action or completely from scratch (not done often). 
// space!
#figure(image("../figures/mcts_pseudo.png", width: 100%),
)
// space!
#underline[*UCB for Trees (UCT)*]\
The _tree policy_ is meant to be exploratory, so it could be $epsilon$-greedy but it could also use *UCB* as follows:
$ A := cases(
 a "if" a "never tried in" s,
 "argmax"_a Q(s,a) + c sqrt(log((N(s)) / (N(s,a)))) "if" a != A^*
) $
where $N(s)$ is number of times $s$ has been visited and $N(s,a)$ is the number of times $a$ was selected in $s$. 
// space!
#figure(image("../figures/simulation_step.png", width: 100%),
)
// space!