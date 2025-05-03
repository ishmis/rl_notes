= Markov-Decision Processes
*MDP: * Consists of the *state* (unlike MAB), action and reward space + environment dynamics $p(s', r bar s, a)$. Finite MDP if $S$, $A$, $R$ finite.  The MDP is controlled with a stochastic or deterministic policy $pi$.  \
// space!
*Environment Dynamics*: 
$  p(s' bar s, a) eq.def sum_(r in R) p(s', r bar s, a) $
$  r(s, a) eq.def sum_(r in R, s' in S) r * p(s', r bar s, a) $
$  r(s, a, s') eq.def sum_(r in R) r * p(s', r bar s, a) / p(s' bar s, a) $
// space!
*Trajectory: * sequence of states, actions and rewards: $S_0, A_0, R_1, S_1, A_1, R_2, ... $ \
// space!
*Markov Property:* Future state and reward are independent of past states and actions, given the current state and action. \ 
// space!
*Markov Chain: * MDP w/ a deterministic policy \ 
// space!
*Return (un-discounted): * $G_t eq.def R_(t+1) + R_(t+2) + ... eq R_(t+1) + G_(t+1)$ \ 
// space!
*Episode: * Particular sequence of agent-environment interaction that ends w/ a terminal state at time step $T$. Tasks with episodes are called *Episodic* and distinguish between $S$ (non-terminal states) and $S^+$ (all states). \ 
*Discounting*: In order to accommodate tasks that continue without limit (called _continuing_ or _non-episodic_), a *discount rate* $gamma$ is introduced: 
$ G_t eq.def R_(t+1)+gamma G_(t+1) eq.def sum_(k=0)^infinity gamma^k R_(t+k+1) "where " 0 <= gamma <= 1 $ For terminating states, $G_T = 0$ by convention. 
The sum itself is finite for $gamma < 1$ and bounded rewards $R_t <= r_max$: \
$ sum_(k=0)^infinity gamma^k R_(t+k+1) <= r_max sum_(k=0)^infinity gamma^k = r_max / (1- gamma ) $ 
// space!
*Absorbing State: * Terminal state that transitions into itself and gives reward 0. \
// space!
*Bellman Equations*: For _state-value function_ it is
$ v_pi (s) &eq.def EE_pi [G_t bar S_t = s] \
           &eq EE_pi [R_(t+1) + gamma G_(t+1) bar S_t = s] \
           &eq sum_a pi(a bar s) sum_(s', r) [r + gamma EE_pi [G_(t+1) bar S_(t+1) = s']] \ 
           &eq sum_a pi(a bar s) sum_(s', r) [r + gamma v_pi (s')] $
This is a direct result of the Markov Property! The _action-value function_: 
$ q_pi (s, a) eq.def EE_pi [G_t bar S_t = s, A_t = a] eq sum_(s',r) p(s', r bar s, a) [r + gamma v_pi (s')] $
// space!
*Optimal Policy*: A policy $pi$ is optimal if $v_pi (s) >= v_(pi') (s), forall s in S$ and $forall pi' != pi$. Any $pi$ that is optimal is denoted $pi_*$ and they all share the same value functions $v_*$ and $q_*$. In particular: 
$ v_*(s) eq.def max_pi v_pi (s) "and" q_*(s,a) eq.def max_pi q_pi (s,a), forall s,a $
// space!
*Bellman Optimality Equations:* 
$ v_*(s) &eq.def max_(a in A) q_pi_* (s,a) \ 
         &eq max_a sum_(s',r) p(s', r bar s, a) [r + gamma v_* (s')] $
$ q_*(s, a) &eq.def EE[R_(t+1) + gamma max_(a') q_* (S_(t+1), a') bar S_t = s, A_t = a] \ 
         &eq sum_(s',r) p(s', r bar s, a) [r + gamma max_(a') q_* (s', a')] $
#figure(image("../figures/backup_optimality_eqns.png", width: 100%),
)
*Note*:
1. The value function $v$ (and $q$) can be computed as a unique solution to $bar"S"bar$ Bellman Equations (one for each state)
2. The optimal $v_*$ could also be computed uniquely with a system of Bellman Optimality Equations, but due to the _max_ operator, the system consists of non-linear equations. 
3. Either above are rarely done as it is computationally intensive and scales with the number of states. Moreover, the true dynamics of the environment is rarely accurately known. 