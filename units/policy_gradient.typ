= Policy Gradient Methods
Three kinds of RL Algorithms:
1. *Value-Based*: use estimates $Q, V$ to find optimal policy.
2. *Model-Based:* learn model through approximations $hat(q), hat(r)$ or have a perfect model like *DP* methods, and *plan* to get optimal policy 
3. *Policy-Based:* Learn policy directly!
// space!
#let btheta = $bold(theta)$
#let bw = $bold(w)$
#underline[*Policy-Based RL*]\ 
Let $btheta in RR^(d')$ be some policy parameter vector such that $ pi(a bar s, btheta) eq Pr{A_t = a bar S_t = s, btheta_t= btheta} $
If a method uses a learned value function, then its weight vector is denoted $bw in RR^d$, as usual, in $hat(v)(s, bw)$. Let $J(btheta)$ denote the scalar performance measure w.r.t. the policy parameter. These methods seek to maximize performance using _gradient ascent_ as follows:
// space!
$ btheta_(t+1) eq btheta_t + alpha hat(nabla J(btheta_t)) $
// space!
where $hat(nabla J(btheta_t)) in RR^(d')$ is a stochastic estimate whose expectation approximates the gradient of the performance measure w.r.t. $btheta_t$. \
*Note:* Methods that learn approximations to both policy and value functions are often called _actor–critic_ methods. _Actor_ refers to the learned policy, and _Critic_ refers to the learned value function, usually a state-value function. 
== Representing the Policy 
The policy can be parametrized in any way, as long as $pi (a bar s, btheta)$ is differentiable w.r.t. its parameters. To ensure exploration, we ensure that the policy is never becomes deterministic. \
*#underline[Soft-Max Representation:]*\
#let bx = $bold(x)$
// space!
$ pi(a bar s, btheta) eq (e^(h(s,a,btheta))) / (sum_(b in A)e^(h(s,b,btheta))) $
// space!
$h$ is a preference function (recall *Gradient Bandits* in 2.2) and can be given linearly as $h(s,a,btheta) eq btheta^T bx(s,a)$ where $bx(s,a)$ is a feature vector $in RR^(d')$. \ $checkmark$ The approximate policy approaches a deterministic policy unlike a $epsilon$-greedy policy $checkmark$ Enables the selection of actions with arbitrary probabilities \ 
*#underline[Gaussian Representation:]*\
$ pi(a bar s, btheta) eq N(mu(s, btheta), sigma^2) "and " mu(s,btheta) eq btheta^T bx(s) $
== Policy-Optimisation Problem 
*Problem:* Given $pi (a bar s)$, interact with *MDP* $M$ and find optimal $btheta$ \ 
We limit our problems to _episodic tasks_ and solve them using a performance measure $J (btheta)$, defined as the value of the *start state* of the episode:
// space!
$ J(btheta) eq.def v_(pi_btheta)(s_0) $
// space!
Issue with $J(btheta)$ is that performance depends on both action selection and the distribution of states in which those selections are made (both affected by $btheta$). Computing the effect of $btheta$ on actions, and thus rewards, is simple. However, the effect of $btheta$ on state distribution requires knowledge of the environment, which is assumed to be unknown. The following helps:
*Policy Gradient Theorem:* \
For any differentiable policy $pi$, the policy gradient is:
$ 
nabla J(btheta) prop sum_s d_pi (s) sum_a q_pi (s, a) nabla pi (a bar s, btheta ) 
$
where $d_pi (s)$ is the _on-policy distribution_ under $pi$. In _episodic tasks_, the above equation is actually an equality, but the proportional case includes continuing tasks. The _on-policy distribution_ can be:
1. Using _start-state value_: $d_pi (s) eq sum_(t=0)^infinity gamma^t Pr{S_t = s bar s_0, pi} $
2. Using _average reward_: $d_pi (s) eq lim_(t->infinity) Pr{S_t = s bar pi} $
*Note:* No environment dynamics involved at all! 
== Policy-Gradient Theorem Breakdown
The theorem is useful as it transforms the earlier _gradient ascent_ rule using $hat(nabla J(btheta_t))$ into a simpler update rule. The derivation is provided below: 
// space!
$ 
nabla J(btheta) 
  &eq sum_s d_pi (s) sum_a q_pi (s,a) nabla pi (a bar s, btheta) \
  & "replace s by sample" S_t tilde pi \
  &eq EE_pi [sum_a q_pi (S_t,a) nabla pi (a bar S_t, btheta)]  \
  & "introduce the policy" \
  &eq EE_pi [sum_a pi (a bar S_t, btheta) q_pi (S_t,a) (nabla pi (a bar S_t, btheta)) / (pi (a bar S_t, btheta))]  \
  & "replace a by sample" A_t tilde pi \
  &eq EE_pi [q_pi (S_t,A_t) (nabla pi (A_t bar S_t, btheta)) / (pi (A_t bar S_t, btheta))]\
  &eq EE_pi [G_t (nabla pi (A_t bar S_t, btheta)) / (pi (A_t bar S_t, btheta))]
$
$
  &eq EE_pi [G_t nabla ln pi (A_t bar S_t, btheta)] "(using" nabla ln x eq nabla x / x ")"\            
$
// space!
Finally, we get the update rule (called *REINFORCE*):
// space!
$ 
theta_(t+1) eq.def theta_t + alpha [G_t (nabla pi (A_t bar S_t, btheta)) / (pi (A_t bar S_t, btheta))] "or "theta_(t+1) eq.def theta_t + alpha G_t nabla ln pi (A_t bar S_t, btheta)
$ 
// space!
The vector is the direction in parameter space that most increases the probability of repeating the action $A_t$ on future visits to state $S_t$. The update increases the parameter vector in this direction proportional to the return, and inversely proportional to the action probability. The former makes sense because it causes the parameter to move most in the directions that favor actions that yield the highest return. The latter makes sense because otherwise actions that are selected frequently are at an advantage (the updates will be more often in their direction) and might win out even if they do not yield the highest return. \
Algorithms that use *REINFORCE* typically approximate the terms $q_pi$ and $nabla ln pi$. *MC* estimates can be used for $q_pi$ (as shown below) whereas $nabla ln pi$ can be estimated as:
$ 
nabla ln pi (A_t bar S_t, btheta_t) := 
  cases(
   bx(s,a) - sum_(a') pi (a' bar s, btheta) bx(s, a') "if using softmax",
   (a - mu(s, btheta))bx(s)/sigma^2 "if using Gaussian",
) 
$
// space!
#figure(image("../figures/REINFORCE.png", width: 100%),
)
// space!
$checkmark$ The expected update over an episode is in the same direction as the performance gradient. This assures an improvement in expected performance for
suciently small $alpha$, and convergence to a local optimum under *SSAC* for decreasing $alpha$ $crossmark$ As a *MC* method, REINFORCE may be of high variance and thus produce slow learning. \
// space! 
#underline[*REINFORCE w/ Baseline*] \
An approach to mitigate variance is to add a baseline $b(s)$ that is deducted from returns and does not change expectation.
$ 
theta_(t+1) eq theta_t + alpha [q_pi (S_t, A_t) - b(S_t)] nabla ln pi (A_t bar S_t, btheta_t) \
"typically: "b_(S_t) eq hat(v)(S_t)
$
The baseline $b(S_t)$ can be any function as long as t does not vary with $a$ as:
$ 
theta_(t+1) eq theta_t + alpha [q_pi (S_t, A_t) - b(S_t)] nabla ln pi (A_t bar S_t, btheta_t)  \
"adding back implicit expectation" \
theta_(t+1) eq theta_t + alpha [EE (q_pi (S_t, A_t) nabla ln pi (A_t bar S_t, btheta_t)) - \ EE (b (S_t) nabla ln pi (A_t bar S_t, btheta_t))] \ 
"we show that the second term above is 0 in expectation" \
EE (b (S_t) nabla ln pi (A_t bar S_t, btheta_t)) \
eq sum_a pi (a bar S_t, btheta) b(S_t) (nabla pi (a bar S_t, btheta)) / (pi (a bar S_t, btheta))  \ eq sum_a b(S_t) nabla pi (a bar S_t, btheta) eq b(S_t) nabla sum_a pi (a bar S_t, btheta) eq b(S_t) nabla 1 = 0 $
// space!
#figure(image("../figures/REINFORCE_with_baseline.png", width: 100%),
)
== Actor-Critic Methods
In *REINFORCE* w/ baseline, the learned state-value function estimates the value of the _first_ state of each state transition. In actor–critic methods, on the other hand, the state-value function is applied
also to the second state of the transition. That is, target becomes $G_(t:t+1)$ (like *TD(0)*). \ 
*Actor-Critic Update:*
// space!
$
theta_(t+1) eq theta_t + alpha [R_(t+1) + gamma hat(v)(S_(t+1), bw) - hat(v)(S_t, bw)] nabla ln pi (A_t bar S_t, btheta_t)
$
// space!
*Critic: * Gives the state-value function estimate $hat(v)(S_t, bw)$ \
// space!
*Actor: * Gives the policy estimate $pi(A_t bar S_t, btheta_t)$ \
// space!
The main appeal of one-step methods is that they are fully online and incremental. Note also that these methods lie between _policy-methods_ and _value-methods_. 
#figure(image("../figures/one_step_actor_critic.png", width: 100%),
)