= Temporal-Difference Learning 
*TD* sits between DP and MC method. Like MC methods, it learns directly from raw experience without knowing any environment dynamics. Like DP, it updates its estimates based in part on other learned estimates, without waiting for a final outcome  (*bootstraps*). 

== TD-Prediction (Policy Evaluation)
Whereas MC methods wait till the end of an episode before updating estimates (only then is $G_t$ known), TD methods only need to wait until the next time step. At time $t+1$, a target is formed using $R_(t+1)$ and estimate for $V(S_(t+1))$. The *TD(0) update* has the form:  
// space!
$ V(S_t) <- V(S_t) + alpha [R_(t+1) + gamma V(S_(t+1)) - V(S_t)] $
// space!
*TD(0)* is one-step TD and it can be used to evaluate a policy as follows:
// space!
#figure(image("../figures/td_policy_eval.png", width: 100%),
)
== TD Advantages and Comparisons
// space! 
*TD estimates as both DP and MC do*. 
1. *MC* estimates as the expected value $EE_pi [G_t bar S_t = s]$ is not known and a sample return is used in place of a real return.
2. *DP* estimates not because of expected values (it is assumed it has a full model), but as it uses $V$ to estimate the true $v_pi$ by bootstrapping. 
3. *TD* estimates as it samples expected values and bootstraps to find $v_pi$
// space!
Both *MC* and *TD* use _sample updates_ as opposed to _expected_ updates like *DP* since they both use a single sample successor during updates.  \ 
// space! 
*TD(0) Error: * $ delta_t eq.def R_(t+1) gamma V(S_(t+1)) - V(S_t) $ 
// space! 
*MC* errors can be given in terms of *TD(0)* errors as:  
$ G_t - V(S_t) eq.def sum_(k=t)^(T-1) gamma^(k-t)delta_k $ 
// space! 
*TD(0) Convergence to $v_pi$ with prob=1 if:*
All states visited infinitely often and *SSAC* met (see 2.1) \
#underline[*TD Advantages*] 
1. Like *MC*, a full model is not required 
2. Unlike *MC*, TD can be fully incremental (as it bootstraps) and thus requires less memory and computation 
3. *TD(0)* is shown to converge with any fixed policy $pi$ and in practice, it is shown to converge faster than *MC*. 
#underline[*Method Summary*]
#figure(table(
  columns: (auto, auto, auto),
  inset: 4pt,
  align: center,
  table.header(
    [*Method*], [*Model-Free?*], [*Bootstrap?*],
  ),
  "DP", 
  table.cell(fill: red, text("No")),
  table.cell(fill: green, text("Yes")),
  "MC", 
  table.cell(fill: green, text("Yes")),
  table.cell(fill: red, text("No")),
  "TD",
  table.cell(fill: green, text("Yes")),
  table.cell(fill: green, text("Yes")),
))
== On-Policy TD(0) Control: SARSA or SARSA(0) 
*Sarsa* uses the quintuples $(S_t, A_t, R_(t+1), S_(t+1), A_(t+1))$ (hence its name) for updates and its incremental update rule for action-value is given as:
// space!
$ Q(S_t, A_t) <- Q(S_t, A_t) + alpha [R_(t+1) + gamma Q(S_(t+1), A_(t+1)) - Q(S_t, A_t)] $
// space!
This update is done after every transition from a non-terminal state and if $S_(t+1)$ were terminal then $Q(S_(t+1), A_(t+1))$ is defined as zero. 
// space!
#figure(image("../figures/sarsa.png", width: 100%),
)
//space!
Convergence is guaranteed if all $(s,a)$ pairs visited infinitely + *SSAC* apply. 
== Off-Policy TD(0) Control: Q-Learning
Q-Learning uses the incremental update rule:
// space!
$ Q(S_t, A_t) <- Q(S_t, A_t) + alpha [R_(t+1) + gamma max_a Q(S_(t+1), a) - Q(S_t, A_t)] $
// space!
#figure(image("../figures/q_learning.png", width: 100%),
)
// space!
Convergence is guaranteed as long as all pairs continue to be updated and a variant of the usual stochastic approximation conditions apply. \ 
*Notes*:
1. The "$max_a Q(s', a)$" is to eventually derive the optimal target policy 
2. The behavioral policy for both TD algorithms uses $epsilon$-greedy to ensure that there is exploration
3. Notice how there is no *IS* in *Q-Learning*. This is because the $a$ in "$max_a Q(s', a)$" is not a random variable! The target policy is deterministic here. 
// space!
== Expected Sarsa (ESarsa)
This is a learning algorithm similar to *Q-Learning*, except it uses:
// space!
$ Q(S_t, A_t) <- Q(S_t, A_t) + alpha [R_(t+1) + gamma EE_pi [Q(S_(t+1), A_(t+1)) bar S_(t+1)] - Q(S_t, A_t)] \ 
  eq Q(S_t, A_t) + alpha [R_(t+1) + gamma sum_a pi(a bar S_(t+1))Q(S_(t+1), a) - Q(S_t, A_t)] $
// space!
That is, it picks the maximum over next state-action pairs through expectation, considering the likelihood of each action under the current policy. Given the next state, $S_(t+1)$, this algorithm moves _deterministically_ in the same direction as *Sarsa* moves _in expectation_ (hence the name). \ 
// space!
$checkmark$ reduces variance through random selection of $A_(t+1)$ $crossmark$ more computationally complex than *Sarsa* \
// space!
#underline[*Sensitivity to $alpha$*]\
The target in *ESarsa* is given as:
// space!
$ T_i = R_(t+1) + gamma sum_a pi(a bar S_(t+1)) Q(S_(t+1), a) $
// space! 
The target is deterministic, i.e. $T_i approx Q(S_t, A_t)$, no matter the step-size $alpha$. For *Sarsa*, as it samples $A_(t+1)$, it might have an action-value that is far from expectation and this is made worse with a larger step-size $alpha$. \
// space!
#underline[*On- or Off-Policy?*] \
*ESarsa* could be either. If it is _off-policy_ and the target policy is greedy + behavioral policy is $epsilon$-greedy then we exactly get *Q-Learning*. 
// space! 
== N-Step TD Prediction
This is a combination of both *MC* and *TD*. The difference is below:
// space!
$ 
  G_(t:t+1) &eq.def R_(t+1) + gamma V_t(S_(t+1)) " TD(0) 1-Step Return" \
  G_(t:infinity) &eq.def sum_(k+1)^infinity gamma^(k-1)R_(t+k) " MC Full Return" \
  G_(t:t+n) &eq.def sum_(k+1)^n gamma^(k-1)R_(t+k) + gamma^n V_(t+n-1)(S_t) " N-Step TD" \
$
// space!
That is, *N-Step TD* bootstraps over multiple steps.
// space!
#figure(image("../figures/n_step_backup.png", width: 100%),
)
// space!
The _n-step- update_ is still TD as it changes an earlier estimate based on how it differs from a later estimate. The later estimate now is just $n$-steps away. The incremental value function update is:
//space!
$ V_(t+n)(S_t) eq.def V_(t+n-1)(S_t) + alpha [G_(t:t+n) - V_(t+n-1)(S_t)] ", " 0 <= t < T $
All updates begin from time $t+n$, when $R_(t+n)$ is first known. Hence for the first $n-1$ steps, there are no changes made in each episode, but an equal number of additional updates are made at the end of the episode after termination but before starting on the next episode. \
// space!
*Note*: If $t+n >= T$, then $G_(t:t+n) = G_t$
// space!
#figure(image("../figures/n_step_td.png", width: 100%),
)
// space!
*Notes:*
1. All load and stores are done in a $N$-sized array, hence the mod
2. $G_(t:t+n)$ is calculated as long as $tau + n < T$. As soon as $tau + n = T$, exactly $n - 1$ more steps are are taken (till $tau = T - 1$). Within those steps, the missing rewards $R_(t+n)$ are just ignored ("corrected for"). 
*Error-Reduction Property: * The worst error of the expected _n-step_ return is guaranteed to be $<= gamma ^n$ times the worst error under $V_(t+n-1)$:
// space!
$ 
max_s bar EE_pi [G_(t:t+n) bar S_t = s] - v_pi (s) bar <= gamma ^n max_s bar V_(t+n-1) (s) - v_pi (s) bar forall n >= 1
$
// space!
Convergence of *n-step* is guaranteed given the _error reduction property_. 

== On-Policy N-Step for TD Control 
For *on-policy n-step*, the action-value will have the form: 
$
Q_(t+n)  &eq.def Q_(t+n-1) (S_t, A_t) + alpha [G_(t:t+n) - Q_(t+n-1)(S_t, A_t)] 
$
Only the future *return* term changes depending on the methods:
1. *N-Step On-Policy Sarsa*:
// space!
$
G_(t:t+n) &eq.def sum_(k=1)^n gamma^(k-1) R_(t+k) + gamma^n Q_(t+n-1)(S_(t+n), A_(t+n)), n>= 1, 0<=t<T-n 
$
// space!
where $G_(t:t+n) = G_t$ if $t+n >= T$.
// space!
#figure(image("../figures/n_step_sarsa.png", width: 100%),
)
// space!
2. *N-Step On-Policy ESarsa*:
// space!
$
G_(t:t+n) &eq.def sum_(k=1)^n gamma^(k-1) R_(t+k) + gamma^n overline(V)_(t+n-1) (S_(t+n), A_(t+n)), t+n < T
$
// space!
where $overline(V)$ is the *expected approximate value* of state $s$:
// space!
$ 
overline(V)_t (s) eq.def sum_a pi(a bar s) Q_t (s,a) "and " overline(V)_t ("terminal") = 0 
$
== Off-Policy N-Step for TD Control 
For *off-policy n-step*, the state-value will have the form: 
$
V_(t+n)  &eq.def V_(t+n-1) (S_t) + alpha rho_(t:t+n-1)[G_(t:t+n) - V_(t+n-1)(S_t)], 0<=t<T 
$
where the *IS ratio* is slightly modified to be:
$
rho_(t:h) eq.def product_(k=t)^(min(h, T-1)) (pi (A_k bar S_k)) / (b (A_k bar S_k ) )
$
Between *Sarsa* and *ESarsa*, only the *IS ratio* changes as shown:
// space!
1. *N-Step Off-Policy Sarsa*
// space!
$
Q_(t+n)  &eq.def Q_(t+n-1) (S_t, A_t) + alpha rho_(#highlight("t+1:t+n"))[G_(t:t+n) - Q_(t+n-1)(S_t, A_t)] 
$
// space!
2. *N-Step Off-Policy ESarsa*
// space!
$
Q_(t+n)  &eq.def Q_(t+n-1) (S_t, A_t) + alpha rho_(#highlight("t+1:t+n-1"))[G_(t:t+n) - Q_(t+n-1)(S_t, A_t)] 
$
(as all last possible actions are taken in account under $pi$ ) \
// space!
The *IS ratio* helps to balance out the importance of certain actions:
1. If $pi(a bar s) = 0$, then no weight is to that action and it is ignored
2. If $pi(a bar s) > b(a bar s)$,  then a higher weight is given to the action as it could be an action common in $pi$ but rare in $b$. If the weighting was greater in $b$, the opposite would hold true. 
3. If $pi(a bar s) = b(a bar s)$, the ratio is $1$ and there are no changes 
// space!
#figure(image("../figures/n_step_sarsa_off_policy.png", width: 100%),
)
// space!