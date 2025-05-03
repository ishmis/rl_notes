= Monte Carlo Methods 
*MC Methods* $eq.def$ methods that learn the value function based on experience. These do not require a complete model, only require sampled episodes. Put another way, MC is a way of solving the RL problem using average sample returns (_returns instead of rewards as we refer to rewards across episodes_). \
// space!
*Experience: * entire episodes $E^i eq <S_0^i, A_0^i, R_1^i, S_1^i, A_1^i, dots >$\
// space! 
*Assumptions of MC Algorithms*: 
1. Episodic tasks
2. Only on a completion of an episode are value estimates and policy updated (hence MC is incremental across episodes but not time steps)

== MC Prediction (Policy Evaluation)
With MC, the value of a state is estimated from experience by simply averaging all returns observed after visits to that state. As more results are observed, the average converges to the expected value. \ 
// space!
*Visit to s: * Each occurrence of state $s$ in an episode. \ 
// space!
*First-Visit MC: * estimate $v_pi (s)$ as the average returns following the first visit to $s$ across episodes. \ 
// space!
*Every-Visit MC: * estimate $v_pi (s)$ as the average returns following all visits to $s$ across episodes.\
// space!
The value function is estimated as below:
$ v_pi (s) eq.def EE_pi [sum_(k = t)^(T - 1) gamma^(k - t) R_(k+1) bar S_t = s] approx 1 / (bar cal(epsilon)(s) bar) sum_(t_i in cal(epsilon) (s)) sum_(k=t_i)^(T_i - 1) gamma^(k - t_i) R_(k+1)^i $
where $ cal(epsilon)(s) := cases(
  "contains first time" t_i "for which" S_(t_i)^i = s "in" E^i  "if First-Visit",
 "contains all times" t_i "for which" S_(t_i)^i = s "in" E^i  "if Every-Visit"
) $
Both methods converge to $v_pi (s)$ as $bar cal(epsilon)(s) bar -> infinity$, i.e. as the number of visits to all states reaches infinity.  
// space!
#figure(image("../figures/mc_pred.png", width: 100%),
)
// space!
*Note*:
1. Every-Visit is the above without the _"Unless $S_t$ appears ..."_ statement. 
2. Unlike DP, estimates for each state in MC are independent and there is no bootstrapping involved. 
3. Actual experience is learned from and not just simulation 
4. If only the value of a particular state is required, sample episodes could be generated starting from that state only to learn from. 
== MC Estimate of Action Value 
The state-value function alone is not sufficient without a model - there are no environment dynamics available. As such, it is necessary to look at action-value instead. However, now both a start state and action must be specified to begin an episode, and this may not be sufficient in ensuring that all state-action pairs are visited (infinitely for convergence). \
// space!
*Exploring Starts (ES): * every $(s,a)$ pair has a non-zero probability of being the starting pair of an episode. (_maintaining exploration_)\
// space! 
*Assumption of ES: * all state-action pairs will be visited an infinite number of times in the limit of an infinite number of episodes. \ 
// space!
== MC Control (GPI)
With alternating policy evaluation and improvement, under the assumption of *ES* and infinite episodes, the approximation of a value function and policy are guaranteed to converge to $q_pi$ and $pi$. \
As before, policy improvement is done by making the policy greedy w.r.t. the action-value function $q_pi (a, s) forall a, s$. The _policy improvement theorem_ applies as discussed before.  
// space!
#figure(image("../figures/mc_es.png", width: 100%),
)
Even-though convergence requires an $infinity$ number of episodes, in practice the _MC ES_ is only ran till a given performance threshold is met.
// space!
== On-Policy MC Prediction w/ $epsilon$-soft policy
*On-Policy: * attempt to evaluate or improve the
policy that is used to make decisions (i.e. from which experience is generated). \
// space!
*Off-Policy: * evaluate or improve
a _target policy_ $pi$ different from that used to generate the data (behavioral policy $b$). \
// space! 
The major issue with _MC ES_ is the assumption of _ES_. Instead of _ES_, another approach for exploration is to use a stochastic policy instead. \
*$epsilon$-soft policy: * a policy where $pi (a bar s) > 0, forall s,a. $.\
// space!
Now an *$epsilon$-soft policy* can be expressed as: \ 
// space!
$ pi(a bar s) := cases(
 1 - epsilon + epsilon / (bar A(S_t) bar) "if" a = A^*,
 epsilon / (bar A(S_t) bar) "if" a != A^*
) $
// space!
Each action has a minimum probability of $ epsilon / (bar A(S_t) bar)$ regardless of $epsilon > 0$. 
// space!
For _policy improvement_, note that now it is not possible to simply pick the greedy action as the policy must remain _soft_. Luckily, GPI does not require that the $epsilon$-soft policy $pi$ be taken all the way to a greedy policy, but only moved toward one.
// space!
#figure(image("../figures/mc_epsilon_soft.png", width: 100%),
)
// space!
To prove that an $epsilon$-greedy policy $pi'$ is an improvement over any $epsilon$-soft policy $pi$, consider this proof of the  _policy improvement theorem_:

// space! (don't remove the blank line above!)
$ q_pi (s, pi' (s)) &eq sum_a pi' (a bar s) q_pi (s, a)\
                    &eq epsilon / (bar A(s) bar) sum_a q_pi (s,a) + (1 - epsilon) max_a q_pi (s,a) \
                    &>= epsilon / (bar A(s) bar) sum_a q_pi (s,a) + (1 - epsilon) sum_a (pi (a bar s) - epsilon / (bar A(s) bar) ) / (1- epsilon) q_pi (s,a) \ 
                    &= sum_a pi (a bar s) q_pi (s, a) \
                    &eq v_pi (s)$\
// space! 
Step beginning $>=$ follows from $pi (a bar s) - epsilon / (bar A(S_t) bar) = 0,  forall a != A^*$. 
// space!
== Off-Policy MC Prediction w/ Importance Sampling 
The _on-policy_ approach before is a compromise, as it does not learn the optimal policy, but a near-optimal policy that still explores. The _off-policy_ approach maintains a *target policy* $pi$ that becomes optimal and a *behavioral policy* $b$ which only generates data and explores. \ 
// space!
*Assumption of Coverage: * Every action taken under $pi$ is also taken at least occasionally under $b$. That is, $pi (a bar s) > 0$ implies $b (a bar s) > 0$. \
// space! 
In order to meet the assumption, $b$ must be stochastic and $pi$ could be either, but is generally a deterministic greedy policy. \ 
// space!
*Importance Sampling (IS):* $eq.def$ Estimating expected values under one distribution given samples from another. \ 
// space!
*Trajectories* Given a starting state $S_t$, the probability of subsequent state-action trajectory under any $pi$ is: 
$  
  &Pr{A_t, S_(t+1), A_(t+1), dots bar S_t, A_(t:T-1) tilde pi } \ 
  &eq pi (A_t bar S_t)p(S_(t+1) bar S_t, A_t) pi (A_(t+1) bar S_(t+1)) dots \ 
  &eq product_(k=t)^(T-1) pi (A_k bar S_k)p(S_(k+1) bar S_k, A_k) 
$
// space!
*IS* is necessary as we estimate values under $pi$ given samples from $b$. \ 
// space!
*Importance Sampling Ratio: * The relative probability of a trajectory occurring under the target and behavior policies. It is given by: 
// space!
$ rho_(t:T-1) eq.def 
    (product_(k=t)^(T-1) pi (A_k bar S_k)p(S_(k+1) bar S_k, A_k)) / (product_(k=t)^(T-1) b (A_k bar S_k)p(S_(k+1) bar S_k, A_k))
    eq product_(k=t)^(T-1) (pi (A_k bar S_k)) / (b (A_k bar S_k ) ) $
// space!
The *IS* ratio only depends on the two policies and not $p(s', r bar s, a)$. Therefore, it is possible to transform the expected return under $b$ as follows: 
// space!
$ EE_b [rho_(t:T-1) G_t bar S_t = s]  \
    = sum_(E:S_t = s)[product_(k=t)^(T-1) b (A_k bar S_k)p(S_(k+1) bar S_k, A_k)] product_(k=t)^(T-1) (pi (A_k bar S_k)) / (b (A_k bar S_k ) ) G_t \
    = sum_(E:S_t = s)[product_(k=t)^(T-1) pi (A_k bar S_k)p(S_(k+1) bar S_k, A_k)] G_t 
    = v_pi (s) $
// space!
The approximation for the value functions is now just:
$ v_pi (s) approx eta^-1 sum_(t_i in cal(epsilon)(s)) p_(t_i : T_i) G_(t_i)^i bold("and") q_pi (s,a) approx eta^-1 sum_(t_i in cal(epsilon)(s,a)) p_(t_i + 1 : T_i) G_(t_i)^i $ 
// space!
Where $cal(epsilon)(s)$/$cal(epsilon)(s,a)$ denote the first/all times $t_i$ for which the $s$/$(s,a)$ occur in $E^i$. Note the $t_i + 1$ for action-value as the input $a$ is not present in the estimated trajectory from $(s,a)$.  The denominator $eta$ describes the which type of *IS* average is required: 
1. *Ordinary IS*: $eta = bar cal(epsilon)(s) bar$ or $eta = bar cal(epsilon)(s,a) bar$
2. *Weighted IS*:
$ eta = sum_(t_i in cal(epsilon)(s)) p_(t_i:T_i) bold("or") eta = sum_(t_i in cal(epsilon)(s)) p_(t_i + 1:T_i) $
Comparisons of *IS* type:
1. For when there is only a single-return under First-Visit MC, the _ordinary_ case gives $v_pi$ (unbiased) whereas _weighted_ gives the observed return under $b$ (*IS* ratio cancels out), which is biased! 
2. The _ordinary_ case's variance is unbounded on a single-return First-Visit MC, but the _weighted_ version is bounded to one.
3. On Every-Visit MC, both *IS* types are biased.\
// space!
== Incremental Implementation of Off-Policy MC
MC is incremental across episodes and not time steps. For the _ordinary_ case, it is only a matter of scaling w.r.t. $cal(epsilon)(s)$/$cal(epsilon)(s,a)$. For the _weighted_ case, the following estimate is needed:
// space!
$ V_n eq.def (sum_(k=1)^(n-1) W_k G_k)/(sum_(k=1)^(n-1)W_k) ", where" W_k = rho_(t_i:T(t_i) - 1) " and " n>=2 $
// space!
In addition to $V_n$, a cumulative sum $C_n$ of the weights given in the first $n$ returns is maintained to provide the following incremental updates:
// space!
$ 
V_(n+1) eq.def V_n + W_n/ C_n [G_n - V_n] "for" n>=1 "as" V_1 "is arbitrary"  \
C_(n+1) eq.def C_n + W_(n+1) "where" C_0 eq.def 0
$
// space!
#figure(image("../figures/mc_off_policy_pred_weighted.png", width: 100%),
)
// space!
#figure(image("../figures/mc_off_policy_gpi_weighted.png", width: 100%),
)
// space!
$crossmark$ A potential problem is that this method learns only from the tails of episodes, when
all of the remaining actions in the episode are greedy. \
$crossmark$ If nongreedy actions are common,
then learning will be slow, particularly for states appearing in the early portions of
long episodes.   \
#underline[*General Notes about Off-Policy*]
1. As data comes from a different policy, convergence is of greater variance and slower than on-policy.
2. Off-policy methods are more "powerful" and general. On-policy is a special case where the target and behavioral policy are the same. 
3. Ordinary importance sampling produces unbiased estimates, but has larger, possibly infinite, variance, whereas weighted importance sampling always has finite variance and is preferred in practice