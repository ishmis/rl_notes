= Dynamic Programming Methods
*DP* Methods compute optimal policies given a perfect model (MDP), thereby are a family of *planning* algorithms. *Planning* refers to the problem of learning a policy given a model. *DP* turn the Bellman Equations into update rules and all updates done in *DP* are called _expected_ updates because they are based on an expectation over all possible next states rather than on a sample next state. \
// space!
*Policy Evaluation (Prediction Problem): * Computing $v_pi$ from $pi$. \
// space!
*Policy Improvement: *Improving $pi$ from $v_pi$. Generally by making it greedy with respect to the value function of the original policy. 
// space! 
== Policy Evaluation 
The $v_pi$ Bellman Equation is turned into an update rule:
$ v_(k+1)(s) &eq.def EE_pi [R_(t+1) + gamma v_k (S_(t+1)) bar S_t = s] \
            &eq sum_a pi(a bar s) sum_(s',r) p(s', r bar s, a) [r + gamma v_k (s')] $ 
It is proven that $v_k$ converges to $v_pi$ as $k -> infinity$. 
== Policy Improvement
*Policy Improvement Theorem: * Let $pi$ and $pi'$ be two deterministic policies such that for all $s in S$, $q_pi (s, pi' (s)) >= v_pi (s)$. Then the policy $pi'$ must be as good as, or better than $pi$ - i.e. $v_(pi') (s) >= v_pi (s), forall s in S$. \
We can define a greedy policy as follows: 
$ pi'(s) &eq "argmax"_a q_pi (s, a) \
         &eq "argmax"_a EE [R_(t+1) + gamma v_pi (S_(t+1)) bar S_t = s, A_t = a] \ 
         &eq "argmax"_a sum_(s',r) p(s', r bar s, a) [r + gamma v_pi (s')] $ 
Once $pi'$ is as good as $pi$, $v_pi = v_(pi')$. It follows that: 
$ v_(pi') (s) &eq max_a EE [R_(t+1) + gamma v_(pi') (S_(t+1)) bar S_t = s, A_t = a] \ 
              &eq max_a sum_(s', r) p(s', r bar s, a) [r + gamma v_(pi') (s')] $
Where the above is exactly the Bellman Optimality Equation, so $v_(pi')$ must be $v_*$ and $pi'$ must be $pi_*$! \
// space!
*Stochastic Policies?*
Only deterministic policies discussed before but all also applies to stochastic policies $pi (a bar s)$. In the case of multiple actions having a maximum value, the stochastic policy could give an equal portion probability to all instead of selecting only one. 
// space !
== Policy Iteration 
#figure(image("../figures/policy_iteration.png", width: 100%),
)
*Note*: 
1. Only ever sweep through non-terminal states. 
2. Implicit is that we are updating incrementally, so $V_(k+1)$ on LHS!
3. Ties broken arbitrarily when making policy greedy (if deterministic)
4. PI often converges in a few iterations + optimality guaranteed!
== Value Iteration 
_Policy evaluation_ in _policy iteration_ requires multiple sweeps through the state set rather than being done iteratively. *Value iteration* truncates _policy evaluation_ (does not wait for it to converge) to a single sweep without losing converge guarantees. The combined policy improvement and truncated policy evaluation iterative step is:
$ 
v_(k+1) (s) &eq.def max_a EE [R_(t+1) + gamma v_k (S_(t+1)) bar S_t = s, A_t = a] \ 
              &= max_a sum_(s', r) p(s', r bar s, a) [r + gamma v_k (s')], forall s in S $
              
The sequence ${v_k}$ is shown to converge to $v_*$ under the same conditions that guarantee the existence of $v_*$.\
#figure(image("../figures/value_iteration.png", width: 100%),
)
// space!
#underline[*General Notes on the DP Methods:*]\ 
1. _Value Iteration_ uses the Bellman optimality equation for the value function while _Policy Iteration_ uses just the Bellman equation. 
2. Both methods formally require an infinite number of iterations to converge exactly to $v_*$, though in practice the iteration is stopped when changes are small between iterations. 
3. Both converge to an optimal policy for discounted finite MDPs. 
4. The entire idea of carrying out policy evaluation and improvement in a cycle is a general idea called *Generalized Policy Iteration* (GPI).
5. *DP* assumes that environment dynamics are completely known!
6. *DP* methods, by using Bellman (Optimality) Equations, *bootstrap* (create estimates based on other estimates).
#underline[*Asynchronous DP*]: 
1. Asynchronous *DP* methods evaluate and improve policy on subset of states instead of sweeping the entire state set. These algorithms are in-place iterative *DP* algorithms that can update the values of states in any order whatsoever. Convergence is guaranteed as long as all states continue to be updated an infinite number of times
3. Greater flexibility to choose best states to updates. For example, now real-time interaction is possible as the *DP* algorithm can be run _at the same time that an agent is actually experiencing the MDP_. Updates only need to be performed on states actually visited.
4. Can perform updates in parallel across multiple processors 