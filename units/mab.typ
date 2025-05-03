#let lefttext(x) = { align(left, text(x)) }
#let codeblock(x) = {
  align(center,
  block(
  fill: luma(230),
  inset: 5pt,
  radius: 4pt,
  lefttext(x))
  )
}
= Multi-Arm Bandits
*Definition* (Think MDP with only one state):
#codeblock[
  *Given*: a set of _k_ actions, $A$, number of rounds $T$: \
  Repeat for $t$ in $T$ rounds:
    1. Algorithm selects arm $A_t$ in $A$
    2. Algorithm observes reward $R_t in [0,1]$
  *Goal*: maximize expected total reward.
]
*Note*: No role of state in MAB + over set number $T$! \
// space!
*Value of Arm $a$:* 
$
q_*(a) eq.def EE[R_t | A_t = a ] "(cannot get directly)"
$
// space!
*Sample Average Estimate of Action-Value: *
$ 
Q_t(a) eq.def 1/(N_t (a)) * sum_(i=1)^(t-i)R_i * bb(1)_(A_t=a)\
"(sum of rewards when" a "taken/num times "a "taken, prior to "t")"
$
// space!
*Incremental Update: *$ Q_(t+1) eq.def Q_t + 1/n [R_n - Q_n] $
*Standard Form of Update:* $ "NewEstimate" <- "OldEstimate + StepSize[Target - OldEstimate]" $ 
*Error* $eq.def$ Target - OldEstimate
// space!
#figure(
  image("../figures/simple_mab_alg.png", width: 100%),
)
The action selection above is $epsilon$-greedy exactly, which is useful when there are noisy rewards and no one action is always best. The above alg only applies to *stationary problems* (where reward probabilities don't change). 
// space!
== Non-Stationary MAB
*NS* = non-stationary \
// space!
*Incremental Update :* 
$
Q_(t+1) eq.def Q_t + alpha [R_n - Q_n] \
"where" alpha "is the step-size parameter (learning rate) given by" alpha_t (a)
$ 
// space!
#underline[*Standard Stochastic Approximation Conditions (SSAC)*]: \ Estimates $Q_n$ converge to $q_*$ with Pr=1 if:\ 
// space!
$ sum_(n=1)^(infinity)alpha_n (a) -> infinity "and " sum_(n=1)^(infinity)alpha_n^2(a) < infinity $
// space!
However, in *NS* problems, $alpha_n (a)$ that do converge (e.g. $1/n$), do so very slowly and need tuning to speed up convergence. It is desirable to use $alpha_n (a)$ that do not meet the conditions (e.g. constant $alpha$) in *NS* problems. 
// space!
== Increasing Exploration in MAB
1. *Optimistic Initial Values*
The initial action-value estimate $Q_1(a)$ _biases_ MAB. Setting a high initial action value instead of just 0 encourages exploration. The learner is 'disappointed' by the lower actual reward and switches to other actions, but this results in several actions being tried before value estimates converge. \ $checkmark$ Simple  $crossmark$ Only useful in stationary problems as all exploration is only ever done initially.
2. *Upper-Confidence-Bound (UCB) Action Selection*
$ A_t eq.def "argmax"_a [Q_t (a) + c sqrt(ln t / N_t (a))] $  $c > 0$ and controls the degree of exploration. Here actions are picked by highest bound.  $checkmark$ Actions picked because either algorithm is uncertain about an action (never picked or picked long ago, scaling with unbounded $ln t$) or it can exploit a high value action. Hence all actions eventually get selected but actions with lower value estimates or those selected frequently, get selected with a decreasing frequency over time. $checkmark$ Performs better than $epsilon$-greedy.
3. *Gradient-Bandit Algorithms* 
This is a policy-based approach and does not focus on a value function. \
*Preference: * Each action is assigned a scalar $H_t (a) in RR$. \ A differentiable policy $pi_t (a | theta)$, where $theta in RR^d$ (policy parameters), is utilized such that _gradient ascent_ could be performed: 
$ theta_(t+1) eq.def theta_t + alpha nabla_(theta_t)EE[R_t] 
$ 
The policy itself could be represented by a softmax distribution $ pi_t(a) eq.def Pr{A_t = a} eq.def e^(H_t (a)) / (sum_(b=1)^k e^(H_t (b))) $
*Preference Update: *$ H_(t+1) (A_t) eq H_t (A_t) + alpha (R_t - overline(R_t))(1 - pi_t (A_t)) $ where $overline(R_t) eq 1/t sum_(i=1)^t R_i$ is the average reward.