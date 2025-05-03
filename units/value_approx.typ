#let bw = $bold(w)$
#let bx = $bold(x)$
#import "@preview/diverential:0.2.0": *
= Value Function Approximation
*Curse of Dimensionality:* The number of states grow _exponentially_ with the number of state variables $k$, i.e. $O(n^k)$ states.\
// space!
Approximation allows us to avoid the above and does not assume 1. we have _unlimited space_ and 2. _unlimited data_.* We first look at on-policy.*\
// space!
*Idea*: Use function approximation of the $v_pi$ by representing it in a parametrized functional form with a weight vector $w in RR^d$: 
$ hat(v)(s, bw) approx v_pi (s) $
The weight vector $bw$ could represent feature weights, connection weights of a neural net, or it could be numbers defining the split points and leaf values of a decision tree. Typically $d < < bar S bar$ and changing one weight changes estimated values of many states (thus generalizing). \
// space!
#underline[*Advantages of Approximation*]
1. If $hat(v)$ does not capture certain aspects of a state, it is almost like those aspects are not observed. Hence, functional approximation allows RL to apply to partially observable problems too! 
2. Tabular methods require storage proportional to $bar S bar$ or $bar S bar * bar A bar$ - $hat(v)$ does not and is *compact*.
3. Other methods suffer from not having any or having insufficient data for estimation. $hat(v)$ *generalizes* to unknown $(s,a)$ as a fix. 
// space!
#underline[*Supervised Learning*]\
Learning a value function from (input, output) pairs is a form of supervised learning which RL also utilizes. Consider pairs of states and their update targets $(S_t, U_t)$ we have looked at so far:
1. *MC*: $U_t = G_t$ 
2. *TD(0)*: $U_t = R_(t+1) + gamma hat(v)(S_(t+1), bw)$ 
3. *n-step TD*: $U_t = sum_(k=1)^(n-1) gamma^(k--1)R_(t+1) + gamma^n hat(v) (S_(t+n), bw_(t+n-1))$
The desired properties of supervised learning are:
1. *Incremental Updates* (update $bw$ using only partial data)
2. *Ability to handle noisy targets*
3. *Ability to handle non-stationary targets*
// space!
If $hat(v)$ or $hat(q)$ were differentiable, *stochastic gradient descent (SGD)* suits. 

// space!
#underline[*Prediction Objective*\ ]
Important to clarify as our assumption is that there are far more states than weights, so making one state's estimate more accurate invariably means making other's less accurate \ 
Let $mu(s) >= 0, forall s in S$ represent how much we "care" about a particular state. Now, we consider the error between $hat(v)(s, bw)$ and $v_pi (s)$ as:  \
*Mean Squared Error :* A continuous measure of prediction quality.
$ 
overline("VE")(bw) eq.def sum_s mu(s) [v_pi (s) - hat(v)(s, bw)]^2
$
// space!
Often $mu(s)$ is chosen as the fraction of time spent in $s$ (i.e. _on-policy distribution_). Goal is to find $bw^*$ that minimizes $overline("VE")$. Finding a _global optimum_ that does so is difficult, so often it is sufficient to seek to converge to just a _local optimum_. 
== Stochastic Gradient and Semi-Gradient Methods
Let $bw$ be a column vector of weights and let $hat(v)(s, bw)$ be a differentiable function of $bw, forall s in S$. Then, _gradient descent_ can be used to find the _local minimum_ iteratively as follows: 
// space!
$ 
bw_(t+1) &eq.def bw_t - 1/2 alpha nabla [v_pi (S_t) - hat(v)(S_t, bw_t)]^2 \
         &eq w_t + alpha [v_pi (S_t) - hat(v)(S_t, bw_t)]nabla hat(v)(S_t, bw_t)
$
// space!
where $alpha > 0$ and $nabla f(bw)$ denotes a column vector of partial derivatives:
// space!
$ 
nabla f(bw) eq.def ( dvp(f(bw), bw_1), dvp(f(bw), bw_2), dots, dvp(f(bw), bw_d) ) 
$
// space!
_Gradient descent_ methods are stochastic and at each step, minimize $overline("VE")$. Convergence of *SGD* assumes that $alpha$ decreases over time + *SSAC* apply.  \
However, $v_pi (S_t)$ is unknown, so use expected _update target_ $U_t$: 
// space!
$ 
bw_(t+1) &eq w_t + alpha [U_t - hat(v)(S_t, bw_t)]nabla hat(v)(S_t, bw_t)
$
// space!
If $U_t$ is unbiased, i.e. $EE[U_t bar S_t = s] = v_pi (s)$, then $bw_t$ is guaranteed to converge to a local optimum given stochastic approximation conditions on a decreasing $alpha$. *SGD* is optimal for *MC* as they produce an unbiased estimate of $v_pi$. For *bootstrapping* methods it is biased:
1. *TD* target $U_t eq R_(t+1) + gamma hat(v) (S_(t+1), bw)$
2. *DP* target: $U_t eq sum_(a, s', r)pi (a bar S_t)p(s', r bar s, a)[r + gamma hat(v)(s', bw)]$
3. *N-Step target*: $U_t eq sum_(k=1)^n gamma^(k-1)R_(t+k) + gamma^n hat(v) (S_(t+n), bw_(t+n))$
All the above use $bw$ and ignore its effect on the target. Hence, they are called *semi-gradient methods* as it is not true *gradient descent*. 
// space!
#figure(image("../figures/mc_sgd_eval.png", width: 100%),
)
// space!
#figure(image("../figures/semi_gradient_td_eval.png", width: 100%),
)
// space!
#underline[*State Aggregation*]\
It is a simple form of generalizing function approximation in which states are grouped together, with one estimated value for all states in a group. The value of a state is estimated as its group's component, and updated when the state is. _State Aggregation_ is a special case of *SGD* in which the gradient $nabla hat(v) (S_t, bw_t) = 1$ for $S_t$'s group's component and 0 for the rest.\ 
// space!
#underline[*Linear Methods*]\
Where $hat(v)$ is a linear function:
// space!
$ 
hat(v)(s, bw) eq.def bw^T bx(s) eq.def sum_(i=1)^d bw_i bx_i (s) 
$
// space!
Where $bx(s)$ is the _feature vector_ representing $s$. In the linear case, the features form a _linear basis_ for the set of approximate functions. The linear case has the following important result:
// space!
$ nabla hat(v) (s, bw) eq bx (s) $
// space!
Hence the *SGD* update for the linear case is simply:
$ 
bw_(t+1) &eq w_t + alpha [U_t - hat(v)(S_t, bw_t)]bx(S_t)
$
Moreover, there is only a single optimum in the linear case, so linear methods, when they converge, converge to or near the _global optimum_. The *MC* gradient updates converge to the _global_ optimum, whereas the *TD* gradient update converge _near global optimum_. 
// space!
== Coarse/Tile Coding 
Both come under _feature construction_ for linear methods and both allow for generalization across disjoint sets of states unlike _state aggregation_ with generalizes within a particular set. _Note_ that linear methods cannot take into account the interaction between features! \ 
#underline[*Coarse Coding*]\
Assuming $S in RR^2$ + binary features, the state space is represented by *overlapping circles* corresponding to features. Each state falls within some number of circles. Any update to the state, updates the features corresponding to those circles and thereby all other states in those circles. \
#underline[*Tile Coding*]\
These allow _coarse coding_ for multi-dimensional spaces. The state space is split into partitions, where each partition is called a _tiling_ and each element of the partition is a _tile_. Each state has exactly the same number of features active as the number of tilings (each _tile_ can be thought of as a grid of features). $bx(s)$ here has one component for each tile. \
Generalization occurs to states other than the one trained if those states fall within any of the same tiles, proportional to the number of states in common.
== On-Policy Control in Episodic Tasks w/ Approximation
State-action values estimated by: $hat(q)(s,a,bw) approx q_pi (s,a)$. \ 
// space!
*Linear approximation*: $hat(q)(s,a,bw) eq.def sum_(i=1)^d bw_i bx_i (s, a)$ \
// space!
*SGD* with action-value: 
// space!
$ 
bw_(t+1) &eq w_t + alpha [U_t - hat(q)(S_t, A_t, bw_t)]nabla hat(q)(S_t, A_t, bw_t)
$
// space!
Update targets for *TD* control methods as below:
1. *Sarsa: * $U_t eq R_(t+1) gamma hat(q) (S_(t+1), A_(t+1), bw_t)$
2. *Q-Learning: * $U_t eq R_(t+1) gamma max_a hat(q) (S_(t+1), a, bw_t)$
3. *ESarsa: * $U_t eq R_(t+1) gamma sum_a pi (a bar S_(t+1))hat(q)(S_(t+1), a, bw_t)$
// space!
#figure(image("../figures/semi_gradient_sarsa.png", width: 100%),
)
// space!
#underline[*Convergence to Global Optimum in Episodic Control*]
#figure(table(
  columns: (auto, auto, auto, auto),
  inset: 4pt,
  align: center,
  table.header(
    [*Algorithm*], [*Tabular*], [*Linear*], [*Non-Linear*]
  ),
  "MC Control", 
  table.cell(fill: green, text("Yes")),
  table.cell(fill: gray, text("Chatter")),
  table.cell(fill: red, text("No")), 
  "Semi-Gradient n-step Sarsa", 
  table.cell(fill: green, text("Yes")),
  table.cell(fill: gray, text("Chatter")),
  table.cell(fill: red, text("No")), 
  "Semi-Gradient n-step Q-Learning", 
  table.cell(fill: green, text("Yes")),
  table.cell(fill: red, text("No")),
  table.cell(fill: red, text("No")), 
))
where _chatter_ refers to being near optimal because optimal policy may not be representable under value function approximation \
#underline[*Deadly Triad*] \
Refers to *risk of divergence* rising when the following three are combined:
*Function Approximation* + *Bootstrapping* + *Off-Policy Learning* \
#underline[Possible fixes:]
1. Use *IS* to warp off-policy distribution into on-policy distribution
2. Use gradient TD methods which follow true gradient of projected Bellman error