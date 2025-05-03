= Introduction 
Let random variables be denoted by $A$, $R$, $R_t$, $A_t$ and their realizations in lower-case. All sets will be: $SS$ or similar. \
*RL* $eq.def$ Learning to solve sequential decision problems via repeated interaction with environment \
// space!
*RL Problem* $eq.def$ Learning a high-value policy by interacting \ with an MDP. \ 
// space!
*Reward Hypothesis* $eq.def$ All goals can be described by the maximization of the expected value of cumulative scalar rewards. \
// space!
*Policy* $eq.def$ State to Action Map \
// space!
*Reward Signal* $eq.def$ Defines goal of RL. At each time step, environment sends the agent a reward. \
// space!
*Value Function* $eq.def$ Value of a state is the total amount of reward an agent can expect to accumulate in the future, starting from the state \
// space!
*Supervised* $eq.def$ Discover unknown function $f(x) = y$ given examples $(x, y = f(x))$ (RL does not get correct actions provided) \
// space!
*Unsupervised* $eq.def$ Discover hidden structure in data $x_1, x_2, x_3, ...$ (no $y$ given) (Rl gets a reward signal to inform correct action)\
// space!
*Key Challenges in RL*: \ 
1. Unknown Environment (how do actions affect state and rewards) 
2. Exploration-Exploitation Dilemma (try new $a$ or stick to best $a$)
3. Delayed Rewards (an $a$ may affect future rewards)