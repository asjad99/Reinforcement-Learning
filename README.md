## Topics 

- Reinforcement Learning Fundamentals 
- Math of Reinforcement Learning (planned)
- Classic Algorithms and Notebooks (planned)
- Reinforcement in Real World 
- Projects/Case Studies 

---------

## Fundamental Concepts: 

Reinforcement Learning is about learning to make good decisions under uncertainty. 


This can be seen as an Optimization problem where our aim is to cumulate as much reward as possible (while learning how the world works) i.e goal is to find an optimal way to make decisions

Mathematically goal is to maximize expected sum of discounted rewardsand Everything happens in a MDP setting.

### Markov decision Processes: 

We can model many problems as a Markov Decision Process or POMDP. We define a reward function that captures out goals and we then find a policy that maximuses the sum of future rewards.  This is similar to Operations Research techniques focused on selecting the best element from a set of available alternatives to maximize a utility function. 



### states, actions, trajectories, policies,
 

#### rewards
Its based on the reward Hypothesis which says: 
That all of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward).


####  value functions, and action-value functions 

A value function is an estimate of expected cumulative future reward, usually as a function of state or state-action pair. The reward may be discounted, with lesser weight being given to delayed reward, or it may be cumulative only within individual episodes of interaction with the environment. Finally, in the average-reward case, the values are all relative to the mean reward received when following the current policy.

The value-function hypothesis
all efficient methods for solving sequential decision-making problems compute, as an intermediate step, an estimate for each state of the long-term cumulative reward that follows that state (a value function). People are eternally proposing that value functions aren’t necessary, that policies can be found directly, as in “policy search” methods (don’t ask me what this means), but in the end the systems that perform the best always use values. And not just relative values (of actions from each state, which are essentially a policy), but absolute values giving an genuine absolute estimate of the expected cumulative future reward.


#### 1. Generalisation

Goal of RL(and exploration) algorithm is to perform well across a family of environments
- “Generalizing between tasks remains difficult for state of the art deep reinforcement
learning (RL) algorithms. Although trained agents can solve complex tasks, they struggle to transfer their experience to new environments. Agents that have mastered ten levels in a video game often fail catastrophically when first encountering the eleventh. Humans can seamlessly generalize across such similar tasks, but this ability is largely absent in RL agents. In short, agents become overly specialized to the environments encountered during training”
- if we don’t generalizations we can’t solve problems of scale, we need exploration with generalization” — ian osbond


#### 2. Exploration vs Exploitation
How can an agent decide whether to attempt new behaviours (to discover ones with higher reward) or continue to do the best thing it knows so far? or Given a long-lived agent (or long-running learning algorithm), how to
balance exploration and exploitation to maximize long-term rewards ?
This is know as the Exploration vs Exploitation which has been studied for about 100 years and still remains unsolved in full RL settings.
Exploration is about Sample efficient(or smart) data collection — Learn efficiently and reliably. Combining generalisation with exploration(below) is an active area of research e.g see Generalisation and Exploration via randomized Value functions.
An interesting question to ask here is:
Can we Build a rigorous framework for example rooted in information theory for solving MDPs with model uncertainty, One which will helps us reason about the information gain as we visit new states?


Online decision-making involves a fundamental choice:
Exploitation: Make the best decision given current information
Exploration: Gather more information The best long-term strategy may involve short-term sacrifices Gather enough information to make the best overall decisions

Two potential definitions of exploration problem:
How can an agent discover high-reward strategies that require a temporally extended sequence of complex behaviours that, individually, are not rewarding?
How can an agent decide whether to attempt new behaviours (to discover ones with higher reward) or continue to do the best thing it knows so far? (from keynote ERL Workshop @ ICML 2019)
Techniques:
There are a variety of exploration techniques some more principled than others.
Naive exploration — greedy methods
Optimistic Initialisation — A very simple idea that usually works very well
Optimism under uncertainty — prefer actions with uncertain values eg. UCB
Probability matching — pick action with largest probability of being optimal eg. Thompson sampling
Information state space — construct augmented MDP and solve it, hence directly incorporating the value of information
They are studied for mulit-armed bandits context (because we understand these settings well):
I recommend watch this lecture for more details: youtube.com/watch?v=sGuiWX07sKw
State of Affairs and Open Problems we look at three questions asked at ERL Workshop @ ICML 2018 panel discussion
How important is RL in exploration?
In this debate, There are two camps — one thinks exploration is super important and you have to get it right — its hard to build intelligent systems without learning and getting good data is important
the other camp thinks there will be so much data that getting the right data doesn’t matter — some exploration will be needed but the technique doesn’t matter.
we take the pendulum example where the task is to balance the pendulum in an upright position.

small cost 0.1 for movingreward when its upright and in the middle
state of the art deep RL uses e-greedy exploration takes 1000 episodes and learns to do nothing. which means default exploration doesnt solve the problem. it requires planning in a certain sense or ‘deep exploration’ (branding done by ian osband) techniques gets there and learn to balance by the end of 1000 episodes.

key lesson: exploration is key to learning quickly(off-policy) so its a core component but some problems can be solved without exploration. e.g driverless cars
State of the art — Where do we stand ?
Exploration is not solved — we don’t have good exploration algorithms for the general RL problem — we need a principled/scaleable approach towards exploration.
Theoretical analysis/tractability becomes harder and harder as we go from top to bottom of the following list
multi-armed bandits ( multi-armed bandits with independent arms is mostly solved )
contextual bandits
small, finite MDPs (exploration in tabular settings is well understood )
large, infinite MDPs, continuous spaces

In many real applications such as education, health care, or robotics, asymptotic convergence rates are not a useful metric for comparing reinforcement learning algorithms. we can’t have millions of trails in the real world and we care about convergence and we care about quickly getting there. To achieve good real world performance, we want rapid convergence to good policies, which relies on good, strategic exploration.
other important open research questions:
how do we formulate the exploration problem ?
how do we explore at different time scales?
safe exploration is an open problem
mechanism is not clear — planning or optimism?
generalisation (identify and group similar states) in exploration?
how to incorporate prior knowledge?
How do we assess exploration methods? and what would it mean to solve exploration?
In General, we care about algorithms that have nice PAC bounds and regret but they have limitations where expected regret make sense — we should aim to design algorithms with low beysien regret. We know Bayes-optimal is the default way to asset these but that can be computationally intractable so literature on efficient exploration is all about trying to get good beyes regret but also maintain statistical efficiency
But how do we quantify notions of prior knowledge and generalisation? Which is why some experts in the area think its hard to pin down the definition of exploration. i.e when do we know when we will have but its not obvious
At the end of the day, we want intelligent directed exploration which means behaviour should be that of learning to explore . i.e gets better at exploring and there exists a Life long — ciricula of improving.
Scaling RL (taken from [1] and [2]):
Real systems, where data collection is costly or constrained by the physical context, call for a focus on statistical efficiency. A key driver of statistical efficiency is how the agent explores its environment.To date, RL’s potential has primarily been assessed through learning in simulated systems, where data generation is relatively unconstrained and algorithms are routinely trained over millions to trillions of episodes. Real systems, where data collection is costly or constrained by the physical context, call for a focus on statistical efficiency. A key driver of statistical efficiency is how the agent explores its environment. The design of reinforcement learning algorithms that efficiently explore intractably large state spaces remains an important challenge. Though a substantial body of work addresses efficient exploration, most of this focusses on tabular representations in which the number of parameters learned and the quantity of data required scale with the number of states. The design and analysis of tabular algorithms has generated valuable insights, but the resultant algorithms are of little practical importance since, for practical problems the state space is typically enormous (due to the curse of dimensionality). Current literature on ‘efficient exploration’ broadly states that only agents that perform deep exploration can expect polynomial sample complexity in learning(By this we mean that the exploration method does not only consider immediate information gain but also the consequences of an action on future learning.) This literature has focused, for the most part, on bounding the scaling properties of particular algorithms in tabular MDPs through analysis. we need to complement this understanding through a series of behavioural experiments that highlight the need for efficient exploration. There is a need for algorithms that generalize across states while exploring intelligently to learn to make effective decisions within a reasonable time frame.

#### 3. Credit assignment

The Structure of RL looks at one more problem which is known as credit assignment. Its important that we have to consider effects of our actions beyond a single timestep. i.e How can an agent discover high-reward strategies that require a temporally extended sequence of complex behaviours that, individually, are not rewarding?

- “RL researchers (including ourselves) have generally believed that long time horizons would require fundamentally new advances, such as hierarchical reinforcement learning. Our results suggest that we haven’t been giving today’s algorithms enough credit — at least when they’re run at sufficient scale and with a reasonable way of exploring.” — OpenAI
An efficient RL agent must address these above challenges simultaneously.
To summarize we want algorithms that perform optimization, handle delayed consequences, perform intelligent exploration of the environment and generalize well outside of the trained environments. And do all of this statistically and computationally efficiently.




### classical RL algorithms


![image](https://user-images.githubusercontent.com/3470924/118244172-9b893080-b4e2-11eb-88b9-399b8e55065b.png)



### Value Functions: 

Its not a solved problem:
“Design Algorithms that have generalisation and sample efficiency in learning to make complex decisions in complex environments” — Wen Sun


Given a policy we want to compute expected return from a given state by computing the value function the simple idea of monte-carlo is that we compute mean return for many episodes and keep track of the number of times that state is visited using expected law of large numbers <br><br> and we dont wait till the end — we update step by step(little bit) in the direction of the error i.e we update the value function by computing the difference between the estimated value function and the actual return at the end of each episode

Value function isn’t the only way to solve the RL problem. We have 2 other approaches as well which we will discuss in coming posts, namely model based and policy based. See figure above for the big picture view.

![image](https://user-images.githubusercontent.com/3470924/118244742-3b46be80-b4e3-11eb-8e2e-68d8bb231d5d.png)


---------


### Reinforcement in Real World 


  - [Exploration vs Exploitaion](https://www.asjadk.io/strategic-exploration-in-online-decision-making/)
  - [RL in the Real World: Challenges and Opportunities](https://asjadkhan.ghost.io/real-world-rl/?fbclid=IwAR0jDaeMeALcSnEZu3gVq1MfZQegeXbWuWYt5W3PJNV1NiSePABsoAvS2EY)
  - [Counterfactual Policy Evaluation](https://www.asjadk.io/counterfactual-policy-evaluation/)  
  - [safe RL](https://github.com/asjad99/Deep-Reinforcement-Learning/blob/main/Safe_RL.MD)




[Combined notes] (https://paper.dropbox.com/doc/0.-Introduction--BFT_TqCOHWpWkWGTSgv3iWSwAg-8pHcTKBEfCeEqLMswL2w0)


----


### Deep Learning and Reinforcement Learning 

[What is DeepRL - David Silver](https://youtu.be/MrIFte_rOh0)

Must know topics: 

Vocabulary:

If you’re unfamiliar, Spinning Up ships with an introduction to this material; it’s also worth checking out the RL-Intro from the OpenAI Hackathon, or the exceptional and thorough overview by Lilian Weng. 

math of monotonic improvement theory (which forms the basis for advanced policy gradient algorithms)



Algorithms:

-  vanilla policy gradient (also called REINFORCE)
-  DQN
-  A2C (the synchronous version of A3C)
-  PPO (the variant with the clipped objective)
-  DDPG

The simplest versions of all of these can be written in just a few hundred lines of code (ballpark 250-300), and some of them even less (for example, a no-frills version of VPG can be written in about 80 lines). Write single-threaded code before you try writing parallelized versions of these algorithms. (Do try to parallelize at least one.)

---------


### Case Studies/Projects: 


#### My Project: 
  - [Supporting Knowledge Instensive Processes in Clincial Settings]() 


#### Deep Q-Network (DQN):
    - For artificial agents to be considered truly intelligent they should excel at a wide variety of tasks that are considered challenging for humans. Until this point, it had only been possible to create individual algorithms capable of mastering a single specific domain. 
    - Successfully combining Deep Learning (processing perception) with RL (decision-making) at scale for the first time 
    - is able to master a diverse range of Atari 2600 games to superhuman level with only the raw pixels and score as inputs.
    - Takes the raw inputs and a reward signal (e.g the score) and it has to then figure out everything else. i.e transform a vector of image pixels into a policy for selecting actions 
    -  They leveraged recent breakthroughs in training deep neural networks to show that a novel end-to-end reinforcement learning agent, termed a deep Q-network (DQN) was able to surpass the overall performance of a professional human reference player and all previous agents across a diverse range of 49 game scenarios
    - One Key ingredient is experience replay - where by a network stores a subset of the training data in an instance-based way and then replays it offline, learning anew from successes or failures that occurred in the past. 
    - This work represents the first demonstration of a general-purpose agent that is able to continually adapt its behavior without any human intervention, a major technical step forward in the quest for general AI.


#### AlphaGo:

   - Enormous Search Space and impossible to write evaluation function
    - Top players use intution and insticts rather than calculuation like in chess
        Components:
        
   - Monte Carlo Tree Search (certain variant with PUCT function for tree traversal):https://int8.io/monte-carlo-tree-search-beginners-guide/
    - Residual Convolutional Neural Networks – policy and value network(s) used for game evaluation and move prior probability estimation
    - Policy Network tries  to predict the move that human was going to
    -     play by training on 100k human games downloaded from internet
    - After training it can output probability distribution of possible moves
    - we can look at top 5 moves 
    - trained the second neural network(called the value net) to predict from the current position who is likely to be a winner 
    - trained on the data produced by first network playing against itself many millions of times 
    - Reinforcement learning used for training the network(s) via self-plays

#### AlphaGo Zero: 

   - Latest and greatest version of AlphaGo
    - Fully automated pipeline - No bootstrapping from human data 
    - Starts from completely random play with 'zero knowledge'
    - plays against itself millions of times
    - Learns incrementally from its own mistakes
    - Even Stronger, more efficient and more general

#### Alpha Zero: 
   - Trained on three perfect information games (chess, shogi and Go)
    - Chess engines are highly specialized systems using a whole bag of handcrafted hurrisitics, extensions and domain knowledge
    - Alpha Zero replaces all of that with Self-play reinforcement Learning + Self-Play Monte Carlo Tree Search 
    - No Openbook or endgame database or Heuristics 
    - Starts from random play 
    - Same Algorithm with Same hyperparameters for all 3 games
    - Proof that system is very general and that learning systems could be better than hand-crafted systems

#### Beating Professional Dota 2 Players: 

OpenAI created a bot which beats the world’s top professionals at 1v1 matches of Dota 2 under standard tournament rules. 
“The bot learned the game from scratch by self-play, and does not use imitation learning or tree search. 
This is a step towards building AI systems which accomplish well-defined goals in messy, complicated situations involving real humans.”
 https://blog.openai.com/dota-2/


#### Resources 
  - Spinning up in DeepRL 
  - [Introduction to deep RL by Lex Fridman](https://www.youtube.com/watch?v=zR11FLZ-O9M&feature=youtu.be)
  - [Long Peek into RL](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)
  - [Math of RL](https://www.youtube.com/watch?app=desktop&v=LiaEmNToeQA&list=PLTPQEx-31JXhguCush5J7OGnEORofoCW9&index=17&t=0s)
  - [RL Course](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver)
  - [Reinforcement learning](https://github.com/aikorea/awesome-rl)
  - [Deep Reinforcement Learning](https://spinningup.openai.com/en/latest/)
  - https://github.com/aikorea/awesome-rl
