### 


#Topics 

## Reinforcement Learning Fundamentals 
## Projects/Case Studies 

Reinforcement Learning is about Optimal Decision Making under uncertainty 

#### probability: 





Learn More: 
- http://www.datasciencecourse.org/notes/probability/ 


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


### Reinforcement in Real World 


#### RL Theory: 
  - [3 pillers of reinforcement learning](https://www.asjadk.io/untitled-4/) 
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


### Case Studies:: 


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
