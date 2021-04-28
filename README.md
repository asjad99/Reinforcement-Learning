### Optimal Decision Making under uncertainty 

#### probability: 

- http://www.datasciencecourse.org/notes/probability/ 


---------


### Markov decision Processes and Reinforcement Learning:  

We can model many problems as a Markov Decision Process or POMDP. We define a reward function that captures out goals and we then find a policy that maximuses the sum of future rewards. This is similar to Operations Research techniques focused on selecting the best element from a set of available alternatives to maximize a utility function. 


#### RL Theory: 
  - 3 pillers of reinforcement learning 
  - Exploration vs Exploitaion 
  - [RL in the Real World: Challenges and Opportunities](https://asjadkhan.ghost.io/real-world-rl/?fbclid=IwAR0jDaeMeALcSnEZu3gVq1MfZQegeXbWuWYt5W3PJNV1NiSePABsoAvS2EY)
  - Counterfactual Policy Evaluation  

[Combined notes] (https://paper.dropbox.com/doc/0.-Introduction--BFT_TqCOHWpWkWGTSgv3iWSwAg-8pHcTKBEfCeEqLMswL2w0)

[safe RL](https://paper.dropbox.com/doc/5.-Literature-Review-Safe-RL--BFQDQH9OuT5kQXiGRJm_uroAAg-5T9ss2yPag01K3m7J2pf8)

### Deep Learning and Reinforcement Learning 

Must know topics: 

Vocabulary: Know what states, actions, trajectories, policies, rewards, value functions, and action-value functions are. If you’re unfamiliar, Spinning Up ships with an introduction to this material; it’s also worth checking out the RL-Intro from the OpenAI Hackathon, or the exceptional and thorough overview by Lilian Weng. Optionally, if you’re the sort of person who enjoys mathematical theory, study up on the math of monotonic improvement theory (which forms the basis for advanced policy gradient algorithms), or classical RL algorithms (which despite being superseded by deep RL algorithms, contain valuable insights that sometimes drive new research).

Algorithms:

-  vanilla policy gradient (also called REINFORCE)
-  DQN
-  A2C (the synchronous version of A3C)
-  PPO (the variant with the clipped objective)
-  DDPG

The simplest versions of all of these can be written in just a few hundred lines of code (ballpark 250-300), and some of them even less (for example, a no-frills version of VPG can be written in about 80 lines). Write single-threaded code before you try writing parallelized versions of these algorithms. (Do try to parallelize at least one.)


### Sucess Stories: 




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

#### My Project: 
  - [Supporting Knowledge Instensive Processes in Clincial Settings]() 

#### Resources 
  - Spinning up in DeepRL 
  - [Introduction to deep RL by Lex Fridman](https://www.youtube.com/watch?v=zR11FLZ-O9M&feature=youtu.be)
  - [Long Peek into RL](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html)
  - [Math of RL](https://www.youtube.com/watch?app=desktop&v=LiaEmNToeQA&list=PLTPQEx-31JXhguCush5J7OGnEORofoCW9&index=17&t=0s)
  - [RL Course](https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver)
  - [Reinforcement learning](https://github.com/aikorea/awesome-rl)
  - [Deep Reinforcement Learning](https://spinningup.openai.com/en/latest/)
  - https://github.com/aikorea/awesome-rl
