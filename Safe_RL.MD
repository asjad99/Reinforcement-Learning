
### Topics: 
   a. Basics 
    b. state of affairs and open Problems
    c. Interesting Papers

-------------------------

1. Safe RL Survey: 

Safe Reinforcement Learning can be defined as the process of learning policies that maximize the expectation of the return in problems in which it is important to ensure reasonable system performance and/or respect safety constraints during the learning and/or deployment processes.

Optimization Criterion. As regards the first, the objective of traditional RL algorithms is to find an optimal control policy; that is, to find a function which specifies an action or a strategy for some state of the system to optimize a criterion. This optimization criterion may be to minimize time or any other cost metric, or to maximize rewards, etc.

we categorize these optimization criteria in four groups: (i) the worst-case criterion, (ii) the risk-sensitive criterion, (iii) the constrained criterion, and (iv) other optimization criteria.


- There can be two ways of modifying the exploration process:
    - Incorporate external knowledge (e.g by  “Learning from demonstrations”)
    - Risk directed exploration (incorporating soft and hard constraints and optimizing the policy following the constraints specified) 


![](https://paper-attachments.dropbox.com/s_ABBBC9E736CAB0F2584291B5AB61E0C1F4B3E4DA5F524AB9890408FF581191C6_1571709882916_Screen+Shot+2019-10-22+at+1.04.31+pm.png)

http://www.jmlr.org/papers/volume16/garcia15a/garcia15a.pdf


2. Safe Exploration by Philip 

He did his PhD on safe RL : https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf
talk: https://youtu.be/saHMbn84V_s
slides: https://web.stanford.edu/class/cs234/slides/2017/cs234_guest_lecture_safe_rl.pdf
 

 What is a safe RL?
![](https://paper-attachments.dropbox.com/s_ABBBC9E736CAB0F2584291B5AB61E0C1F4B3E4DA5F524AB9890408FF581191C6_1571707534785_Screen+Shot+2019-10-22+at+12.25.08+pm.png)


 

![](https://paper-attachments.dropbox.com/s_2F354B41B4A40F70196480248B40F8F4D9F06FD7FFAD49D18D8772FE0BC947DA_1566902386258_image.png)



- we are assuming we already have an initial policy and batch setting (instead of online)
- Safety is defined here as with probability atleast 1-δ, the current policy is not changed to a worse the the current policy, where δ represents the probability of choosing a bad policy(which user can choose). - low probability of failure would mean a log data would need to be collected by the algorithm before gureentee 
- a sample efficient safe RL algorithm is yet to be achieved.

 

    Why do need Safe RL algorithms? - Applications


- Intelligent Tutoring: Student learning about a topic (challenge is to give the problem at the right level of difficulty)
- Target advertising is a sequential decision making problem (sometimes treated as bandit problem). Safety here is not showing the wrong ads (incorrect policy)
- Diabetes Treatment is a control problem: how much insulin a person should inject prior to a meal to keep their blood sugar level near optimal without driving it too low or from keeping too high

  

![3 decision at 3 meals: CF can be tunned](https://paper-attachments.dropbox.com/s_ABBBC9E736CAB0F2584291B5AB61E0C1F4B3E4DA5F524AB9890408FF581191C6_1571708092232_Screen+Shot+2019-10-22+at+12.34.43+pm.png)

-   doc tells the patient which values of CF and CR will likely work for them (to be worked out again 3-6months) 
- can this process be automated? 
- Master thesis by Meysam Bastani at univeritsy of alberta:  https://era.library.ualberta.ca/items/fee1e7a7-1993-43f6-8d93-1d93855f6275 , https://www.amii.ca/intelligent-diabetes-management/
- RL problem: state is state of person, observation is current blood sugar level and size of the meal they are about to eat - the action is how much insulin a person should inject  - to be repeated 3 times a day(which is an episode)
- safety concerns: Default classic RL algorithms can’t guarantee that the policy they will produce will be better than the current policy e.g parameters chosen for diabates problem will be better than current ones
-  First time you run an RL algorithm it doesn’t run right - you need to tune it (only possible in simulated environment)

  

![](https://paper-attachments.dropbox.com/s_2F354B41B4A40F70196480248B40F8F4D9F06FD7FFAD49D18D8772FE0BC947DA_1566903176335_image.png)

- Let J(π) denote the performance of a policy (for now assume it means how good the policy is). Suppose you are in a casino, which policy would you want to follow? The red one or the blue one? The red one! It gives a more expected reward. But suppose the agent is a surgeon, the blue policy works better because it has lesser variance, i.e less chance it will screw up because low rewards are less probable.
  Creating a safe reinforcement learning algorithm
- we currently optimize for maximizm the expected return
- Risk-Sensitive Criterion
![lamda is scaling factor - try to get a policy that performs as well as possible and has low variance in the results it produces](https://paper-attachments.dropbox.com/s_ABBBC9E736CAB0F2584291B5AB61E0C1F4B3E4DA5F524AB9890408FF581191C6_1571721731399_Screen+Shot+2019-10-22+at+4.21.53+pm.png)



  

  3 steps(sub-problems) to developing safe RL algorithms 

  

----------

 

 Andreas Krause - Safe Exploration in Reinforcement Learning”: 

https://www.youtube.com/watch?v=5vMyz4HWYfw


- How can a learning system autonomously explore while guaranteeing safety ?  
- Safe exploration is an open problem in the reinforcement learning community and
- several definitions of safety have been proposed
- Safety here is being defined as avoiding the bad states/actions - but we don’t know them ahead of time - The goal is to identify the maximum safely reachable region starting from S0 without visiting any unsafe state
- The work is around the experiment  ‘Tuning the swiss free electron laser’ 

 Papers by Andreas Krause:

    https://las.inf.ethz.ch/files/berkenkamp17safe_model.pdf
    https://las.inf.ethz.ch/files/turchetta16safemdp.pdf



1. Counterfactual sequential reasoning 

off policy batch reinforcement learning to mine old data to make right decision for right context at scale 

Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning: 

https://arxiv.org/pdf/1805.09045.pdf


- we propose a new algorithm for tackling this performance prediction problem, which is called the offpolicy policy evaluation (OPE) problem

When Simple Exploration is Sample Efficient: Identifying Sufficient Conditions for Random Exploration to Yield PAC RL Algorithms:  https://arxiv.org/pdf/1805.09045.pdf

off Policy Batch Reinforcement Learning to mine old data to make right decision for right context at scale. 


----------
Representation Balancing MDPs for off-policy policy evaluation: 


https://www.youtube.com/watch?v=97wIrrCEkkM&


https://youtu.be/97wIrrCEkkM
 

----------
- Avoid bad state/actions
- But we don’t know the dynamics of the world ahead of time so we don’t know which actions are safe
- How can a learning system autonomously explore while guaranteeing safety ?  
- Safe exploration is an open problem in the reinforcement learning community and
- several definitions of safety have been proposed
- Safety here is being defined as avoiding the bad states/actions - but we don’t know them ahead of time - The goal is to identify the maximum safely reachable region starting from S0 without visiting any unsafe state



----------


Interesting Papers: 
----------

https://arxiv.org/pdf/1801.08757.pdf

Safe Exploration for Reinforcement Learning
https://www.tu-ilmenau.de/fileadmin/media/neurob/publications/conferences_int/2008/Hans-ESANN-08.pdf

1. Safe Policy Improvement with Soft Baseline Bootstrapping:  https://www.microsoft.com/en-us/research/publication/safe-policy-improvement-with-soft-baseline-bootstrapping/ (proposes safe and efficient Batch RL algorithms)

1. Multi-batch Reinforcement Learning: 
2. Learning Robust Options: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16825/16216
3. Lyapunov-based Safe Policy Optimization for Continuous Control 

https://www.youtube.com/watch?v=5vMyz4HWYfw
The work is around the experiment  ‘Tuning the swiss free electron laser’ 
Papers by Andreas Krause:

    https://las.inf.ethz.ch/files/berkenkamp17safe_model.pdf
    https://las.inf.ethz.ch/files/turchetta16safemdp.pdf
----------

https://arxiv.org/pdf/1801.08757.pdf

Safe Exploration for Reinforcement Learning
https://www.tu-ilmenau.de/fileadmin/media/neurob/publications/conferences_int/2008/Hans-ESANN-08.pdf
 

Tuning the swiss free electron Laser : https://icml.cc/media/Slides/icml/2019/seasideball(13-16-00)-13-17-10-5076-adaptive_and_sa.pdf
  
https://arxiv.org/pdf/1902.03229.pdf
Safe Policy optimization
Parametrs reward noisy reward
Unkown objectives, unknown constrainsts while gureenteeing feasabiloty along the way



How can a learning system autonomously explore while guaranteeing safety ?


- This problem is approached by either Changing the optimization criteria or by Changing the exploration processSafe Model-based Reinforcement Learning with Stability Guarantees

paper: https://arxiv.org/abs/1705.08551
slides: file:///Users/asjad/Dropbox/deeplearning2017_thomas_safe_rl_01.pdf
talk: https://youtu.be/Xwu38vQb9Gk
