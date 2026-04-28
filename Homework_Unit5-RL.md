# Assignment: Reinforcement Learning 

## Assignment Overview

We work on a taxi game problem in this homework. The problem is described in https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/. 
We use OpenAI Gym to set up the game environment, and use Q-learning to train a smart agent that can drive the taxi to send a passenger to the destination location. 

Reference codes can be found at: 
Part 1: https://colab.research.google.com/drive/1esrOdkIb-U30V5potACczNXToZYdffDq?usp=sharing
Part 2: https://colab.research.google.com/drive/1gQ8ANgkI_kg66HPfKQ2V_0jFoWTlG7Ut?usp=sharing 


Note, if you see the error of installiation, please just rerun !pip install gym==0.15.7

<img width="283" alt="image" src="https://github.com/user-attachments/assets/7c664e55-f5f4-4de2-9b63-ff5233fe974e">

You may not have any errors.

<img width="381" alt="image" src="https://github.com/user-attachments/assets/a20de4f2-c326-4096-bcf2-d764fea3c897">


 
## Task 1: Taxi Game Understanding (40pt)

**Objective**: Run the code of Part 1, and understand the game (see the comments in the code for explanation).

### Instructions

1. **Number of States** (10 pts):
   
   (a) The current game setting has 500 states. If the number of pickup/destination locations is reduced to be 3. How many states are there?
   
   (b) In a real-world setting, passengers can make a request of pickup/destination at any place in a city. If in this game the number of pickup/destination locations is 25 (pickup/destination at any of the 25 locations), how many states are there?
   
2. **Solving the Taxi Game by an Untrainedv Agent (Random Policy)** (19pts): Run the code of Part 1 to solve the task:  taxi at (row=0, column=0), a passenger asks to be picked up at Y, and destination is G. Note that in this setting the agent solves the task by taking random actions without any learning.
   
   (a) (1 pts) How long (many steps) does it take? from the first move of the taxi, to the end of the task.
   
   (b) (1 pts) How many penalties incurred in the whole trip?

   (c) (1 pts) At which step, the taxi finally picked up the passenger (taxi block changes from yellow to green)?
   
   (d) (1 pts) Since the passenger sat on the taxi, after how many steps the passenger finally arrived?
   
   (e) (10pts) Change the code of Part 1, and calculate how much money the agent earned in this trip. (Hint: calculate the accumulated reward, it should be negative)

   (f) (5 pts)  Re-run the process of solving the same task:  taxi at (row=0, column=0), a passenger asks to be picked up at Y, and destination is G.  index=1 (G)).  Do you get the same solution (the agent drove the taxi in the same path)? why?
   
3. (5 pts)  Run 100 tasks (episodes).  On average for each task, the agent must take how many timesteps to finish the task? How many penalties received for each task?
   
4. (6 pts)  Change the code of Part 1, and calculate how much money the agent can earn  on average for each task (episode)?    
   

## Task 2: Solving the Taxi Game by Q-learning (60pt)

**Objective**: Run the code of Part 2, and understand well Q-learning for solving the game.

### Instructions

1. **Run Q-learning for 100000 episodes** (25 pts): 
   a)	**The number of epochs per eposode during  training** (5pts):  See the plot of the number of epochs in the 100000 training episodes. Explain why the curve shows a decrease of the number of epochs per episode. And in the last 1000 episodes, the number of epochs fluctuates around 15. Why?
   
   b) **The penalty curve during training** (5pts): Explain the penalty curve as well. Why the penalties are mostly 0 or 1, but sometimes can be 2, 3, 4, or even 5?
   
   c)	**Solving the taxi game by the trained model** (5pts): Use the trained model (the updated Q-table) to solve the task:  taxi at (row=0, column=0), a passenger asks to be picked up at Y, and destination is G. 
      i.	(1 pts) How long (many steps) does it take? from the first move of the taxi, to the end of the task.
      ii.	(1 pts) How many penalties incurred in the whole trip?
      iii.	(3 pts) How much money the agent earned in this trip. (Hint: calculate the accumulated reward) 

   d) **Check the Q-table** (2 pts): Check the Q-table we have, when the taxi starts at state=9, which action is the best to take? Is it reasonable to take it? 


   e)	**Try another task** (3 pts): Try taxi at (row=3, column=1), passenger at (B), destination is (R).   Check the Q-table, when the taxi starts at state=332, which action is the best to take? Is it reasonable to take it?
   
   f)	**Run 100 tasks (episodes)** (5 pts):  On average for each task, the agent takes how many timesteps to finish the task? How many penalties received for each task?  How much money the agent can earn on average for each task (episode)? 


2.	**Change the Hyperparameters** (35pts): change the Hyperparameters in Q-learning epsilon = 0.000001 and Run Q-learning only for 10000 episodes.

  	a)	(5 pts) plot the number of epochs again. Do you see any change, compared with the setting without change? Why?

  	b)	(5 pts) plot the penalty curve as well, and any change? compared with the setting without change. And why?

  	c)	(5 pt) Solve 100 tasks (episodes) with the trained Q-table.  There are tasks unsolvable within 1000 steps.  On average for each task, the agent takes how many timesteps to finish the task?

  	d)	(10 pts) If excluding those unsolvable tasks,  for the solved tasks, the agent takes how many timesteps to finish the task on average?

  	e)	(10 pts) Check the unsolvable tasks, and explain why they cannot be solved?
 
## Submission Guidelines

Submit the following items:

1. **Code**: Provide code for Tasks 1 and 2 in a Jupyter notebook or Python script. You can have two submission files, one for each task.
2. **Results**: The submission should include results and discussion.
   
