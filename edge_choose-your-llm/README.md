# choose-your-llm
Choose the best-performing open source LLM that suits your local environment.

Goal: Develop a general framework for edge users to select the best small LLM model that suits their specific needs and constraints
 
- [x] Step 1: Develop a crawler program to pick the best performer LLM in each size range from HuggingFace’s Open LLM leaderboard.
 
- [x] Step 2: Download and deploy each best performer. Conduct test runs to measure real-time resource usage and collect data.
 
- [x] Step 3: Develop a program to compute the available resource situation in the end user.
 
- [x] Step 4: Develop an algorithm to choose the most appropriate best performer we collect in Step 2.
 
In short, we would like to enable the edge user to have a “one-click” experience to select and download the best LLM model according to their local environment as well as task objectives.

# use this repo

`git clone <link_to_repo>`

`cd edge_choose-your-llm`

`pip install -r requirements.txt`

`python choose_best_model.py`