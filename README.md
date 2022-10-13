# Risky Choice with Optimal Priors

Research project with Mattar lab investigating the use of an optimally trained neural network used as a prior for predicting human behavior. using the choices13k dataset [1, 2, 3].
 
## wandb logging
This project uses Weights and Biases (wandb.ai) for logging. 
View the project [here](https://wandb.ai/brookeryan/choices13k-optimal/). 

# References 
Thanks to Joshua Peterson for providing the wonderful HURD library code. I copied the relevant files into the /hurd/ directory, and altered or extended a few functions to tailor it to the project.  

[4] used for determining how to hook up wandb logging to flax training loops. 

## Citations 
[1] Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., & Griffiths, T. L. (2021). Using large-scale experiments and machine learning to discover theories of human decision-making. Science, 372(6547), 1209-1214.

[2] Peterson, J. C., Human Risky Decision-Making (HURD) Toolkit, https://github.com/jcpeterson/hurd

[3] Peterson, J. C., choices13k Dataset, https://github.com/jcpeterson/choices13k

[4] Soumik Rakshit, Saurav Maheshkar, Writing a Training Loop in JAX + FLAX, https://colab.research.google.com/github/wandb/examples/blob/master/colabs/jax/Simple_Training_Loop_in_JAX_and_Flax.ipynb
