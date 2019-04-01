Hamiltonian Neural Networks
=======
Sam Greydanus, Misko Dzamba, Jason Yosinski | 2019

![toy.png](static/toy.png)

Usage
--------

To train a Hamiltonian Neural Network (HNN):
 * Toy task (circular vector field + noise): `python3 train_toy.py`
 * Lipson data (simulated/real pendulum data): `python3 train_lipson.py --name {sim,real}`
 * Pixel observations (from OpenAI Gym): `python3 train_pixels.py`

To analyze results
 * Toy task: `python3 analyze_toy.py`
 * Lipson data: (coming soon)
 * Pixel observations: run the `analyze-pixels.ipnyb` notebook

Summary
--------

 * [Planning doc](https://docs.google.com/document/d/1WLprq600etYrqc51GLm5uTd2sTBeMYB5MUakJigCSEw/edit)
 * [Shared folder](https://drive.google.com/open?id=1869p7KJfOV5rI5HflTb7DmdnuSNbMyFU)

Modeling the conserved quantities of a physical system is one gateway to understanding its dynamics. Physicists use a mathematical object called the Hamiltonian to do this. They often use domain knowledge and trickery to write down the proper Hamiltonian, but here we take a different approach: we parameterize it with a differentiable model and then attempt to learn it directly from real-world data.

The HNN recipe
--------

1. Make a dataset of pixel-space observations of a physical system where energy is conserved. Here we're working with a pendulum.

![pendulum-dataset.gif.png](static/pendulum-dataset.gif.png)

2. Train an autoencoder on the dataset. This autoencoder is a but unusual - its latent representation gets fed to the HNN, which tries to model the system's dynamics in latent space.

![autoencoder-hnn.png](static/autoencoder-hnn.png)

3. Since the HNN uses the latent representation to model dynamics, we can think of the latent factors as being analogous to canonical coordinates (e.g. position and velocity).

![latents-hnn.png](static/latents-hnn.png)

4. [Phase space plots](https://en.wikiversity.org/wiki/Advanced_Classical_Mechanics/Phase_Space) are a common way to visualize Hamiltonians. We can make a phase space plot in the autoencoder's latent space. We can also integrate along the energy contours of phase space to predict the dynamics of a system (in the figure below, we intentionally "add energy" halfway through).

![integrate-latent-hnn.png](static/integrate-latent-hnn.png)

5. After integrating in latent space, we can project back into pixel space to simulate the dynamics of the system.

![pendulum-sim-hnn.gif.png](static/pendulum-sim-hnn.gif.png)

Dependencies
--------
 * OpenAI Gym
 * PyTorch
 * NumPy
 * ImageIO
 * Scipy

Known issues
--------
 * None so far \m/