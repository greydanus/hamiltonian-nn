Hamiltonian Neural Networks
=======
Sam Greydanus, Misko Dzamba, Jason Yosinski | 2019

* Paper: [arxiv.org/abs/1906.01563](https://arxiv.org/abs/1906.01563)
* Blog: [greydanus.github.io/2019/05/15/hamiltonian-nns/](https://greydanus.github.io/2019/05/15/hamiltonian-nns/)

![overall-idea.png](static/overall-idea.png)

Basic usage
--------

To train a Hamiltonian Neural Network (HNN):
 * Task 1: Ideal mass-spring system: `python3 experiment-spring/train.py --verbose`
 * Task 2: Ideal pendulum: `python3 experiment-pend/train.py --verbose`
 * Task 3: Real pendulum (from this [Science](http://science.sciencemag.org/content/324/5923/81) paper): `python3 experiment-real/train.py --verbose`
 * Task 4: Two-body problem: `python3 experiment-2body/train.py --verbose`
 * Task 4b: Three-body problem: `python3 experiment-3body/train.py --verbose`
 * Task 5: Pixel pendulum (from OpenAI Gym): `python3 experiment-pixels/train.py --verbose`

To analyze results
 * Task 1: Ideal mass-spring system: [`analyze-spring.ipnyb`](analyze-spring.ipynb)
 * Task 2: Ideal pendulum: [`analyze-pend.ipnyb`](analyze-pend.ipynb)
 * Task 3: Real pendulum: [`analyze-real.ipnyb`](analyze-real.ipynb)
 * Task 4: Two-body problem: [`analyze-2body.ipnyb`](analyze-2body.ipynb)
 * Task 4b: Three-body problem: [`analyze-3body.ipnyb`](analyze-3body.ipynb)
 * Task 5: Pixel pendulum: [`analyze-pixels.ipnyb`](analyze-pixels.ipynb)

Summary
--------

Even though neural networks enjoy widespread use, they still struggle to learn the basic laws of physics. How might we endow them with better inductive biases? In this paper, we draw inspiration from Hamiltonian mechanics to train models that learn and respect exact conservation laws in an unsupervised manner. We evaluate our models on problems where conservation of energy is important, including the two-body problem and pixel observations of a pendulum. Our model trains faster and generalizes better than a regular neural network. An interesting side effect is that our model is perfectly reversible in time.


The HNN recipe
--------

1. Make a dataset of pixel-space observations of a physical system where energy is conserved. Here we're working with a pendulum.

![pendulum-dataset.gif.png](static/pendulum-dataset.gif.png)

2. Train an autoencoder on the dataset. The autoencoder is a bit unusual: its latent representation gets fed to the HNN, which tries to model the system's dynamics in latent space.

![autoencoder-hnn.png](static/autoencoder-hnn.png)

3. Since the HNN uses the latent representation to model dynamics, we can think of the latent factors as being analogous to canonical coordinates (e.g. position and velocity).

![latents-hnn.png](static/latents-hnn.png)

4. [Phase space plots](https://en.wikiversity.org/wiki/Advanced_Classical_Mechanics/Phase_Space) are a common way to visualize Hamiltonians. We can make a phase space plot in the autoencoder's latent space. We can also integrate along the energy contours of phase space to predict the dynamics of the system (in the figure below, we intentionally "add energy" halfway through).

![integrate-latent-hnn.png](static/integrate-latent-hnn.png)

5. After integrating in latent space, we can project back into pixel space to simulate the dynamics of the system.

![pendulum-compare-labeled.gif](static/pend-compare.gif)

Here's what it looks like when we add energy halfway through the simulation:

![pendulum-compare-labeled.gif](static/addenergy.gif)

Modeling larger systems
--------

We can also model larger systems -- systems with more than one pair of canonical coordinates. The two-body problem, for example, has four coordinate pairs.

![orbits-compare.gif](static/orbits-compare.gif)


Numbers
--------

### Train loss
* Choose data of the form `x=[x0, x1,...]` and `dx=[dx0, dx1,...]` where `dx` is the time derivative of `x`
* Let `dx' = model.time_derivative(x)`
* Compute L2 distance between `dx` and `dx'`

|               	| Baseline NN 		| Hamiltonian NN 	|
| ------------- 	| :---------------: | :---------------: |
| Ideal mass-spring |  	3.7134e-02 +/- 1.9143e-03 | 3.6933e-02 +/- 1.9128e-03 		|
| Ideal pendulum 	|   3.2606e-02 +/- 1.7434e-03 | 3.2787e-02 +/- 1.7567e-03 		|
| Real pendulum 	|   2.7455e-03 +/- 2.0735e-04 | 9.2376e-03 +/- 5.0263e-04 		|
| 2 body problem 	|   3.2682e-05 +/- 1.2022e-06 | 2.9959e-06 +/- 6.5500e-08		|
| 3 body problem 	|   9.5573e-02 +/- 6.5610e-02 | 8.0346e-02 +/- 2.1470e-02		|
| Pixel pendulum 	|   1.7731e-04 +/- 2.4202e-06 | 1.8683e-04 +/- 2.4238e-06 		|


### Test loss
Do the same thing with test data

|               	| Baseline NN 			| Hamiltonian NN 	|
| ------------- 	| :-------------------: | :---------------: |
| Ideal mass-spring |  	3.6656e-02 +/- 1.8652e-03 | **3.5928e-02 +/- 1.8328e-03** 	|
| Ideal pendulum 	|   **3.5273e-02 +/- 1.7970e-03** | 3.5586e-02 +/- 1.8178e-03 	|
| Real pendulum 	|   **2.1864e-03 +/- 3.3296e-04** | **5.9584e-03 +/- 6.1798e-04** 	|
| 2 body problem 	|   2.9575e-05 +/- 8.8900e-07 | **2.8218e-06 +/- 4.2020e-08** 	|
| 3 body problem 	|   **3.8000e-01 +/- 4.1612e-01**  	| 4.8809e-01 +/- 4.7745e-01	|
| Pixel pendulum 	|   **1.7306e-04 +/- 3.2413e-06**   | 1.8451e-04 +/- 3.3422e-06 |

### Energy MSE
* Choose a trajectory `[x0, x1,...]` from test data
* Use RK4 integration to estimate `[x0', x1',...]` using the model
* Compute the L2 distance between `[energy(x0), energy(x1),...]` and `[energy(x0'), energy(x1'),...]`

|               	| Baseline NN 			| Hamiltonian NN	|
| ------------- 	| :-------------------:	| :---------------:	|
| Ideal mass-spring | 1.7077e-01 +/- 2.06e-02  | **3.8416e-04 +/- 6.53e-05** |
| Ideal pendulum 	| 4.1519e-02 +/- 9.62e-03  | **2.4852e-02 +/- 5.42e-03** |
| Real pendulum 	| 3.8564e-01 +/- 6.92e-02  | **1.4477e-02 +/- 4.65e-03**	|
| 2 body problem 	| 6.3276e-02 +/- 3.36e-02  | **3.8751e-05 +/- 5.04e-06** |
| 3 body problem 	| 1.0906e+02 +/- 7.74e+01  | **4.1926e-02 +/- 3.39e-02**	|
| Pixel pendulum 	| 9.2748e-03 +/- 1.14e-03  | **1.5315e-04 +/- 8.42e-06** 	|


Dependencies
--------
 * OpenAI Gym
 * PyTorch
 * NumPy
 * ImageIO
 * Scipy
 
This project is written in Python 3.