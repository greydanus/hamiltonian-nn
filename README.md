Hamiltonian Neural Nets for fun and profit.
=======
Sam Greydanus, Misko Dzamba, Jason Yosinski | 2019

Usage
--------

Experiments
 * To train HNN on pendulum tasks:
 	* Lipson data (simulated pendulum) `python3 hnn_lipson_task.py --name sim`
 	* Lipson data (real pendulum) `python3 hnn_pixel_task.py --name real`
 	* Pixel pendulum experiments coming soon...

About
--------

* [Planning doc](https://docs.google.com/document/d/1WLprq600etYrqc51GLm5uTd2sTBeMYB5MUakJigCSEw/edit)
* [Shared Drive folder](https://drive.google.com/open?id=1869p7KJfOV5rI5HflTb7DmdnuSNbMyFU)
Modeling the conserved quantities of a physical system is one gateway to understanding its dynamics. Physicists use a mathematical object called the Hamiltonian to do this. They often use domain knowledge and trickery to write down the proper Hamiltonian, but here we take a different approach: we parameterize it with a differentiable model and then attempt to learn it directly from real-world data.
