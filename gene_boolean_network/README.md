## Gene Regulatory Boolean Network
Our approach for enhancing exploration and convergence within RL-dFBA with gene regulation lies in utilization of attractor states derived from a gene regulatory network (GRN). A GRN represents the intricate web of interactions between the genes, where nodes represent individual genes and edges depict their regulatory influences. By simplifying this network into a boolean model, where genes can be either on or off, we can identify stable configurations of gene expression known as attractor states. The proposition here focuses on employing these attractor states as the initial population for a genetic algorithm (GA) used within the RL-dFBA framework with gene regulation. 

Random initialization, commonly used in GAs, can lead to exploration of a vast and potentially irrelevant search space. By leveraging attractor states as starting points, exploration becomes directed towards regions of the search space with welldefined regulatory configurations and their corresponding metabolic outcomes. This targeted approach has the potential to significantly reduce the time required to converge to a solution.

Attractor states represent more biologically feasible configurations of gene expression. Utilizing them ensures that the GA operates within a biological relevant space. The inherent regulatory logic embedded within attractor states can be beneficial during the evolutionary process. These states represent a pre-defined subset of gene expression patterns that may already lead to desirable metabolic outcomes. By starting from these configurations, the GA can potentially leverage this built-in regulatory logic to identify optimal solutions for efficiently.

We have considered two subsets of gene regulatory networks which are:
* ITP metabolic reaction sub-network

| ---------------- | ------------------------------------------- | ---------------------------------- |

| Reaction IDs     | Reaction Names                              | Regulating Genes (IDs)             |

| ---------------- | ------------------------------------------- | ---------------------------------- |

| `NTP 10`         | Nucleoside Triphosphatase                   | `b0474`                            |

| `ATPHs`          | ATP adenine hydrolysis                      | `s0001`                            |

| `ADK4`           | Adenylate Kinase                            | `b2954`                            |

| `NTPP 9`         | Nucleoside Triphosphate Pyrophosphorylase   | `b4161 or b4394`                   |

| ---------------- | ------------------------------------------- | ---------------------------------- |

* GTP metabolic reaction sub-network

| ---------------- | ------------------------------------------- | ---------------------------------- |

| Reaction IDs     | Reaction Names                              | Regulating Genes (IDs)             |

| ---------------- | ------------------------------------------- | ---------------------------------- |

| `GTPtex`         | GTP transport via diffusion                 | `b0929 or b1377 or b2215 or b0241` |

| `NTP3pp`         | Nucleoside Triphosphatase                   | `b0929 or b1377 or b2215 or b0241` |

| `GDPtex`         | GDP transport via diffusion                 | `b0980`                            |

| ---------------- | ------------------------------------------- | ---------------------------------- |