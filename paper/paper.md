---
title: "PyTupli: A Lightweight Infrastructure for Collaborative Offline Reinforcement Learning Projects"
authors:
  - name: Hannah Markgraf
    orcid: 0009-0006-4603-9420
    affiliation: 1
    corresponding: True
  - name: Michael Eichelbeck
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Daria Cappey
    affiliation: 1
  - name: Selin Demirtürk
    affiliation: 1
  - name: Yara Schattschneider
    affiliation: 1
  - name: Matthias Althoff
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Technical University of Munich, Germany
    index: 1
date: 20.11.2025
bibliography: paper.bib
tags:
  - Python
  - offline reinforcement learning
  - collaboration
  - datasets
  - infrastructure
---

# Summary

Offline reinforcement learning (RL) offers a powerful way to derive effective decision-making policies for control problems using data that has already been collected. However, managing and sharing such datasets in collaborative research settings can be challenging, as proper versioning, filtering, and access control are essential for reproducibility and reliable experimentation. PyTupli is a free, Python-based toolkit designed to support this workflow by providing an easy-to-use yet specialized infrastructure that remains independent of external service providers. Its client library allows users to serialize and store control problems, upload new data, and retrieve precisely the subsets they need through flexible and expressive filtering tools. Built-in metrics help evaluate dataset coverage and utility, informing both dataset selection and algorithm design for offline RL. A container-based server component offers authentication, role-based access control, and automated certificate provisioning, enabling secure deployment without added operational complexity. Together, these capabilities enable researchers to create, manage, exchange, and analyze datasets for offline RL in a robust and accessible manner.

# Statement of need
Reinforcement learning (RL) provides powerful methods for decision-making under uncertainty, but training RL agents typically requires extensive interaction with the underlying system or a computationally expensive simulator. Because the success of machine learning often depends on large datasets, offline RL has emerged as a paradigm that alleviates this requirement by training agents solely on previously collected data [@lange2012batch]. Such datasets consist of tuples *(state, action, next state, reward)* obtained from historical trajectories or generated through simulators.

Effectively managing, sharing, and curating these datasets is essential for collaborative offline RL research, yet existing tools provide only partial support. Version control systems such as GitHub offer basic sharing and provenance features but are not designed for large datasets and lack mechanisms for tracking internal dataset structure or performing efficient, fine-grained queries. Traditional databases are better suited for this purpose but require substantial expertise to design and maintain robust workflows.

PyTupli addresses this gap by providing a dedicated Python toolkit for creating, storing, and sharing tuple datasets for custom environments built on the gymnasium framework [@towers2024gymnasium]. Through containerized deployment, PyTupli allows researchers to host their own server with a concise API for uploading, downloading, and distributing benchmarks and their corresponding tuple datasets. Benchmarks are stored as JSON-serialized objects and can be linked to related artifacts, including time-series data, algorithm hyperparameters, or trained policies, allowing multiple benchmarks to reference shared resources. Because it is built for scalable collaboration, PyTupli includes integrated user and access-management features.

Since the performance of offline RL algorithms often depends critically on the quality of the underlying dataset [@schweighofer2022dataset; @suttle2025behavioral; @asadulaev2025expert], PyTupli offers extensive filtering capabilities. Whereas established offline RL datasets primarily support filtering by entire episodes [@minari; @liu2023datasets], PyTupli enables tuple-level filtering as well. This can be used, for example, to rebalance datasets with sparse rewards or selectively include transitions from specific regions of the state space. In addition, PyTupli provides a suite of metrics for assessing dataset coverage and reward characteristics, which can serve as predictors of offline RL performance [@schweighofer2022dataset; @asadulaev2025expert] and help guide algorithm selection.

# Related Software
Publicly available tuple datasets have been essential for advancing offline RL algorithms [@kumar2020conservative; @kostrikov2021offline]. These curated collections span domains such as robotics and games [@fu2020d4rl; @minari; @formanek2023off; @gulcehre2020rl], power system control [@qin2022neorl], and autonomous driving [@liu2023datasets; @lee2024ad4rl], and are typically designed to support the development of improved offline RL methods. As offline RL techniques mature, they are being applied to increasingly diverse and task-specific control problems. Yet, to the best of our knowledge, no existing toolbox supports the collaborative creation, management, and sharing of datasets for custom environments. Minari [@minari] is the closest related tool, offering a repository of standardized datasets along with functionality for filtering, environment reconstruction, and recording new interactions. Its focus, however, remains on distributing datasets for established benchmarks.

PyTupli instead targets collaborative workflows for custom control tasks. It enables researchers to share datasets and benchmarks directly within project teams without dependence on a central public server. Although Minari permits users to request publication on its official platform, this approach is often unsuitable for ongoing or proprietary work or for datasets that evolve over time. In addition, Minari does not provide tools for assessing dataset quality or coverage, which PyTupli includes to support informed dataset curation and algorithm selection in offline RL research.

![Overview of the core functionalities of PyTupli. \label{fig:overview} ](pytupli_overview.pdf)

# Core Functionalities
We briefly describe the core functionalities of PyTupli which are illustrated in \autoref{fig:overview}.

**Benchmark and Artifact Management**: PyTupli enables users to store any control task defined as a gymnasium environment. Environments may include parameterizable configurations that produce variations in task dynamics. A fully specified environment is stored as a benchmark, providing a unique reference for reproducible evaluation of controllers. Benchmarks can reference additional data such as exogenous inputs, time-series data, or pre-trained models. These external units, referred to as artifacts, are stored independently and linked to multiple benchmarks to avoid duplication.

**Data Management**: PyTupli supports ingesting, storing, and querying structured datasets (RL tuples), including their relation to existing benchmark problems and any relevant metadata. MongoDB serves as the backend, providing scalable storage and efficient retrieval for large datasets. \autoref{tab:times} shows how ingestion and retrieval times scale with dataset size.

**Multi-User Collaboration and Access Control:** PyTupli facilitates collaborative workflows through private, group, and public scopes. Based on their assigned role, users can store, retrieve, delete, and publish objects. A server-side backend with FastAPI provides a REST interface for secure, programmatic access, while token-based authentication ensures secure sharing across teams or organizations.

**Integration with Existing Offline RL Infrastructure:** An interface to the gymnasium framework enables users to record interactions with gymnasium environments as RL tuples. Furthermore, retrieved tuple datasets are made available in a form that can easily be converted into the dataset formats used by existing offline RL libraries such as d3rlpy [@seno2022d3rlpy].

**Assessment of Dataset Quality:** PyTupli implements metrics to evaluate dataset quality in terms of coverage and expected returns. These metrics inform dataset selection and can provide guidance for algorithm choice. Detailed formulations are provided in the following section.

| Dataset                | Size    | $M$ | $N$  | Upload (s) | Download (s) |
| ---------------------- | ------- | --- | ---- | ---------- | ------------ |
| **D4RL**               |         |     |      |            |              |
| door/human-v2          | 3.5MB   | 25  | 7K   | 0.83       | 0.24         |
| hammer/human-v2        | 6.2MB   | 25  | 11K  | 1.11       | 0.40         |
| antmaze/medium-play-v1 | 605.2MB | 1K  | 1M   | 173.26     | 126.62       |
| **Atari**              |         |     |      |            |              |
| pitfall/expert-v0      | 351.7MB | 10  | 65K  | 18.56      | 16.51        |
| **Mujoco**             |         |     |      |            |              |
| ant/expert-v0          | 1.92GB  | 2K  | 2M   | 64.17      | 29.65        |
| humanoid/expert-v0     | 2.95GB  | 1K  | 999K | 96.61      | 55.32        |
|                        |         |     |      |            |              |
:Upload and download times for established datasets averaged over 10 runs. We chose two examples with low, medium, and high dataset size from the Minari collection. However, not only the size, but also the nature of observations has a strong influence on processing times.\label{tab:times}

# Quality Metrics

### Return-Based Metrics
Return-based metrics such as trajectory quality (TQ) [@schweighofer2022dataset] or the average Q-value [@asadulaev2025expert] can inform algorithm decision. For example, @schweighofer2022dataset show that behavioral cloning (BC) performs well despite its simplicity if the dataset has a high TQ. For datasets with low TQ, algorithms from the DQN family perform well in their experiments as they do not constrain the learned policy towards the distribution of the behavioral policy. TQ normalizes the average return of a dataset with respect to the returns obtained by a minimal performant and an expert policy. To provide similar insights without relying on such additional information, estimated relative return improvement (ERI) [@swazinna2021measuring] relates the maximum trajectory return in the dataset to its average return. While ERI and TQ operate on a trajectory level, average Q-value estimation offers insights on the tuple level, making it a better predictor of offline RL performance  [@asadulaev2025expert]. It requires fitting a Q-function using Bellman updates, which is closely related to the objectives used in offline RL training. However, for continuous action spaces, the user needs to provide an evaluation policy to predict the next actions in the Bellman target. Such a policy can, for example, be obtained using BC on the dataset.
### Coverage-Based Metrics
An important question when assessing a dataset is whether the behavioral policy (or policies) used to generate it did explore the state and action space well enough to learn a meaningful target policy from the data. A common approach for quantifying explorativeness is to approximate the entropy of transition probabilities for the behavior policy. For discrete state and action spaces, @schweighofer2022dataset suggest to approximate this by counting unique state-action pairs. Optionally, this value can be normalized using a reference dataset $\mathcal{D}_\text{ref}$ of same size, for example, the replay buffer collected during online training. @schweighofer2022dataset show that low SACo values hinder performance of a large variety of algorithms. While SACo aims at estimating the Shannon entropy of the transition probabilities, @suttle2025behavioral suggest that datasets that maximize their proposed behavioral entropy (BE) metric support better offline RL performance. They suggest a $k$-nearest-neighbor estimator of the true BE that relies on density-based weighting of different regions in the state-action space.

# Acknowledgements

This work was partially supported by the German Research Foundation (AL 1185/9-1).

# References
