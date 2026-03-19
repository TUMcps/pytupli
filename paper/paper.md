---
title: "PyTupli: Enabling Collaboration in Offline Reinforcement Learning"
authors:
  - name: Hannah Markgraf
    orcid: 0009-0006-4603-9420
    affiliation: 1
    corresponding: true
  - name: Michael Eichelbeck
    orcid: 0000-0002-1522-8767
    affiliation: 1
  - name: Daria Cappey
    affiliation: 1
  - name: Selin Demirtürk
    affiliation: 1
  - name: Yara Schattschneider
    affiliation: 1
  - name: Matthias Althoff
    orcid: 0000-0003-3733-842X
    affiliation: 1
affiliations:
  - name: Technical University of Munich, Germany
    index: 1
date: 10.12.2025
bibliography: paper.bib
tags:
  - Python
  - offline reinforcement learning
  - collaboration
  - datasets
  - infrastructure
---

# Summary

Offline reinforcement learning (RL) offers a powerful way to derive effective decision-making policies for control problems using pre-collected data. However, managing and sharing such datasets in collaborative research settings can be challenging, as proper versioning, filtering, and access control are required. PyTupli is a free, Python-based toolkit designed to support this workflow by providing an easy-to-use yet specialized framework that remains independent of external service providers. Unlike centralized platforms, PyTupli is designed for self-hosted deployment, giving research teams full control over their data infrastructure and access policies. Its client library allows users to serialize and store control problems, upload new data, and retrieve precisely the subsets they need through flexible and expressive filters. Built-in metrics help evaluate dataset coverage and utility, informing both dataset selection and algorithm design for offline RL. To ensure secure deployment, a container-based server component offers authentication, role-based access control, and automated certificate provisioning. Together, these capabilities enable researchers to create, manage, exchange, and analyze datasets for offline RL in a robust and accessible manner.

# Statement of need

Reinforcement learning (RL) provides powerful methods for decision-making under uncertainty, but training RL agents typically requires extensive interaction with the underlying system or a computationally expensive simulator. Offline RL alleviates this requirement by training agents solely on previously collected data [@lange2012batch]. Such datasets contain tuples of *state, action, next state,* and *reward*, obtained from real systems or simulators.

Effectively managing, sharing, and curating these datasets is essential for collaborative offline RL research, yet existing tools provide only partial support. Platforms like Zenodo or the free version of HuggingFace allow users to share finalized datasets but are not suitable for ongoing or private collaborations. Furthermore, they lack mechanisms for tracking dataset structure or performing efficient queries. Traditional databases are better suited but require expertise to design and maintain workflows.

PyTupli addresses this gap by providing a Python toolkit for creating, storing, and sharing tuple datasets for custom environments. We provide a Docker container that can be deployed independently to maintain control over data. Each dataset is associated with a benchmark, stored as JSON and linked to artifacts such as time-series data, hyperparameters, or trained policies. PyTupli includes integrated user and access management.

Since offline RL performance depends on dataset quality [@schweighofer2022dataset; @suttle2025behavioral; @asadulaev2025expert], PyTupli offers extensive filtering capabilities [@minari; @liu2023datasets]. PyTupli enables tuple-level filtering, allowing rebalancing datasets or selecting transitions from specific regions of the state space. In addition, PyTupli provides metrics for assessing dataset coverage and reward characteristics, which can predict performance [@schweighofer2022dataset; @asadulaev2025expert] and guide algorithm selection.


# State of the Field

Publicly available tuple datasets have been essential for advancing offline RL algorithms [@kumar2020conservative; @kostrikov2021offline]. These collections span domains such as robotics, games [@fu2020d4rl; @minari; @formanek2023off; @gulcehre2020rl], power systems [@qin2022neorl], and autonomous driving [@liu2023datasets; @lee2024ad4rl]. As offline RL matures, it is applied to more diverse, task-specific problems. However, no toolbox supports collaborative dataset creation and sharing for custom environments.

Minari [@minari] is the closest related tool, offering standardized datasets and functionality for filtering and recording interactions. However, it focuses on distributing datasets for established benchmarks. Although Minari permits users to request publication of custom datasets on its official platform, this approach is often unsuitable for ongoing or proprietary work or for datasets that evolve over time. PyTupli instead targets collaborative workflows for custom control tasks, enabling teams to share datasets without relying on a central server. Additionally, it provides tools for assessing dataset quality or coverage.

![Overview of the core functionalities of PyTupli. \label{fig:overview} ](pytupli_overview.pdf)

# Core Functionalities
We briefly describe the core functionalities of PyTupli which are illustrated in \autoref{fig:overview}.

**Benchmark and Artifact Management**: PyTupli enables users to store any control task defined as a gymnasium environment. Environments may include parameterizable configurations that produce variations in task dynamics. A fully specified environment is stored as a benchmark, providing a unique reference for reproducible evaluation of controllers. Benchmarks can reference additional data, such as exogenous inputs, time-series data, or pre-trained models. These external units, referred to as artifacts, are stored independently and linked to multiple benchmarks to avoid duplication.

**Data Management**: PyTupli supports ingesting, storing, and querying structured datasets (RL tuples), including their relation to existing benchmark problems and any relevant metadata. MongoDB serves as the backend, providing scalable storage and retrieval. \autoref{tab:times} shows how ingestion and retrieval times scale with dataset size.

**Multi-User Collaboration and Access Control:** PyTupli facilitates collaborative workflows through private, group, and public scopes. Based on their assigned role, users can store, retrieve, delete, and publish objects. A server-side backend with FastAPI provides a REST interface for secure, programmatic access, while token-based authentication ensures secure sharing across teams or organizations.

**Integration with Existing Offline RL Infrastructure:** An interface to the gymnasium framework enables users to record interactions with gymnasium environments as RL tuples. Furthermore, retrieved tuple datasets are made available in a form that can easily be converted into the dataset formats used by existing offline RL libraries such as d3rlpy [@seno2022d3rlpy].

**Assessment of Dataset Quality:** PyTupli provides metrics for coverage and expected returns to guide dataset selection and algorithm choice.


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

:Upload and download times for established datasets averaged over 10 runs. We chose two examples with low, medium, and high dataset size from the Minari collection. However, not only the size, but also the nature of observations has a strong influence on processing times.\label{tab:times}

# Quality Metrics

### Return-Based Metrics
Return-based metrics such as trajectory quality (TQ) [@schweighofer2022dataset] and average Q-value [@asadulaev2025expert] inform algorithm choice. High TQ favors behavioral cloning, while low TQ favors value-based methods. Estimated return improvement [@swazinna2021measuring] relates maximum and average returns. Average Q-value operates on tuple level and is a strong predictor of performance [@asadulaev2025expert]. It requires fitting a Q-function via Bellman updates.

### Coverage-Based Metrics
Dataset quality also depends on state-action coverage, where low coverage reduces performance [@schweighofer2022dataset]. A common approach approximates entropy via unique state-action pairs [@schweighofer2022dataset]. Behavioral entropy [@suttle2025behavioral] extends this idea using density-based weighting of state-action space regions.

# Software Design

PyTupli is designed around a clear separation between a lightweight client library and a centralized server backend. On the client side, PyTupli integrates via a Gymnasium environment wrapper, which is a well-established abstraction in the RL ecosystem. The server component exposes a REST API that implements functionality specific to offline RL datasets, such as structured storage of benchmarks, episodes, and artifacts, as well as flexible filtering and access control. A centralized service was favored over object storage or git-based workflows because offline RL datasets require domain-specific querying and metadata handling that these approaches do not natively support. Deployment is simplified via a Docker Compose setup, providing a production-ready stack that can be launched with minimal configuration. Here, the API server is hidden behind an Nginx web server acting as a reverse proxy for increased scalability. We chose MongoDB as the backend due to its ability to store heterogeneous, evolving data without rigid schemas while supporting efficient indexing for nested fields. GridFS enables integrated storage of large artifacts, avoiding external dependencies.

Existing tools such as Minari focus on distributing pre-existing datasets for offline RL. PyTupli instead targets research groups that need to create, host, and curate custom benchmarks collaboratively. This fundamental difference in scope and architecture made contributing to existing projects impractical, motivating the development of new software tailored to this use case.

# Research Impact Statement

Since its release on PyPI in May 2025, PyTupli has been downloaded approximately 65 times per month, suggesting early uptake. The project is actively maintained by two contributors with automated testing and CI/CD pipelines. Documentation is available via ReadTheDocs, and tutorials provide practical guidance. PyTupli has been integrated into the CommonPower framework, demonstrating applicability in real research settings.

# AI Usage Disclosure

Visual Studio Code with Claude Sonnet 4.5 was used for scaffolding tests. GPT 5.2 was used for refining the manuscript.

# Acknowledgements

This work was partially supported by the German Research Foundation (AL 1185/9-1).

# References
