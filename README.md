<div align="center">
  <img src="docs/source/pytupli_logo.svg" alt="PyTupli Logo">
</div>

# PyTupli

[![pipeline status](https://gitlab.lrz.de/cps/cps-power/pytupli/badges/main/pipeline.svg)](https://gitlab.lrz.de/cps/cps-power/pytupli/-/commits/main)
[![coverage report](https://gitlab.lrz.de/cps/cps-power/pytupli/badges/main/coverage.svg)](https://gitlab.lrz.de/cps/cps-power/pytupli/-/commits/main)
![Latest Release](https://gitlab.lrz.de/cps/cps-power/pytupli/-/badges/release.svg)

PyTupli is a Python library for creating, storing, and sharing benchmark problems and datasets for offline reinforcement learning (RL). PyTupli includes a lightweight client library with defined interfaces for uploading and retrieving benchmarks and data. It supports fine-grained filtering at both the episode and tuple level, allowing researchers to curate high-quality, task-specific datasets. A containerized server component enables production-ready deployment with authentication, access control, and automated certificate provisioning for secure use. By addressing key barriers in dataset infrastructure, PyTupli facilitates more collaborative, reproducible, and scalable offline RL research.

By using PyTupli, you can:

- ✅ Create benchmarks from any Gymnasium-compatible environment
- ✅ Share environments without exposing sensitive implementation details
- ✅ Record episode data from interactions with the environment or store static datasets associated to a benchmark
- ✅ Download datasets and convert them into formats compatible with popular offline RL libraries such as d3rlpy
- ✅ Store and manage artifacts like trained models or time series data

## Getting started

### Installation

You can install PyTupli using pip:

```bash
pip install pytupli
```

Or if you're using Poetry:

```bash
poetry add pytupli
```

For local development in editable mode,
navigate to the package directory and run
```bash
poetry install
```

### Optional Dependencies

PyTupli has several optional dependency groups that can be installed based on your needs:

> **Server Components**: To install dependencies for running the PyTupli server:
> ```bash
> poetry install --with server
> ```

> **Documentation**: To build the documentation:
> ```bash
> poetry install --with docs
> ```

> **Testing**: To run tests:
> ```bash
> poetry install --with tests
> ```

You can combine multiple groups:
```bash
poetry install --with server,docs,tests
```

## Deployment

For deployment instructions, please refer to the [deployment documentation](deployment/README.md).

## CLI Usage

PyTupli provides a command-line interface for the TupliAPIClient. After deployment, first log in to the server and specify the URL:

```bash
pytupli login --username your_username --password your_password --url http://your-server:port
```

The URL will then be remembered for all future interactions with the server. As an alternative to handing it over during login, you can call
```bash
pytupli set_url --url http://your-server:port
```

For user management, you can create new users and change passwords (requires admin privileges):
```bash
# Create a new user
pytupli signup --username new_user --password initial_password

# Change a user's password
pytupli change_password --username target_user --password new_password
```

Other useful utilities of the CLI are listing benchmarks or artifacts:
```bash
# List available benchmarks
pytupli list_benchmarks

# List episodes for a benchmark
pytupli list_artifacts
```

You can get detailed help on any command by using the --help flag:
```bash
# Show all available commands
pytupli --help

# Get help on a specific command
pytupli command_name --help
```

> **Note:** If you have IPython installed, it must be version <8.4 for the help functionality to work properly. This is due to a known issue that has been fixed in the underlying Fire library but is not yet available in the latest release.

## Basic Usage Example

PyTupli makes it easy to create and share reinforcement learning benchmarks and associated datasets for offline RL. Here's an example of how a collaborative offline RL project based on PyTupli might look like:

1.  Organization A has developed an environment for their specific use case (e.g., an energy management system). They have some historix data that they want to use to train an offline RL baseline.
2.  They wrap their environment using PyTupli's wrapper classes to standardize the interface
3.  They can then store and publish the benchmark through the PyTupli API.
4.  They upload the historic data as episodes associated to the newly-created benchmark, making it available to other organizations.
5.  Organization B can access the benchmark and download the data.
6.  Before training their algorithms on it, B can filter the dataset, for example, for data created during a specific time period.
7.  Finally, trained agents can be uploaded as artifacts associated to the benchmark

Code example (abbreviated):

```python
# Organization A: Instantiate API storage object
tupli_storage = TupliAPIClient()
tupli_storage.set_url("https://company-a-server.com/api")
# Instantiate gymnasium environment
custom_env = PowerSystemEnv()
# Wrap environment
tupli_env = TupliEnvWrapper(env=custom_env, storage=tupli_storage)
# Store and publish the benchmark
tupli_env.store(
    name='EMS_benchmark',
    description="Energy management system control task"
    )
tupli_env.publish()
# Load the historical data
historic_episodes = load_historic_data()
# Record and publish the episodes
for eps in historic_episodes:
    eps_item = Episode(
        benchmark_id=tupli_env.id,
        metadata=eps.metadata,
        tuples=eps.tuples
        )
    eps_header = tupli_storage.record(eps_item)
    tupli_storage.publish(eps_header.id)

# Organization B: Instantiate API storage object
tupli_storage = TupliAPIClient()
tupli_storage.set_url("https://company-a-server.com/api")
# We assume that this is the id of the previously stored benchmark
stored_id = "dl345kn456mlkl230"
# Download benchmark
loaded_tupli_env = TupliEnvWrapper.load(
    storage=tupli_storage,
    benchmark_id=stored_id
    )
# Create dataset containing all episodes recorded during the summer months
months = ["June", "July", "August"]
filter_summer = FilterOR(filters=[FilterEQ(key="month", value=m) for m in months])
filter_benchmark = FilterEQ(key='id', value=stored_id)
dataset_summer = TupliDataset(
    storage=tupli_storage
    ).with_benchmark_filter(filter_benchmark).with_episode_filter(filter_summer)
dataset_summer.load()
# Convert to d3rlpy dataset
obs, act, rew, term, trunc = dataset_summer.convert_to_numpy()
```
For a comprehensive guide covering most of PyTupli's functionality, including recording episodes, managing artifacts, and creating datasets, please refer to the Introduction.ipynb tutorial in the docs/source/tutorials directory.


## Documentation
A detailed documention of the client library as well as our tutorials are available on [ReadtheDocs](https://pytupli.readthedocs.io).

## Reference
PyTupli is maintained by the Cyber-Physical Systems Group at the Chair for Robotics and Embedded Systems at Technical University of Munich.

If you use PyTupli, please include the following reference
```
@article{markgraf2025pytupli,
  title={PyTupli: A Scalable Infrastructure for Collaborative Offline Reinforcement Learning Projects},
  author={Markgraf, Hannah and Eichelbeck, Michael and Cappey, Daria and Demirt{\"u}rk, Selin and Schattschneider, Yara and Althoff, Matthias},
  journal={arXiv preprint arXiv:2505.16754},
  year={2025}
}
```
