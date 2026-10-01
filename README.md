Stochastic SIR epidemics on networks, built mainly using numpy.

## installation
the repository uses the uv package manager, it will be easiest to install if you have uv — if not, in pyproject.toml you can find all dependencies and manually install (not recommended)
uv will setup the environment and installation properly

git clone <url> && cd sir-graph-emm
uv sync 

## quickstart
there is an example simulation in notebooks — a more elaborate user guide will follow

## extra
graphs are stored as compressed matrix structures
node attributes will have their own class structure not touching the current simulation
