# TVM integration for HANNAH

To get the basic installation use: 

1. `poetry shell`
2. `poetry install`
3. checkout submodules: `git submodules update --init --recursive`

For the tvm installation there are the following installation options. 

## MicroTVM installation

This installation installs tvm with microtvm support without llvm backend

```
./scripts/install_micro.sh
./scripts/install_zephyr.sh
```

Zephyr installation is optional but is currently needed for host driven execution. 

## MicroTVM with pulp-llvm support

This installation option activates the pulp-llvm based backend for direct vectorization on xpulpv targets. 
For pulp targets no zephyr support is needed at the moment. 

```
./scripts/install_micro_pulp.sh
```

## Full installation

The full installation uses the host provided llvm backend and activates cuda and opencl if available. 

```
./scripts/install_full.sh
```


# Common error reasons

1. Pythonpath not set when using automate runner

  Add `AcceptEnv LANG LC_* PYTHONPATH` to `/etc/ssh/sshd_config` and restart server
