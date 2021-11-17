# TVM integration for HANNAH

To get the basic installation use:

1. Install `git-lfs` on ubuntu `apt install git-lfs`
2. checkout submodules: `git submodule update --init --recursive`
3. `poetry install`
4. `poetry shell`

For the tvm installation there are the following installation options.

## MicroTVM installation

This installation installs tvm with microtvm support without llvm backend

```
./scripts/install_micro.sh
./scripts/install_zephyr.sh
source env
```

Zephyr installation is optional but is currently needed for host driven execution.
For an example using host driven execution and the micro tvm zephyr runtime on stm32f429i discovery boards
see `samples/micro/zephyr_host_driven_stm32f429i.py`.

## MicroTVM with pulp-llvm support

This installation option activates the pulp-llvm based backend for direct vectorization on xpulpv targets.
For pulp targets no zephyr support is needed at the moment.

```
./scripts/install_micro_pulp.sh
```

## Full installation

The full installation uses the host provided llvm backend and activates cuda and llvm backends if available.

```
./scripts/install_full.sh
```


# Common error reasons

1. Pythonpath not set when using automate runner on schrank boards

  Add `AcceptEnv LANG LC_* PYTHONPATH` to `/etc/ssh/sshd_config` and restart server
