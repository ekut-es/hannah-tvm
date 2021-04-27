# TVM backend for HANNAH

To install use:

1. activate HANNAH poetry shell
2. checkout submodules: `git submodules update --init --recursive`
3. Install backend:  `./install.sh`


# Common error reasons

1. Pythonpath not set when using automate runner

  Add `AcceptEnv LANG LC_* PYTHONPATH` to `/etc/ssh/sshd_config` and restart server
