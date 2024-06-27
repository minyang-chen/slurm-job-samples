# Using Docker under Slurm
Ask Slurm to run job in a docker container.


### Sample test job submission
```
$ docker image ls
$ docker run --rm docker/welcome-to-docker
```
specify resource limits
```
docker run --rm --cpus=6 --memory=1024m docker/welcome-to-docker
```

note: `--rm` means self clean up, remove the container up on completion.

