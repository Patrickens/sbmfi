This is a small repository to Dockerize the _pta_

### Building the Docker
Some very vague explanation is given from the [gurobi website](https://hub.docker.com/r/gurobi/optimizer)
The container _Dockerfile_ for gurobi we get from [github](https://github.com/Gurobi/docker-optimizer/blob/master/9.5.2/Dockerfile).
We edit this container with the rest of the dependencies for _pta_ such as a C++ compiler.
From the _docker_pta_ directory where this `Dockerfile` is saved, we build the container as follows:
```
docker build -t pta . 
```
This means that the `Dockerfile` is used to make a Docker container with everything that is specified in the file.
Note that this requires a terminel (conda or MS PowerShell) with elevated (=administrator) privilages.

### gurobi
For gurobi to work, we need a remote licence that works from a docker. 
90 day licences are available from the [licence portal](https://portal.gurobi.com/iam/licenses/list).
You get a gurobi.lic file with `WLSSECRET, WLSSECRET` and `LICENSEID` codes.
These are the ones that are used below in docker run, they are set as environment variables.

### Create image from container
With this command a new container is created named `PTAG`, and you get to the bash terminal inside:
```
docker run --volume=C:/python_projects/pysumo/docker_pta/gurobi.lic:/opt/gurobi/gurobi.lic:ro -it --entrypoint bash --name PTA pta
```

```
docker cp test_pta.py PTA:/test_pta.py
```


### Start existing docker container
```
docker start PTA
```

### Bash terminal of an existing container
```
docker attach PTA
```

Below powershell command runs all the things necessary to make a tfs model in sequence, 
by default arguments are optional, but `path` and `image` are strictly necessary
```
powershell.exe ".\run_pta.ps1 -path 'C:\python_projects\pysumo\src\sumoflux\models\sbml\e_coli_tomek.xml' 
-image 'PTA' -conc_prior 'gluc_aero'"
```


To get volesti to work:
```
curl -sSL https://install.python-poetry.org | python3 -
git clone https://github.com/GeomScale/dingo
cd dingo
wget -O boost_1_76_0.tar.bz2 https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.bz2
tar xjf boost_1_76_0.tar.bz2
git submodule update --init
sudo apt-get install libsuitesparse-dev
```


```
powershell.exe ".\run_volesti.ps1 -path 'pol.p' -image 'PTA'"
```