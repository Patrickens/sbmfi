This is a small repository to Dockerize the _13cflux_ software.

### Building the Docker
From the root directory of pysumo execute the following command to create a Docker image named `13cflux`:
```
docker build -t 13cflux .
```
This means that the `Dockerfile` is used to make a Docker image with everything that is specified in the file.
Note that this requires a terminel (conda or MS PowerShell) with elevated (=administrator) privilages.
If a container with the name `13cflux` already exists, delete it first and then execute the command.


### Create image from container
With this command a new container is created named `FLUX`, and you get to the bash terminal inside:
```
docker run -it --entrypoint bash --name FLUX 13cflux
```

### Copying files from (Windows) host to a container
To copy a file, say `spiro.fml` to a folder `input` inside the image named `FLUX`, use the following command
```
docker cp C:\python_projects\pysumo\src\sumoflux\models\fml\spiro.fml FLUX:/input/spiro.fml
```

### Copying files from container to host (Windows)
To copy a file, say `spiro.fml` to a folder `input` inside the image named `FLUX`, use the following command
```
docker cp FLUX:/output/spiro_o.fml C:\python_projects\pysumo\C13Flux2\spiro.fml
```

### Exiting container
This can be done either from the `Docker` GUI or inside the terminal by typing `exit`

### Start existing docker container
```
docker start FLUX
```

### Bash terminal of an existing container
```
docker attach FLUX
```

## 13CFLUX2 commands container bash

These commands are to used to execute different parts of 13Cflux2. The `USER=nzamboni` 
prefix is necessary for all commands, since that is who the license is made out to. 

Get help for a command:
```
USER=nzamboni fwdsim --help
```

Wolfgang Wiechert & co are completely retarded, and thus we need to send a file back and forth
to some goddarn server. Guy is a complete wanker...
```
USER=nzamboni fmlsign -i /input/spiro.fml -o /signed/spiro.fml
```
TODO: figure out how to do this outomatically for every file coming into the /input folder; 
all files should be signed and deposited into the signed folder. 
The original file in input can subsequently be deleted.

To execute `fwdsim` inside `FLUX` from root
```
USER=nzamboni fwdsim -i /signed/spiro.fml -o /output/spiro_o.fml -s
```

# PowerShell single fml file processing
Open a windows PowerShell with admin rights. 
The command currently has two options, `-path` and `-image`, which are self-explanatory:

```
.\run_docker_on_file.ps1 -path "C:\python_projects\pysumo\src\sumoflux\models\fml\spiro.fml" -image "FLUX" -nconfig 3
```

To run the same script from a cmd terminal (with admin privelages) use:

```
powershell.exe ".\run_docker_on_file.ps1 -path 'C:\python_projects\pysumo\src\sumoflux\models\fml\spiro.fml' -image 'FLUX' -nconfig 3"
```