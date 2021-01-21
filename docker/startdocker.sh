progetto='/full/path/to/project/folder'

docker run -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --gpus all -v $progetto:/progetto monkey:Dockerfile

