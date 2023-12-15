# Designing materials by laser powder bed fusion with machine learning-driven multi-objective optimization

An demonstration of multi-objective optmization for LPBF is given in the `multiobj_slm.ipynb` file.

There are two options how to run the demo code `multiobj_slm.ipynb`:

1) Run `multiobj_slm.ipynb` on a jupyter server in the Docker container. 
First build the Docker image 

```docker build -f Dockerfile -t multiobjslm .``` 

Next, run the Docker container forwarding the port 8888 of the container to 8888 port on the host 

```docker run --rm -p 8888:8888 multiobjslm jupyter notebook --allow-root --ip 0.0.0.0 --no-browser```

Now,`multiobj_slm.ipynb` is available via the url `http://localhost:8888/`.

2) Run `multiobj_slm.ipynb` using the environment in `env.yml` as a kernel for jupyter. Create conda environment with the command 

```conda env create -f env.yml```

Python 3.7 or newer is required.