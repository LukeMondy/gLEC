FROM jupyter/scipy-notebook:6d42503c684f
MAINTAINER Luke Mondy

ENV GRANT_SUDO yes
USER root

RUN DEBIAN_FRONTEND=noninteractive \
    && apt-get update -q \
    && apt-get install -y --no-install-recommends \
        libopenmpi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#RUN pip3 install --no-cache-dir packaging \

RUN conda install -c conda-forge \
        jigsaw \
        jigsawpy \
        meshio \
        mpi4py \
        rasterio

RUN cd work \
    && git clone https://github.com/LukeMondy/gLEC.git \
    && chown -R jovyan:users gLEC \
    && cd gLEC \
    && python3 setup.py install

CMD ["start.sh", "jupyter", "notebook", "--NotebookApp.token=''"]
