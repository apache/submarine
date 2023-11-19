# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:18.04

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_PREFIX="/"
ARG NB_PORT=8888
ARG MLFLOW_TRACKING_URI="http://submarine-mlflow-service:5000"

USER root

SHELL ["/bin/bash", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    apt-transport-https \
    build-essential \
    curl \
    wget \
    git \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    run-one && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "$LOG_TAG Set locale" && \
    echo "LC_ALL=en_US.UTF-8" >> /etc/environment && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf && \
    locale-gen en_US.UTF-8

ENV NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_PREFIX=$NB_PREFIX \
    NB_PORT=$NB_PORT \
    CONDA_DIR=/opt/conda \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI

# Create NB_USER user with UID=1000 and in the 'users' group
RUN useradd -M -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown -R ${NB_USER}:users /usr/local/bin && \
    mkdir -p $HOME && \
    chown -R ${NB_USER}:users ${HOME} && \
    mkdir -p ${CONDA_DIR} && \
    chown -R ${NB_USER}:users ${CONDA_DIR} && \
    chmod g+w /etc/passwd

# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN mv /tini /usr/local/bin/tini && chmod +x /usr/local/bin/tini

# Install conda
USER $NB_UID
ARG PYTHON_VERSION=default
# update conda version to 23.10.0-1
ENV MINICONDA_VERSION=23.10.0-1 \
    MINICONDA_SHA256=d0643508fa49105552c94a523529f4474f91730d3e0d1f168f1700c43ae67595 \
    CONDA_VERSION=23.10.0

WORKDIR /tmp
# repo in https://repo.anaconda.com/miniconda/
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py311_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${MINICONDA_SHA256} *Miniconda3-py311_${MINICONDA_VERSION}-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash Miniconda3-py311_${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py311_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda config --system --prepend channels conda-forge && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda config --system --set channel_priority flexible && \
    if [ ! $PYTHON_VERSION = 'default' ]; then conda install --freeze-installed --yes python=$PYTHON_VERSION; fi && \
    conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    conda init bash && \
    source ~/.bashrc && conda activate && \
    # it is used for jupyter lab build
    conda install --freeze-installed -c conda-forge nodejs jupyterlab==4.0.8 && \
    # todo: jupyterlab 4.0 was not supported before jupyterlab-git 0.5.0, so we temporarily used 0.50.0rc0. 
    #       More details can be found in the issue:
    #       https://github.com/jupyterlab/jupyterlab-git/issues/1245
    pip --no-cache-dir install --pre "jupyterlab-git==0.50.0rc0" && \
    conda install --freeze-installed -c conda-forge cvxpy && \
    jupyter lab build && \
    # remove node_modules
    rm -rf /home/$NB_USER/.cache/yarn && \
    rm -rf /opt/conda/share/jupyter/lab/staging/node_modules/* && \
    # uninstall nodejs, nodejs should not be used after jupyter lab has been built
    conda uninstall nodejs -y && \
    # clear conda to remove index cache, lock files, unused cache packages, and tarballs in /opt/conda/pkgs
    conda clean -a -y

### We removed the qlib dependency because the project currently does not have good support for python 3.9 and above. 
### If you need this project, you can choose the py38 or py39 miniconda and rebuild the secondary image
# RUN pip --no-cache-dir install pyqlib==0.9.3
# RUN wget https://raw.githubusercontent.com/microsoft/qlib/v0.9.3/examples/workflow_by_code.ipynb -P $HOME

# Install latest sumbarine python sdk and notebook
RUN git clone --depth=1 https://github.com/apache/submarine && \
    pip --no-cache-dir install submarine/submarine-sdk/pysubmarine[tf,pytorch] && \
    # Add DeepFM example into notebook
    cp submarine/submarine-sdk/pysubmarine/example/submarine_experiment_sdk.ipynb $HOME && \
    cp -r submarine/submarine-sdk/pysubmarine/example/{data,deepfm_example.ipynb,deepfm.json} $HOME && \
    rm submarine -rf

ENV MLFLOW_S3_ENDPOINT_URL http://submarine-minio-service:9000
ENV AWS_ACCESS_KEY_ID submarine_minio
ENV AWS_SECRET_ACCESS_KEY submarine_minio

EXPOSE $NB_PORT
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]
COPY --chown=${NB_USER}:users start-notebook.sh /usr/local/bin
WORKDIR ${HOME}
