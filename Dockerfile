FROM nvcr.io/nvidia/pytorch:23.03-py3

RUN pip install gdal rasterio pyproj shapely xarray python-eccodes dask netCDF4 bottleneck pandas seaborn

# RUN conda install -c conda-forge libstdcxx-ng==12.2.0
# RUN conda install -c conda-forge gdal rasterio pyproj shapely
# python-eccodes: xarray complains, "import cfgribs" throws "RuntimeError: Cannot find the ecCodes library", this should help
# dask netCDF4 bottleneck: optional dependencies of xarray
# RUN conda install -c conda-forge xarray python-eccodes dask netCDF4 bottleneck pandas seaborn
RUN pip install ipyleaflet jupytext
RUN pip install scikit-image pytorch_lightning geemap segmentation_models_pytorch earthengine_api geetools
RUN pip install ujson torchgeometry
RUN pip install yapf  # code formatting
RUN pip install hydra-core  # for config management
RUN pip install rich # nice progress bar
RUN pip install h5py # hdf5 file support
RUN pip install geopandas retry
# downgrade protobuf, as otherwise the following error is thrown
# TypeError: Descriptors cannot not be created directly.
# https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly
# likely finding the correct dependencies might be better than downgrading, but whatever
# RUN pip install protobuf==3.20.*

WORKDIR /ws

RUN pip install jupyter_contrib_nbextensions
RUN pip install ipython_unittest
RUN jupyter contrib nbextension install --system --Application.log_level=WARN
RUN jupyter nbextension enable toc2/main --system
RUN jupyter nbextension enable code_prettify --system

# COPY opt-jupyter_notebook_config.py /opt/conda/etc/jupyter/jupyter_notebook_config.py
# install config
#RUN mkdir -p /home/l91bthro/.jupyter/
#COPY jupyter_notebook_config.py /home/l91bthro/.jupyter/

ENV PYTHONPATH=$PYTHONPATH:/ws/dev/sen12tp/:/ws/dev/ndvi-prediction/
ENV HOME=/ws
ENV JUPYTER_PORT=9009
ENV JUPYTER_TOKEN=ahnaeziZ1mouthahxa6a
# an error about an incorrect library version of libstdc++ is raised
# when importing/using pandas
# the error is, that the system-included version is older than the
# version by conda. therefore force using the conda provided version
# ENV LD_PRELOAD=/opt/conda/lib/libstdc++.so


# enable ssh for root on port 20022 with key auth
# CMD ["jupyter", "notebook", "--no-browser", "--ip=0.0.0.0"]
