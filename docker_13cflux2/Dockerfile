FROM ubuntu:18.04

COPY C13FLUX2/15_Zamboni_Zurich_Switzerland_Ubuntu_18_04_64bit /c13install
COPY ../src/sumoflux/models/fml /input

# Copying license
RUN mkdir /etc/x3cflux
RUN cp /c13install/packete_lic_15/License/license.dat /etc/x3cflux/license.dat

RUN mkdir signed
RUN mkdir output

# Installing the Dependencies
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y vim
RUN apt-get install -y coinor-libipopt1v5 libblas3 libc6 libcolamd2 libcxsparse3 libgcc1 libgfortran3 libgmp10 libhdf5-dev
RUN apt-get install -y liblapack3 libmumps-seq-5.1.2 libssl1.0.0 libstdc++6 libxerces-c3.2 libsoap-lite-perl perl-tk libxml-libxml-perl
RUN apt-get install -y python-h5py python-numpy
RUN apt install -y gdebi

# Installing package
RUN gdebi -n /c13install/packete_lic_15/ubuntu64/x3cflux_2.1-4_amd64.deb
