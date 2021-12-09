FROM ubuntu:latest

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /usr/src/Final
WORKDIR /usr/src/Final

# ensure everything is up to date
RUN apt update && apt upgrade -y

# install the needed applications
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
RUN apt install -y python3-dev \
    		   texlive-full \
		   python3-pip \
		   gfortran \
		   libblas-dev \
		   liblapack-dev

# install pip dependencies
RUN python3 -m pip install numpy \
    	       	   	   control \
			   typing \
			   slycot \
			   pygments \
			   pytest

# Build images
CMD ["/bin/bash", "./build.sh"]