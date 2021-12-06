FROM ubuntu:latest

SHELL ["/bin/bash", "-c"]

RUN mkdir -p /usr/src/Final
WORKDIR /usr/src/Final

# ensure everything is up to date
RUN apt update && apt upgrade -y

# Add Bazel source
RUN apt install -y apt-transport-https curl gnupg 
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg
RUN mv bazel.gpg /etc/apt/trusted.gpg.d/
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

# install the needed applications
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
RUN apt install -y python3-dev \
    		   texlive-full \
		   python3-pip \
		   gfortran \
		   libblas-dev \
		   liblapack-dev \
		   bazel

# install pip dependencies
RUN python3 -m pip install numpy \
    	       	   	   control \
			   typing \
			   slycot

# Build images
CMD ["/bin/bash", "./build.sh"]