

# EE510 Take Home Final

This is a take home final for EE510. In making this I found myself following good software practices such as 
DRY(Don't Repeat Yourself). In addition due to the fact that one of the dependencies, Slycot, required some 
base installation software I decided to move the system onto docker in order to have complete control over the
system and allow others to be able to build it with minimum issues


## Setup

To set up this system you need two applications: Docker and Docker-Compose. The first is to make the isolated 
system(which you can think of a superlight vm), the second is to make it easier to start the former repeatedly 
and record all options in a file.

Chances are when you download docker you will get compose, but just in case both links are provided

Simply download the tools from the appropriate websites:

 - [Docker](https://docs.docker.com/get-docker/)
 - [Docker Compose](https://docs.docker.com/compose/install/)


After that you simply need to be in the root directory of the projectopen a (preferably bash) terminal and run:

``` bash
docker-compose up --build
```
