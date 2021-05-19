#!/bin/bash

docker rm -fv pg4kamino 2>/dev/null

docker run --name pg4kamino -v /tmp:/tmp -e POSTGRES_DB=db4kamino -e POSTGRES_USER=kamino -e POSTGRES_PASSWORD=kamino -p 5432:5432 -d postgres:11


