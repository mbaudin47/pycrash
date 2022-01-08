#!/bin/bash
#
#cp smolpack.h /$HOME/include
#
gcc -c -g ccsmolyak.c
if [ $? -ne 0 ]; then
  echo "Errors compiling ccsmolyak.c."
  exit
fi
#
gcc -c -g smolyak.c
if [ $? -ne 0 ]; then
  echo "Errors compiling smolyak.c."
  exit
fi
#
ar rcv libsmolpack.a *.o

gcc -o smolpack_interactive smolpack_interactive.c libsmolpack.a -lm

