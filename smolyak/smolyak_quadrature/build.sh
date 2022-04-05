#!/bin/bash
#
#cp smolpack.h /$HOME/include
#
gcc -c -g smolyak.c
if [ $? -ne 0 ]; then
  echo "Errors compiling ccsmolyak.c."
  exit
fi
#

gcc -o smolyak_driver -g smolyak_driver.c smolyak.o -lm

