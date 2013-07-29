#
CC	= gcc
CFLAGS	= 
LIBS	= -lpthread

all	: gauss_seq 

clean	:
	rm -f gauss_seq *.o a.out core

gauss_seq: gauss_seq.c
	$(CC) $(CFLAGS) $(MAKEFLAGS) -o gauss_seq gauss_seq.c $(LIBS)

