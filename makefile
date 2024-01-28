CC = gcc
CFLAGS = -I/Users/davidnordstrom/opt/anaconda3/include/python3.9 -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -fwrapv -O2 -Wall -fPIC -isystem /Users/davidnordstrom/opt/anaconda3/include -arch x86_64

your_program: micro_torch.c
	$(CC) -o micro_torch micro_torch $(CFLAGS)

run: micro_torch
	./micro_torch


clean:
	rm -f micro_torch
