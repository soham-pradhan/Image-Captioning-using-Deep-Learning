#!C:\Users\Ashish\Anaconda3\python.exe
import sys
import socket
s = socket.socket()   
port = 12358
s.connect(('127.0.0.1', port))
file = sys.argv[1]
s.sendall(file.encode('utf-8'))
cap1 = s.recv(10240).decode('utf-8')
s.close()
if cap1!='':
    print(cap1)
