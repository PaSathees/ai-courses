import socket
import time
# This socket wants to connect
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# s.connect((socket.gethostname(),1234))
s.connect(('127.0.0.1',1234))

while True:
    msg = s.recv(1024) # this is the size of data we want to recieve
    print(msg.decode("utf-8"))
    time.sleep(5)