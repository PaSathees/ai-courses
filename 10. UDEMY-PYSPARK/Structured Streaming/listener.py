# This script is also commonly called server.py in many examples you will find online
import socket
import time

# First we set up our socket object
s = socket.socket()

# Tell the socket what your host is (it's set to local right now)
host = "127.0.0.1" # socket.gethostname() # or you could use this to get the hostname for you
# You'll want to make sure this port is not being used elsewhere, otherwise you'll get an error
# Typically ports 0-1023 are reserved for the operating system, so stay above that
port = 1234 

# Bind the host and port 
s.bind((host, port))
# Send a message so we know this actually happened
print("Listening on port: %s" % str(port))

# Listen for connections made to the socket
# The number 5 here is the size of the backlog
# That means that the listening socket will let 5 connection requests in pending state before they are accepted. 
# Each time a connection request is accepted, it is no longer in the pending backlog. 
s.listen(5)

# This sends us back a tuple with the data and the addresss where it came from
clientsocket, address = s.accept() # address is ip address
# Let's print it so we can confirm that when we are at the command line
print("Received request from: " + str(address)," connection created.")

###### Here is where things will start to look different in our Twitter example
data_stream = ["test1, ","test2, ","test3, ","test4, ","test5, "]
for data in data_stream:
    # convert to bytes
    print("Sending:", data)
    bytes_data = bytes(data, 'utf-8')
    clientsocket.send(bytes_data) # send to the client socket
    # We will sleep for 2 second here to demostrate how data can come faster than we are collecting it
    time.sleep(2)

clientsocket.close()