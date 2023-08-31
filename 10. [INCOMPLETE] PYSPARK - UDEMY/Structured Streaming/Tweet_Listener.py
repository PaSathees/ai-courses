# # Tweet Listener class set up
# This is the more complex version for more advanced Python Programmers
# Please use the "simpler" version if you are a beginner to Python programming
# Source: https://ch-nabarun.medium.com/easy-to-play-with-twitter-data-using-spark-structured-streaming-76fe86f1f81c


import tweepy 
from tweepy import OAuthHandler # to authenticate Twitter API
from tweepy import Stream 
from tweepy.streaming import StreamListener
# These you don't need to pip install, they are native to Python
import socket 
import json 


# Twitter developer Credentials to connect to twitter account
# These are no longer active so you need to set up a Twitter Developer Account to get your own
access_token = "1139543220004773889-elE3LXEV7Vm6fX29YF2mHTz87zssEa"
access_secret = "khZBtbSsvyDMWPleFmJnP6BIEKKSQGjuNCnhIWgjQg0g9"
consumer_key = "bfcluTbm7SpMNpgcapL0PBoQF" # API key
consumer_secret = "wd0FuFxkNW6QpSm9x15SLcYMAUpm0cZsYF9w77J61vE0zAoBpa" # API secret key
# An access token used in authentication that allows you to pull specific data.
bearer_token = "AAAAAAAAAAAAAAAAAAAAAP4QNAEAAAAAZx4LTI1FjLw6TkBCevSkT6oO1yI%3DjUBdnAuec6Jb0YSOe9X0pV3uxvnQkUtuzffo25Bh0n5MyBUnyK"


class TweetsListener(StreamListener):
    # initialized the constructor
    def __init__(self, csocket):
        self.client_socket = csocket

    def on_data(self, data):
        try:
            # read the Twitter data which comes as a JSON format
            msg = json.loads(data)

            # the 'text' in the JSON file contains the actual tweet.
            # We will encode this with utf-8 which will clean out any emojis and stuff that may cause errors for us
            # We can come back and change this later on if we need to
            print(msg['text'].encode('utf-8'))

            # the actual tweet data is sent to the client socket
            self.client_socket.send(msg['text'].encode('utf-8'))
            return True

        except BaseException as e:
            # Error handling
            print("Ahh! Look what is wrong : %s" % str(e))
            return True
    # If there actually is an error, we will print the status
    def on_error(self, status):
        print(status)
        return True


# Now let's set up our connection using the access tokens we got from twitter
def sendData(c_socket):
    # authentication
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    # twitter_stream will get the actual live tweet data
    # This is a stream object
    twitter_stream = Stream(auth, TweetsListener(c_socket))
    # filter the tweet feeds related to "corona"
    twitter_stream.filter(track=['corona'])
    # in case you want to pass multiple criteria
    # twitter_stream.filter(track=['DataScience','python','Iot'])


if __name__ == '__main__':
    # create a socket object
    s = socket.socket()

    # Get local machine name : host and port
    host = "127.0.0.1"
    # You'll want to make sure this port is being used elsewhere, otherwise you'll get an error
    port = 3350

    # Bind to the port
    s.bind((host, port))
    print("Listening on port: %s" % str(port))

    # Wait and Establish the connection with client.
    s.listen(5)
    # This sends us back a tuple with the data and the addresss where it came from
    c, addr = s.accept()

    # Let's print it so we can confirm that when we are at the command line
    print("Received request from: " + str(addr))

    # Keep the stream data available
    sendData(c)

