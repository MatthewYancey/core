from twitter import *

class direct_message(object):

    def __init__(self, text_message):

        access_token = '2157858768-wPO8dLiXGW8sPVQg3Bct1NdAGvwGFYLqyZVhY2M'
        access_token_secret = 'oK14HEuwT2g9blazr8NSReSMO2YP1b9eZ5Q3sbI2cnERh'
        consumer_key = '1PVrW5SonNbS1spTrffHRKZMK'
        consumer_secret = '0MCM2KTan3Pb83K4BQyRHG8adA2UCzZWNKgK59mbDrSDq84k1l'
        t = Twitter(auth = OAuth(access_token, access_token_secret, consumer_key, consumer_secret))

        t.direct_messages.new(user = 'mattcyancey', text = text_message)
