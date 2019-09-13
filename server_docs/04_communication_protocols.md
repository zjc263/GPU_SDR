Communication protocols
=======================

There are two differennt communications protocols implemented in the system on top the TCP protocol.
Server and client shares two TCP connection on two different sockets.

## Async protocol
The asynchrnous communication protocol is used to pass informations about system server and to request a measure to the server.
A typical example of usage is a mesurement request. As illustrated in the figure below, once the connection between the server and the client application has been established, the vlient API send a JSON string containing the anennae configuration and the DSP parameter; the server execute a small check on the parameters (to ensure hardware doability and to avoid segfault due to erroneus parameters values), propagate the necessary information to the USRP and runs the appropriate threads; if the check results positive a confirmation JSON string is sent to the API. After the measures end an other JSON packet is sent from the server to the client API in order to confirm the joining of the server-side threads and the availability for a new command.

###Client side
On the client side the communication protocol is handled via the receiver thread, waiting for incoming messages from the server and a 'parameter' class. The last is a class that allow to compose and send a measure request to the GPU server.

###Server side
The sevrer-side async communication is handled by a receiver thread and a send message function respectively implemented in #link and #link. The receiver thread listen for messages from the client API, interprets the resulting JSON string and pushes the result in a queue (#link) in the form of a command struct (#link). The queue is the checked by the server main loop which start the process requested by the user.

## Sync protocol
The synchronous protocol is tought only for data and metadata passing.

###Client side
Description on the client side protocol
###Server side
Description on the server side protocol
