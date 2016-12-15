# coding: utf-8
##  @addtogroup blocks
# @{

##  @defgroup Server Server
# @{

## @file _server.py
# @brief This block allows to create a server to send data over network with sockets.
# @author Robin Siemiatkowski
# @version 0.1
# @date 11/07/2016

import os
import time
from multiprocessing import connection, Process

from _masterblock import MasterBlock


# def send_wrapper(f, input_, conn):
#     f(input_, conn)

def send(input_, conn):
    try:
        while True:
            if not input_.in_.closed:
                if input_.in_.poll():
                    data = input_.recv()
                    if data == 'close':
                        raise Exception("close instruction received from pipe")
                    else:
                        conn.send(data)
                if conn.poll():
                    data = conn.recv()
                    if data == 'close':
                        raise Exception("close instruction received from client")
            else:
                raise Exception("closed pipe")
        conn.close()
    except Exception as e:
        print "Exception in process n°{0}: {1}".format(os.getpid(), e)
        try:
            conn.send('close')
        except Exception as e:
            print e
            pass
        finally:
            print 'closing connection...'
            conn.close()

    except KeyboardInterrupt:
        print "KeyboardInterrupt received, link name: {0} (process n°{1}).".format(input_.name, os.getpid())
        conn.send('close')
        conn.close()
        pass
    except Exception as e:
        print 'Unexpected exception: ', e


class Server(MasterBlock):
    """
    Send data over network with sockets.
    """
    def __init__(self, ip="localhost", port=8888, time_sync=False):
        """
        Use it for testing.
        
        Parameters
        ----------
        port
        time_sync
        ip: IP address
        """
        super(Server, self).__init__()
        self.ip = ip
        self.port = port
        self.time_sync = time_sync
        self.sock = connection.Listener(address=(self.ip, self.port))
        if self.time_sync:
            try:
                c = self.sock.accept()
                c.recv()
                self.t0 = time.time()
                c.send(self.t0)
                c.recv()
                c.close()
                self.sock.close()
            except Exception as e:
                raise Exception("Cannot synchronize time with client: %s" % e)

    def main(self):
        try:
            self.sock = connection.Listener(address=(self.ip, self.port))
            conn = []
            procs = {}
            for input_ in self.inputs:
                conn.append(self.sock.accept())
            for i in range(len(conn)):
                name = conn[i].recv()
                for input_ in self.inputs:
                    if (input_.name == name):
                        procs[i] = Process(target=send, args=(input_, conn[i],))
                        procs[i].start()
            for i in range(len(procs)):
                procs[i].join()
            for i in range(len(conn)):
                if not conn[i].closed:
                    conn[i].close()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print "Exception in server: ", e
        except:
            print "Unexpected exception."
        finally:
            try:
                for i in range(len(procs)):
                    procs[i].join()
                for i in range(len(conn)):
                    if not conn[i].closed:
                        conn[i].close()
            except Exception as e:
                print "on exit: ", e
