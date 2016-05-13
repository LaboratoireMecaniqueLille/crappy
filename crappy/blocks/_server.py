# coding: utf-8
from _meta import MasterBlock
import time
from multiprocessing import connection, Pipe, Process
from ..links._link import TimeoutError
import os
import sys

class Server(MasterBlock):
    """
    Send a fake stream of data.
    """
    def __init__(self, ip="localhost", port=8888, time_sync=False):
        """
        Use it for testing.
        
        Parameters
        ----------
        ip: IP address of the 
        """
        self.ip = ip
        self.port = port
        self.time_sync = time_sync
        self.sock = connection.Listener(address="{0}:{1}".format(self.ip, self.port))
        if self.time_sync:
            try:
                c= self.sock.accept()
                c.recv()
                self.t0 = time.time()
                c.send(self.t0)
                c.close()
            except Exception as e:
                raise Exception("Cannot synchronize time with client: %s"%e)
    def send(self, input_, conn):
        try:
            while True:
                if(not input_.in_.closed):
                    if(input_.in_.poll()):
                        data = input_.recv()
                        if data == 'close':
                            break
                        else:
                            conn.send(data)
                    if(conn.poll()):
                        data = conn.recv()
                        if data== 'close':
                            raise Exception("close instruction received from client")
                            break
                else:
                    raise Exception("closed pipe")
                    break
            conn.close()
        except Exception as e:
            print "Exception in process n°{0}: {1}".format(os.getpid(), e)
            try:
                conn.send('close')
            except:
                print 'error'
                pass
            finally:
                print 'closing connection...'
                conn.close()
                
        except KeyboardInterrupt:
            print "KeyboardInterrupt received, link name: {0} (process n°{1}).".format(input_.name, os.getpid())
            conn.send('close')
            conn.close()
        except:
            print 'Unexpected exception'
    def main(self):
        try:
            conn=[]
            procs={}
            for input_ in self.inputs:
                conn.append(self.sock.accept())
            for i in range(len(conn)):
                name = conn[i].recv()
                for input_ in self.inputs:
                    if(input_.name == name):
                        procs[i]=Process(target=self.send,args=(input_, conn[i],))
                        procs[i].start()
            print("Done")
        except Exception as e:
            print "Exception in server: ", e
        except KeyboardInterrupt:
            pass
        except:
            print "Unexpected exception."
        finally:
            try:
                for i in range(len(conn)):
                    for i in range(len(procs)):
                        procs[i].join()
                    if not conn[i].closed:
                        conn[i].close()
            except Exception as e:
                print "On exit: ", e