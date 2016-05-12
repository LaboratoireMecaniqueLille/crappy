# coding: utf-8
from _meta import MasterBlock
import time
from multiprocessing import connection, Pipe, Process
from ..links._link import TimeoutError
import os

class Client(MasterBlock):
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
        try:
            if self.time_sync:
                c= connection.Client("{0}:{1}".format(self.ip, self.port))
                c.send('go')
                t0_serv = c.recv()
                t_client = time.time()
                delta = t_client-t0_serv
                self.t0 = t0_serv+delta
                #c.send('go')
                #for i in range(10):
                    #time.sleep(2)
                    #print 'time{0}: {1}'.format(i, time.time()-self.t0)
                c.close()
        except Exception as e:
            raise Exception("Cannot synchronize time with server: %s"%e)
        
    def recv(self, output_, conn):
        try:
            conn.send(output_.name)
            while True:
                if(not output_.out_.closed):
                    if(conn.poll()):
                        data = conn.recv()
                        if(data!='close'):
                            output_.send(data)
                        else:
                            raise Exception("close instruction received from server")
                            break
                else:
                    raise Exception("Closed pipe")
                    break
        except Exception as e:
            print "Exception in process n°{0}: {1}".format(os.getpid(), e)
            try:
                try:
                    conn.sendall('close')
                except:
                    pass
                conn.close()
            except Exception as e1:
                print "Exception ", e1
            except IOError as ioe:
                print "IOError: ", ioe
            except:
                print "Unexpected exception."
        
        except KeyboardInterrupt:
            print "KeyboardInterrupt received in process {0} (pid:{1}).".format(output_.name, os.getpid())
            try:
                conn.send('close')
                conn.close()
            except Exception as e:
                print "Exception::", e
        except:
            print "Unexpected exception."
    def main(self):
        try:
            conn=[]
            procs={}
            i=0
            for output_ in self.outputs:
                conn.append(connection.Client("{0}:{1}".format(self.ip, self.port)))
                procs[i]=Process(target=self.recv,args=(output_, conn[i],))
                procs[i].start()
                i=i+1
            for i in range(len(procs)):
                procs[i].join()
            print("Done")
        except Exception as e:
            print "Exception in Client: ", e
        except KeyboardInterrupt:
            pass
        except:
            print "Unexpected exception."
        finally:
             for i in range(len(conn)):
                    if not conn[i].closed:
                        conn[i].close()