# coding: utf-8
import os
import time
from multiprocessing import connection, Process

from _meta import MasterBlock


# def recv_wrapper(f, output_, conn):
#     f(output_, conn)

def recv(output_, conn):
    try:
        conn.send(output_.name)
        while True:
            if not output_.out_.closed:
                if conn.poll():
                    data = conn.recv()
                    if data != 'close':
                        output_.send(data)
                    else:
                        raise Exception("close instruction received from server")
            else:
                raise Exception("Closed pipe")

    except Exception as e:
        print "Exception in process n°{0}: {1}".format(os.getpid(), e)
        try:
            try:
                conn.sendall('close')
            except Exception as e:
                print e
            conn.close()
        except IOError as ioe:
            print "IOError: ", ioe
        except Exception as e1:
            print "Exception ", e1

    except KeyboardInterrupt:
        print "KeyboardInterrupt received in process {0} (pid:{1}).".format(output_.name, os.getpid())
        pass
        try:
            conn.send('close')
            conn.close()
        except Exception as e:
            print "Exception::", e
    except:
        print "Unexpected exception."


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
        super(Client, self).__init__()
        self.ip = ip
        self.port = port
        self.time_sync = time_sync
        try:
            if self.time_sync:
                c = connection.Client((self.ip, self.port))
                c.send('go')
                t0_serv = c.recv()
                t_client = time.time()
                delta = t_client - t0_serv
                self.t0 = t0_serv + delta
                c.send('ok')
                c.close()
        except Exception as e:
            raise Exception("Cannot synchronize time with server: %s" % e)

    def main(self):
        try:
            conn = []
            procs = {}
            i = 0
            for output_ in self.outputs:
                conn.append(connection.Client((self.ip, self.port)))
                procs[i] = Process(target=recv, args=(output_, conn[i],))
                procs[i].start()
                i += 1
            for i in range(len(procs)):
                procs[i].join()
            for i in range(len(conn)):
                if not conn[i].closed:
                    conn[i].close()

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print "Exception in server: ", e
        finally:
            try:
                for i in range(len(procs)):
                    procs[i].join()
                for i in range(len(conn)):
                    if not conn[i].closed:
                        conn[i].close()
            except Exception as e:
                print "on exit: ", e
