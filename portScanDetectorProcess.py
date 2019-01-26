import multiprocessing
import time
from slips.core.database import __database__
import json

# Port Scan Detector Process
class PortScanProcess(multiprocessing.Process):
    """ 
    A class process to find port scans
    This should be converted into a module that wakesup alone when a new alert arrives
    """
    def __init__(self, inputqueue, outputqueue, config):
        multiprocessing.Process.__init__(self)
        self.inputqueue = inputqueue
        self.outputqueue = outputqueue
        self.config = config
        self.processname = 'portscan'

    def run(self):
        try:
            while True:
                if self.inputqueue.empty():
                    # Do stuff
                    try:
                        self.outputqueue.put('10|'+self.processname+'|['+self.processname+'] ' + 'Detecting port scans')
                        profiles = __database__.getProfiles()
                        for profileid in profiles:
                            self.outputqueue.put('10|'+self.processname+'|['+self.processname+'] ' + 'Profile: {}'.format(profileid))
                            lasttw = __database__.getLastTWforProfile(profileid)
                            lasttw_id, lasttw_time = lasttw[0]
                            dstips = __database__.getDstIPsfromProfileTW(profileid, lasttw_id)
                            # Convert to python data
                            dstips = json.loads(dstips)
                            self.outputqueue.put('10|'+self.processname+'|['+self.processname+'] ' + 'Tuples: {}'.format(dstips))
                    except Exception as inst:
                        self.outputqueue.put('01|'+self.processname+'|['+self.processname+'] ' + 'Error in run() of '+self.processname)
                        self.outputqueue.put('01|'+self.processname+'|['+self.processname+'] ' + '{}'.format(type(inst)))
                        self.outputqueue.put('01|'+self.processname+'|['+self.processname+'] ' + '{}'.format(inst))

                    time.sleep(2)

                elif not self.inputqueue.empty():
                    line = self.queue.get()
                    if 'stop' != line:
                        self.outputqueue.put('01|'+self.processname+'|['+self.processname+'] ' + 'Stopping the '+self.processname+ 'process')
                        return True
        except KeyboardInterrupt:
            self.outputqueue.put('01|'+self.processname+'|['+self.processname+'] ' + 'Stopping the '+self.processname+ 'process')
            return True
        except Exception as inst:
            self.outputqueue.put('01|'+self.processname+'|['+self.processname+'] ' + 'Error in '+self.processname)
            self.outputqueue.put('01|'+self.processname+'|['+self.processname+'] ' + '{}'.format(type(inst)))
            self.outputqueue.put('01|'+self.processname+'|['+self.processname+'] ' + '{}'.format(inst))
            return True
