from multiprocessing import Process
import sys
import time

def say_hello(name='world'):
    print("Hello, %s" % name)
    print('Starting:', p.name, p.pid)
    sys.stdout.flush()
    print('Exiting :', p.name, p.pid)
    sys.stdout.flush()
    time.sleep(20)

p = Process(target=say_hello)
p.start()
# no p.join()