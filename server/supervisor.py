"""Keep a worker process tree tied to the API process and execution lease."""
import argparse
import os
import signal
import subprocess
import time
from server.manager import Manager


def run(parent_pid, lease_fd, command):
    stopped=False
    def stop(signum, frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    if os.getppid()!=parent_pid:return 125
    worker=subprocess.Popen(command,start_new_session=True,pass_fds=(lease_fd,))
    try:
        while worker.poll() is None:
            if stopped or os.getppid()!=parent_pid:
                Manager.terminate(worker)
                return 143 if stopped else 125
            time.sleep(.2)
        return worker.returncode
    finally:
        if worker.poll() is None:Manager.terminate(worker)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-pid',required=True,type=int)
    parser.add_argument('--lease-fd',required=True,type=int)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:parser.error('Missing worker command')
    raise SystemExit(run(args.parent_pid,args.lease_fd,command))
