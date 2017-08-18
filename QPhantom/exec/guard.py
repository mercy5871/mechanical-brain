from QPhantom.core.notify import MailNotifier
import shlex
import sys, getopt
import subprocess

noti = MailNotifier.default(to='duanhong@ppmoney.com', project='Test')

opts, args = getopt.getopt(sys.argv[1:], 'p::t::l::e::', ["project=", "to=", "label="])

cmd = ''
label = 'DailyDataUpdate'

for op, value in opts:
    if op == "--project":
        project = value if value is not None else 'Test'
    elif op == "--to":
        to = value if value is not None else 'duanhong@ppmoney.com'
    elif op == "--label":
        label = value if value is not None else 'DailyDataUpdate '
    else:
        raise Exception('args_error command need args : --to')

for arg in args:
    cmd = cmd + arg + ' '

with noti.guardian(label):
    if cmd is None:
        raise Exception('no command to exe')
    noti.set_project(project)
    noti.set_receiver(to)
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = proc.stdout.decode() if proc.stdout is not None else ''
    stderr = proc.stderr.decode() if proc.stderr is not None else ''
    print(stdout, file=sys.stdout)
    print(stderr, file=sys.stderr)
    noti.log_buffer.append(stdout)
    noti.log_buffer.append('--------------------------------------')
    noti.log_buffer.append(stderr)

    return_code = proc.returncode

    if return_code > 0:
        raise Exception('wrong exit code = ' + str(return_code))
    else:
        noti.logger.info('success')
