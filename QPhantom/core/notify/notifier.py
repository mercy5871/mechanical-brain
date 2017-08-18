import smtplib
import traceback
import functools
import logging
import socket
from datetime import datetime
from textwrap import dedent
from contextlib import contextmanager

class Notifier(logging.Handler):
    def __init__(self, project=None):
        logging.Handler.__init__(self)
        self.project = project
        self.log_buffer = list()
        self.logger = logging.getLogger(project)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        self.setFormatter(formatter)
        self.logger.addHandler(self)
        self.collect = False

    def get_title(self, level, label):
        subject_tmpl = '[{level}][{host}][{date_time}][{project}][{label}]'
        return subject_tmpl.format(
            level=level,
            date_time=datetime.now().strftime('%Y-%m-%d %X'),
            host=socket.gethostname(),
            project=self.project,
            label=label
        )

    def set_project(self, project):
        self.project = project
        self.logger.name = self.project

    def image(self, img, name="img"):
        self.log_buffer.append({"type": "img", "body": img, "name": name})

    def send(self, subject, content):
        print("warning: notifier: default print")
        print(subject)
        print(content)

    def error(self, label, msg, traceback):
        subject = self.get_title(level="ERROR", label=label)
        content = dedent('''\
            {msg}
            - - -
            {traceback}''').format(msg=msg, traceback=traceback)
        self.send(subject=subject, content=content)

    def warning(self, label, msg):
        subject = self.get_title(level="WARN", label=label)
        self.send(subject=subject, content=msg)

    def info(self, label, msg):
        subject = self.get_title(level="INFO", label=label)
        self.send(subject=subject, content=msg)

    @contextmanager
    def guardian(self, label, success_message='SUCCESS', fail_message='FAILED'):
        '''
        with statment support, eg.

        with notifier.guardian(label):
            your code here
        '''
        try:
            yield
            self.info(label, success_message)
        except Exception as e:
            traceback_msg = traceback.format_exc()
            fail_msg = '{fail_msg}: {err}'.format(fail_msg=fail_message, err=str(e))
            self.error(label, fail_msg, traceback=traceback_msg)

    def guard(self, label, success_message='SUCCESS', fail_message='FAILED'):
        '''
        function wrapper for guardian, eg.

        @notifier.guard(label)
        def your_function(*args, **kwargs):
            your func body
        '''
        def wrapper(func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self.guardian(label, success_message, fail_message):
                    return func(*args, **kwargs)
            return wrapped
        return wrapper

    def emit(self, record):
        msg = self.format(record)
        self.log_buffer.append(msg)
