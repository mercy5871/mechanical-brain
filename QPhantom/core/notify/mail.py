import smtplib
import traceback
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from QPhantom.core.notify.notifier import Notifier

class MailNotifier(Notifier):
    '''
    目前只能用于文本内容的发送
    password 是发送邮箱的授权码，用来替代登录密码
    smtpserver 是发送邮箱的发送服务器
    eg：
        sender = 'q_phantom@163.com'
        receiver = 'duanhong@ppmoney.com'
        smtpserver = 'smtp.163.com'
        password = 'qphantom1'
    '''
    def __init__(self, sender, password, smtpserver, smtpport, to=None, project=None):
        super(MailNotifier, self).__init__(project)
        self.sender = sender
        self.password = password
        self.smtpserver = smtpserver
        self.smtpport = smtpport
        self.to = to
        self.collect = True

    @staticmethod
    def default(to=None, project=None):
        return MailNotifier(
            to=to,
            project=project,
            sender='q_phantom@163.com',
            password='qphantom1',
            smtpserver='smtp.163.com',
            smtpport=25
        )

    def set_receiver(self, to):
        self.to = to

    def send(self, subject=None, content=None):
        assert(self.to is not None)
        subject = subject if subject is not None else ""
        content = content if content is not None else ""
        log_buffer = self.log_buffer
        self.log_buffer = list()
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.sender
        msg['To'] = self.to
        msg.attach(MIMEText(f"<pre><code>{content}</code></pre>", "html"))
        for i, c in enumerate(log_buffer):
            if isinstance(c, dict):
                if c["type"] == "img":
                    msg.attach(MIMEText(f'<img src="cid:image{i}">', "html"))
                    msg_img = MIMEImage(c["body"], name=c["name"])
                    msg_img.add_header("Content-ID", f"<image{i}>")
                    msg.attach(msg_img)
                else:
                    msg.attach(MIMEText(f"<pre><code>UNKNOWN MSG with type: {type(c['body'])}</code></pre>", "html"))
            else:
                msg.attach(MIMEText(f"<pre><code>{c}</code></pre>", "html"))
        smtp = smtplib.SMTP(self.smtpserver, port=self.smtpport)

        smtp.login(self.sender, self.password)
        smtp.sendmail(self.sender, self.to, msg.as_string())
        smtp.quit()
