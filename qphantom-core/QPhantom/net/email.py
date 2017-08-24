import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class MailSender(object):
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
    def __init__(self, sender, password, smtpserver, smtpport):
        self.sender = sender
        self.password = password
        self.smtpserver = smtpserver
        self.smtpport = smtpport
        self.collect = True

    @staticmethod
    def default():
        return MailSender(
            sender='notify@q-phantom.com',
            password='q-phantom_notify',
            smtpserver='smtp.q-phantom.com',
            smtpport=25
        )

    def send(self, to, subject, content, html=False):
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.sender
        msg['To'] = to
        msg.attach(MIMEText(content, "plain" if html == False else "html", "utf-8") )
        smtp = smtplib.SMTP(self.smtpserver, port=self.smtpport)
        smtp.login(self.sender, self.password)
        smtp.sendmail(self.sender, to, msg.as_string())
        smtp.quit()
